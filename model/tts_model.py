# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random
import numpy as np
import torch

from model import monotonic_align
from model.base import BaseModule
from model.acoustic_feature_encoder import AcousticFeatureEncoder
from model.diffusion_model import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
import torch.nn.functional as F

import sys, json
#sys.path.append('./model/diffwave/')
from model.diffwave.model import DiffWave
from model.diffwave.params import AttrDict, params
from torchvision.transforms import v2

class ZeroVoiceModel(BaseModule):
    def __init__(self, n_vocab,  n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(ZeroVoiceModel, self).__init__()
        self.n_vocab = n_vocab
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.acoustic_feature_encoder = AcousticFeatureEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
            
        self.mel_decoder = Diffusion(n_feats, dec_dim, beta_min, beta_max, pe_scale)

        beta = np.array(params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = torch.nn.L1Loss()
        self.wav_decoder = DiffWave(params)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, y_ref, y_ref_lengths, f0, energy,
                temperature=1.5, stoc=False, spk=None, length_scale=1.0, fast_sampling = True):

        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths, y_ref, y_ref_lengths, f0, energy = self.relocate_input([x, x_lengths, y_ref, y_ref_lengths, f0, energy])
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask  = self.acoustic_feature_encoder(x, x_lengths, y_ref,  f0, energy)

        #print(mu_x.shape,logw.shape)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        acoustic_feature_encoder_outputs = mu_y[:, :, :y_max_length]
        
        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature 
        # Generate sample by performing reverse dynamics
        mel_decoder_outputs, xt = self.mel_decoder(z, y_mask, mu_y, n_timesteps,  stoc, spk)
        #decoder_outputs = decoder_outputs[:, :, :y_max_length]

        fast_sampling = fast_sampling

        training_noise_schedule = np.array(self.wav_decoder.params.noise_schedule)
        inference_noise_schedule = np.array(self.wav_decoder.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        audio = torch.randn(mel_decoder_outputs.shape[0], self.wav_decoder.params.hop_samples * mel_decoder_outputs.shape[-1], device=mel_decoder_outputs.device)
        audio = audio / temperature
        
        for n in range(len(alpha) - 1, -1, -1):
            
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * self.wav_decoder(audio, torch.tensor([T[n]], device=audio.device), mel_decoder_outputs).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)

        return acoustic_feature_encoder_outputs, mel_decoder_outputs, attn[:, :, :y_max_length], audio
    
    def spec_f0_energy_mask (self, mel, f0, energy):

        for i in range (0,2):

            mask_length = np.random.randint(mel.shape[2] // 4, mel.shape[2] // 2 + 1, size=1)[0]
            start_idx = np.random.randint(0, mel.shape[2]-mask_length, size=1)[0]
            end_idx = start_idx + mask_length + 1
            mel = torch.cat((mel[:,:,:start_idx] , mel[:,:,end_idx:] ), dim=-1)

            f0 = torch.cat((f0[:,:start_idx] , f0[:,end_idx:] ), dim=-1)
            energy = torch.cat((energy[:,:start_idx] , energy[:,end_idx:] ), dim=-1)
        
        #print(mu_y.shape,f0.shape,energy.shape)

        factor = np.random.uniform(0.5, 1.5, size=1)[0]
        mu_y = v2.Resize(size = (mel.shape[1], int (mel.shape[-1]*factor))) (mel)
        
        f0 = f0.unsqueeze(1)
        f0 = v2.Resize(size = ( f0.shape[1], int (f0.shape[-1]*factor))) (f0)
        energy = energy.unsqueeze(1)
        energy = v2.Resize(size = ( energy.shape[1], int (energy.shape[-1]*factor))) (energy)

        #print(mu_y.shape,f0.shape,energy.shape)

        return  mu_y, f0.squeeze(1), energy.squeeze(1)
    

    def compute_loss(self, x, x_lengths, y, y_lengths, audio, f0, energy, out_size=None):

        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths, f0, energy = self.relocate_input([x, x_lengths, y, y_lengths, f0, energy])

        y_ref_masked, f0_masked, energy_masked = self.spec_f0_energy_mask(y, f0, energy)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask  = self.acoustic_feature_encoder(x, x_lengths, y_ref_masked, f0_masked, energy_masked )

        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            audio_cut = torch.zeros(audio.shape[0], out_size * 256, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                audio_cut[i, :y_cut_length * 256] = audio[i, cut_lower * 256 :cut_upper * 256]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask
            audio = audio_cut

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        prior_loss = F.mse_loss( y, mu_y, reduction='mean') 

        if y_mask.shape[2] != y.shape[2]:
            print(y_mask.shape, y.shape)
            y_mask = torch.ones(y.shape[0],1,y.shape[2]).cuda()

        diff_loss_mel, xt , mean = self.mel_decoder.compute_loss(y, y_mask, mu_y)

        N, T = audio.shape
        self.noise_level = self.noise_level.to( audio.device)

        t = torch.randint(0, len(self.wav_decoder.params.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

        predicted = self.wav_decoder(noisy_audio, t, y)
        diff_loss_wav = self.loss_fn(noise, predicted.squeeze(1))

        return dur_loss, prior_loss, diff_loss_mel, diff_loss_wav
