import random, os
import argparse
import json
import datetime as dt
import numpy as np

import torch, torchaudio

import params 
from model.tts_model import ZeroVoiceModel
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
import torchaudio as ta
from librosa import magphase, pyin
import librosa
import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

def stft(
    *,
    y: np.ndarray = None,
    fft_size: int = None,
    hop_length: int = None,
    win_length: int = None,
    pad_mode: str = "reflect",
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Librosa STFT wrapper.

    Check http://librosa.org/doc/main/generated/librosa.stft.html argument details.

    Returns:
        np.ndarray: Complex number array.
    """
    return librosa.stft(
        y=y,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        pad_mode=pad_mode,
        window=window,
        center=center,
    )

def compute_energy(y: np.ndarray, **kwargs) -> np.ndarray:
    """Compute energy of a waveform using the same parameters used for computing melspectrogram.
    Args:
      x (np.ndarray): Waveform. Shape :math:`[T_wav,]`
    Returns:
      np.ndarray: energy. Shape :math:`[T_energy,]`. :math:`T_energy == T_wav / hop_length`
    Examples:
      >>> WAV_FILE = filename = librosa.example('vibeace')
      >>> from TTS.config import BaseAudioConfig
      >>> from TTS.utils.audio import AudioProcessor
      >>> conf = BaseAudioConfig()
      >>> ap = AudioProcessor(**conf)
      >>> wav = ap.load_wav(WAV_FILE, sr=ap.sample_rate)[:5 * ap.sample_rate]
      >>> energy = ap.compute_energy(wav)
    """
    x = stft(y=y, **kwargs)
    mag, _ = magphase(x)
    energy = np.sqrt(np.sum(mag**2, axis=0))
    return energy

def compute_f0(
    *,
    x: np.ndarray = None,
    pitch_fmax: float = None,
    pitch_fmin: float = None,
    hop_length: int = None,
    win_length: int = None,
    sample_rate: int = None,
    stft_pad_mode: str = "reflect",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Compute pitch (f0) of a waveform using the same parameters used for computing melspectrogram.

    Args:
        x (np.ndarray): Waveform. Shape :math:`[T_wav,]`
        pitch_fmax (float): Pitch max value.
        pitch_fmin (float): Pitch min value.
        hop_length (int): Number of frames between STFT columns.
        win_length (int): STFT window length.
        sample_rate (int): Audio sampling rate.
        stft_pad_mode (str): Padding mode for STFT.
        center (bool): Centered padding.

    Returns:
        np.ndarray: Pitch. Shape :math:`[T_pitch,]`. :math:`T_pitch == T_wav / hop_length`

    Examples:
        >>> WAV_FILE = filename = librosa.example('vibeace')
        >>> from TTS.config import BaseAudioConfig
        >>> from TTS.utils.audio import AudioProcessor
        >>> conf = BaseAudioConfig(pitch_fmax=640, pitch_fmin=1)
        >>> ap = AudioProcessor(**conf)
        >>> wav = ap.load_wav(WAV_FILE, sr=ap.sample_rate)[:5 * ap.sample_rate]
        >>> pitch = ap.compute_f0(wav)
    """
    assert pitch_fmax is not None, " [!] Set `pitch_fmax` before caling `compute_f0`."
    assert pitch_fmin is not None, " [!] Set `pitch_fmin` before caling `compute_f0`."

    f0, voiced_mask, _ = pyin(
        y=x.astype(np.double),
        fmin=pitch_fmin,
        fmax=pitch_fmax,
        sr=sample_rate,
        frame_length=win_length,
        win_length=win_length // 2,
        hop_length=hop_length,
        pad_mode=stft_pad_mode,
        center=center,
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.1,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )
    f0[~voiced_mask] = 0.0

    return f0


def pre_process_f0_energy (file_path):

    audio, sr = ta.load(file_path)
    if audio.shape[0] > 1:
        audio = audio[0].unsqueeze(0)
    audio = ta.functional.resample(audio, orig_freq= sr, new_freq=22050)#.cpu().detach().numpy()
    mel = mel_spectrogram(audio, params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                            params.win_length, params.f_min, params.f_max, center=False).squeeze()

    audio = audio.squeeze().detach().cpu().numpy()

    f0 = compute_f0( 
        x= audio,
        pitch_fmax = 640,
        pitch_fmin = 1,
        hop_length =  params.hop_length ,
        win_length  =  params.win_length,
        sample_rate = params.sample_rate,
        stft_pad_mode = "reflect",
        center = True)

    energy = compute_energy(audio)

    f0 = torch.from_numpy(f0) 
    energy = torch.from_numpy(energy)
    
    return mel, f0, energy


def tts_inference (text, mel_, f0_, energy_) :
    print(text)
    with torch.no_grad():

        x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols)))[None]
        x_lengths = torch.LongTensor([x.shape[-1]])
        
        mel = torch.zeros((1, params.n_feats, mel_.shape[-1] + 1), dtype=torch.float32)
        f0 = torch.zeros((1, f0_.shape[-1]), dtype=torch.float32)
        energy = torch.zeros((1, energy_.shape[-1]), dtype=torch.float32)

        mel[0, :, : mel_.shape[-1]] = mel_
        f0[0, : f0_.shape[-1]] = f0_
        energy[0, : energy_.shape[-1]] = energy_


        t = dt.datetime.now()
        y_enc, y_dec, attn, audio = model.forward(x, x_lengths, 50, mel, 
                                                torch.LongTensor( mel.shape[-1]) , f0, energy,
                                                temperature=1.5,
                                                stoc=False,  length_scale=1.0, fast_sampling = True)
        t = (dt.datetime.now() - t).total_seconds()
        print(f'Time: {t * 22050 / (y_enc.shape[-1] * 256)}')
        
        ta.save('out.wav', audio.cpu(), sample_rate=24000)

        return audio #'out.wav'


nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)
cmu = cmudict.CMUDict('./resources/cmu_dictionary')
model_ckpt_path = 'logs/zero_voice/grad_16.pt'

model = ZeroVoiceModel(nsymbols, params.n_enc_channels,
                    params.filter_channels, params.filter_channels_dp, 
                    params.n_heads, params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, 
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
model.load_state_dict(torch.load(model_ckpt_path, map_location=lambda loc, storage: loc), strict=False)
_ = model.cuda().eval()
print(f'Number of parameters: {model.nparams}')

ref_speech_path = '/workspace/dm_datasets/LibriTTS/train-clean-100/7190/90543/7190_90543_000005_000001.wav'
mel, f0, energy = pre_process_f0_energy (ref_speech_path)
text = 'What sort of evidence is there?'
tts_inference (text, mel, f0, energy)

