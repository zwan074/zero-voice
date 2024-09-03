import numpy as np
from tqdm import tqdm
import argparse

import torch, torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model.tts_model import ZeroVoiceModel
from data import  ZeroVoiceBatchCollate, ZeroVoiceDataset
from utils import plot_tensor, save_plot, save_plot_f0
from text.symbols import symbols
from scipy.io.wavfile import write

import sys, json
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
HIFIGAN_CONFIG = './hifi-gan/checkpts/config.json'
HIFIGAN_CHECKPT = './hifi-gan/checkpts/g.pt' 

nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cntn', '--continue_train', type=bool, required=False, default=False, help='continue train?')
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=False, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-i', '--starting_epoch', type=int, required=False, default=1, help='starting epoch')
    args = parser.parse_args()

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=params.log_dir)

    print('Initializing data loaders...')
    train_dataset = ZeroVoiceDataset ( params.train_filelist_path, params.cmudict_path, params.add_blank, 
                                        params.n_fft, params.n_feats, params.sample_rate, 
                                        params.hop_length, params.win_length, 
                                        params.f_min, params.f_max )
    #train_dataset, _ = torch.utils.data.random_split(train_dataset, [len(train_dataset)//8, len(train_dataset) - len(train_dataset)//8])
    print (len(train_dataset))

    batch_collate = ZeroVoiceBatchCollate()
    loader = DataLoader(dataset= train_dataset, batch_size=params.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    test_dataset = ZeroVoiceDataset ( params.valid_filelist_path, params.cmudict_path, params.add_blank, 
                                      params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                      params.win_length, params.f_min, params.f_max)
    print('Initializing model...')


    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    model = ZeroVoiceModel(nsymbols, params.n_enc_channels,
                    params.filter_channels, params.filter_channels_dp, 
                    params.n_heads, params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, 
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    
    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)

    i = 0
    for item in test_batch:
        mel = item['y']
        i += 1 
        #logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{params.log_dir}/original_{i}.png')

    if args.continue_train : 
        print('Loading previous model...')
        model.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc), strict=False)
    
    print(f'Number of parameters: {model.nparams}')
    model.cuda()


    print('Start training...')
    iteration = 0
    for epoch in range(args.starting_epoch, params.n_epochs + 1):
        model.eval()
        print('Synthesis...')

        with torch.no_grad():
            i = 0 
            for item in test_batch:
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                i += 1 
                
                y_ref = item['y'].unsqueeze(0).cuda()
                y_ref_lengths = torch.LongTensor([y_ref.shape[-1]]).cuda()

                f0 = item['f0'].unsqueeze(0).cuda()
                energy = item['energy'].unsqueeze(0).cuda()

                y_enc, y_dec, attn, audio  = model(x, x_lengths, 50, y_ref, y_ref_lengths , f0, energy )

                torchaudio.save(
                     f'{params.log_dir}/audio_{i}.wav', audio.cpu(), sample_rate=24000
                 )
                
                audio_hg = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                write(f'{params.log_dir}/audio_hg_{i}.wav', 24000, audio_hg)

                save_plot(y_enc.squeeze().cpu(), 
                          f'{params.log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), f'{params.log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{params.log_dir}/alignment_{i}.png')

                save_plot_f0 (audio.squeeze().cpu(), f'{params.log_dir}/audio_{i}.png')

        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses_mel = []
        diff_losses_wav = []

        with tqdm(loader, total=len(train_dataset)//params.batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()

                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                audio = batch['audio'].cuda()
                f0 = batch['f0'].cuda()
                energy = batch['energy'].cuda()

                dur_loss, prior_loss, diff_loss_mel, diff_loss_wav = model.compute_loss(x, x_lengths,
                                                                        y, y_lengths,audio,f0, energy,
                                                                        params.out_size)
                loss = sum([dur_loss, prior_loss,  diff_loss_mel, diff_loss_wav])
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                
                optimizer.step()
                
                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss_mel: {diff_loss_mel.item()}, diff_loss_wav: {diff_loss_wav.item()}'
                progress_bar.set_description(msg)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses_mel.append(diff_loss_mel.item())
                diff_losses_wav.append(diff_loss_wav.item())
                iteration += 1

        msg = 'Epoch %d: duration loss = %.5f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.5f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.5f' % np.mean(diff_losses_mel)
        msg += '| diffusion loss2 = %.5f\n' % np.mean(diff_losses_wav)

        with open(f'{params.log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % params.save_every > 0:
            continue
        
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{params.log_dir}/grad_{epoch}.pt")
