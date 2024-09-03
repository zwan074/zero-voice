import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

from librosa import magphase, pyin
import librosa


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


class ZeroVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, pre_process_f0_energy = False):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        self.pre_process_f0_energy = pre_process_f0_energy
        random.seed(random_seed)
        

    def get_triplet(self, line):

        filepath, text, speaker = line[0], line[1], line[2]
        filepath = filepath.replace('DUMMY','/workspace/dm_datasets/LibriTTS/')
        #print(filepath,text)
        text_gt = text
        text = self.get_text(text, add_blank=self.add_blank)
        mel, audio, sr = self.get_mel(filepath)

        if self.pre_process_f0_energy :
            self.comput_f0_energy (audio,filepath)

        f0 , energy = self.load_f0_energy(filepath)

        #print(f0.shape,energy.shape, mel.shape)
        speaker = self.get_speaker(speaker.replace('p','').replace('s',''))

        return (text, mel, speaker, filepath, text_gt, audio, sr, f0, energy) # new

    def comput_f0_energy (self, audio, filepath):
        #print(filepath,audio.shape)
        audio = audio.detach().cpu().numpy()
        f0 = compute_f0 ( 
            x= audio,
            pitch_fmax = 640,
            pitch_fmin = 1,
            hop_length =  self.hop_length,
            win_length  =  self.win_length,
            sample_rate = self.sample_rate,
            stft_pad_mode = "reflect",
            center = True)

        Energy = compute_energy(audio)

        np.save ( '/workspace/LibriTTS/train-clean-100-f0-energy/' + 'f0_' + filepath.split('/')[-1].replace('wav','npy'), f0)
        np.save ( '/workspace/LibriTTS/train-clean-100-f0-energy/' + 'Energy_' + filepath.split('/')[-1].replace('wav','npy'), Energy)

    def load_f0_energy (self, filepath):

        f0 = np.load ( '/workspace/ZS-TTS_ref_f0_engery/resources/filelists/libri-tts/f0/' + filepath.split('/')[-1].replace('.wav','')+ '_f0.npy'  )
        energy = np.load ( '/workspace/ZS-TTS_ref_f0_engery/resources/filelists/libri-tts/energy/' + filepath.split('/')[-1].replace('.wav','')+ '_energy.npy' )
        return torch.from_numpy(f0) , torch.from_numpy(energy)

        
    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, sr, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel, audio.squeeze() , sr 

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, file_path, text_gt, audio, sr, f0, energy = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker,  
                'file_path':file_path, 'text_gt': text_gt, 
                'audio':audio, 'sr':sr, 'f0':f0, 'energy':energy}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

    def get_ref_batch(self, size):
        idx = np.arange(size)
        test_batch = []
        #print(self.filelist)
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class ZeroVoiceBatchCollate(object):
    def __call__(self, batch):

        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, (y_max_length//4 ) * 4 + 4), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        audio = torch.zeros((B, y.shape[-1] * 256 ), dtype=torch.float32)

        f0 = torch.zeros((B, (y_max_length//4 ) * 4 + 4), dtype=torch.float32)
        energy = torch.zeros((B, (y_max_length//4 ) * 4 + 4), dtype=torch.float32)

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)    

            audio_ = item['audio']
            audio[i,:audio_.shape[-1]] = audio_

            f0_ = item['f0']
            f0[i,:f0_.shape[-1]] = f0_

            energy_ = item['energy']
            energy[i,:energy_.shape[-1]] = energy_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)

        return {'x': x, 'x_lengths': x_lengths, 
                'y': y, 'y_lengths': y_lengths, 'audio' : audio, 'f0':f0, 'energy':energy,
                'spk': spk
                }