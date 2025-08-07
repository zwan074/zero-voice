# Zero-Voice: A Low-Resource Approach For Zero-shot Text-to-Speech

This repo holds the codes of paper: "Zero-Voice: A Low-Resource Approach For Zero-shot Text-to-Speech".
https://doi.org/10.36227/techrxiv.173750018.83521236/v1

## News

**[Aug. 29, 2024]** We release the Demos for Zero-Voice on LibriTTS (train-clean-100) and Te Reo Maori;  ASR on Te Reo Maori.

**[July. 21, 2025]** We release Synthetic Te Reo Māori Speech Samples for the paper.

## Overview

We propose Zero-Voice, a novel end-to-end zero-shot text-to-speech (TTS) model that adopts a hierarchical framework comprising a Source Filter Network, an Acoustic Feature Encoder, a diffusion-based Acoustic Feature Refiner, and a diffusion-based Waveform Vocoder. This architecture avoids information compression and loss, facilitating effective transfer of speech style features from reference inputs to synthesized speech, while addressing the mismatch between acoustic representations, mel-spectrograms, and waveform outputs during both training and inference.
At the core of Zero-Voice is a novel Source Filter Network that enables the unsupervised decoupling of prosodic components(i.e., pitch, rhythm, and timbre from reference speech). A key innovation in our approach is the random cut-and-connect data augmentation operation, which is applied to reference speech to enhance pattern diversity during training. This operation significantly improves the model’s zero-shot generalization, particularly under low-resource conditions.
We conduct objective experiments under low-resource settings to compare our model with recent strong zero-shot TTS baseline methods under high-resource settings (e.g., StyleTTS 2 and HierSpeech++). 
Experimental results demonstrate that Zero-Voice achieves comparable performance to these high-resource methods. 
Notably, Zero-Voice demonstrates strong generalization and robustness even when trained on a very small number of speakers and small datasets (e.g., 5-8 hours of transcribed data). 
Moreover, we collect and label 27 hours Te Reo Māori speech data (i.e., an official and endangered language of New Zealand). We train the Zero-Voice model on this dataset, and use it to synthesize Te Reo Māori speech data to enhance speech recognition models for the language. This approach yields state-of-the-art results for the Māori (language code: nz\_mi) test set of Google Fleurs dataset.

<figure>
<img src="assets/modeltrainingandinference.svg" alt="modeltrainingandinference" style="zoom: 50%;" />
<figcaption>In the Training Procedure (a), we jointly train three components: the Acoustic Feature Encoder, the Acoustic Feature Refiner, and the Waveform Vocoder. During the Inference Procedure (b), the Acoustic Feature Encoder processes speech prompts and phonemes to generate the aligned acoustic features, denoted as $\mu_{aligned}$. The Acoustic Feature Refiner then refines these features, reducing noise to produce the predicted mel-spectrogram output, $y_{pred\_mel}$. Finally, the Waveform Vocoder converts $y_{pred\_mel}$ into the final waveform output, $y_{pred\_wave}$. Additionally, Fig. \ref{data_preprocessing} provides detailed information on pipeline of the speech prompts processed by the Source Filter Network within the Acoustic Feature Encoder.</figcaption>
</figure>

<figure>
<img src="assets/data_preprocessing.svg" alt="/data_preprocessing" style="zoom: 70%;" />
<figcaption>\textbf{Pipeline of Source Filter Network:} (a) 
During training, the speech prompts $x_{ref\_mel}$, $x_{ref\_pitch}$, and $x_{ref\_energy}$, which represent the mel-spectrograms, pitch, and energy derived from the waveform, undergo a random cut-and-connect process that truncates their time-domain features. These truncated features are stretched or compressed to distort their rhythm, and are then aligned by three encoders: the Mel-Style Encoder, Pitch Encoder and Energy Encoder. Subsequently, elementwise-summation of the outpts from the three encoders, and the Transformer Encoder modules produce $x_{ref}$;
(b) During inference, the speech prompts $x_{ref\_mel}$, $x_{ref\_pitch}$, and $x_{ref\_energy}$ are directly used as input for the three encoders and the Transformer Encoder to produce $x_{ref}$. </figcaption>
</figure>


## Te Reo Māori Synthetic Speech Samples

[[Synthesized Māori Speech](https://drive.google.com/drive/folders/1xAimbNIDO9dP1aePiEhwYoBgsB5077bU?usp=drive_link)], partial samples for fintuning the whisper-large-v3 model in the paper.


## Hugging Face Space Demo

Some key checkpoints in the paper (presented as huggingface space):

1. Zero-Voice (Trained on LibriTTS (train-clean-100) subset) [[zero-voice](https://huggingface.co/spaces/zwan074/zero-voice)] 
2. Zero-Voice Low-Resource (Trained on 1/8 size of LibriTTS (train-clean-100) subset)  [[zero-voice-lr](https://huggingface.co/spaces/zwan074/zero-voice-lr)]
3. 
## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

## Training

1. Make filelists of your training audio data into `resources/filelists` folder.
2. Use`pre_preprocess_f0_energy.py` to pre-process f0 and energy  
3. Set experiment configuration in `params.py` file.
4. letters can be set up in `/text/symbols.py` file, phonemes are set up in `/text/cmudict.py` for English. 
6. Specify your GPU device and run training script:
    ```bash
    export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
    python train.py 
    ```
7.  You can download pre-trained HiFi-GAN checkpoints from [here](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y), and save under the `/hifi-gan/` to recieve waveform outputs from the acoustic feature refiner during the training.
8.  During training all logging information and checkpoints are stored in `log_dir`, which you can specify in `params.py` before training.

## Inference

1. Set up `model_ckpt_path`, `ref_speech_path`, and `text` in `inference.py` file.
4. inference:
    ```bash
    model_ckpt_path = 'logs/zero_voice/ckpt.pt'
    ref_speech_path = '/workspace/dm_datasets/LibriTTS/train-clean-100/7190/90543/7190_90543_000005_000001.wav'
    text = 'What sort of evidence is there?'
    ```
5. Check out folder called `out.wav` for generated audios.

   
## Reference

Some code snippets are from :

1. Grad-TTS [[Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)] 
2. DiffWave [[Grad-TTS](https://github.com/lmnt-com/diffwave)] 
