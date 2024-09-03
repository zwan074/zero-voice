from model.utils import fix_len_compatibility


# data parameters

train_filelist_path = 'resources/filelists/libri-tts/train-clean-100.txt'
valid_filelist_path = 'resources/filelists/libri-tts/valid.txt'

cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_fft = 1024
sample_rate = 24000
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# acoustic encoder parameters
n_enc_channels = 512  
filter_channels = 768
filter_channels_dp =  512  
ms_hd = 512
n_enc_layers = 6  
enc_kernel = 3
enc_dropout = 0.1
n_heads = 6  
window_size = 4

# mel decoder parameters
dec_dim = 256  
beta_min =  0.05
beta_max = 20.0 
pe_scale = 1000  

# training parameters

log_dir = 'logs/zero_voice'

test_size = 20
n_epochs = 100
batch_size = 8 
learning_rate = 1e-4 
seed = 32
save_every = 4
out_size = 72 #fix_len_compatibility(2*22050//256)
