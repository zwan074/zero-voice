import params
from model.tts_model import ZeroVoiceModel
from data import  ZeroVoiceBatchCollate, ZeroVoiceDataset


nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)



if __name__ == "__main__":

    print('Initializing data loaders...')
    train_dataset = ZeroVoiceDataset(params.valid_filelist_path, params.cmudict_path, params.add_blank,n_fft, 
                                     params.n_feats, params.sample_rate, params.hop_length, params.win_length, 
                                     params.f_min, params.f_max, pre_process_f0_energy = True)

    batch_collate = ZeroVoiceBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=False,
                        num_workers=16, shuffle=False)
    
    print('Start Processing...')
    iteration = 0
    for epoch in range( 1):
        i = 0
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                print('test')

        break
        
    print('fin')
