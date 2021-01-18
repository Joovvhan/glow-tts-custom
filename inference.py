import numpy as np
import os
import argparse

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import models
import utils

import json
import jamotools
from glob import glob
from g2pk import G2p
g2p = G2p()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference settings.')
    parser.add_argument('-t', type=str, default='안녕, 세상!', help='Script')
    parser.add_argument('-m', type=str, default='kss', help='TTS model name')
    parser.add_argument('-v', type=str, default='', help='Vocoder model name')
    parser.add_argument('-f', type=str, default='tst_stns.txt', help='Sentence file list')
    parser.add_argument('-n', type=float, default=0.333, help='Noise scale')

    args = parser.parse_args()

    language = json.load(open("language_setting.json", 'r'))['language']

    if language == 'english':
        cleaners = 'english_cleaners'
    elif language == 'korean':
        cleaners = 'korean_cleaners'
        args.t = jamotools.split_syllables(args.t, jamo_type="JAMO")
    elif language == 'korean_phoneme':
        cleaners = 'korean_phoneme_cleaners'
        args.t = g2p(args.t, descriptive=True, group_vowels=True)
        args.t = jamotools.split_syllables(args.t, jamo_type="JAMO")
    else:
        assert False, f'Language Error [{language}]!'

    # Clear remains
    for f in glob('./hifi-gan/test_mel_files/*.npy'): os.remove(f)
    for f in glob('./generated_files_from_mel/*.wav'): os.remove(f)

    # model_dir = "./logs/kss/"
    # model_dir = "./logs/ljspeech2/"
    model_dir = f"./logs/{args.m}/"
    hps = utils.get_hparams_from_dir(model_dir)
    checkpoint_path = utils.latest_checkpoint_path(model_dir)

    # If you are using a provided pretrained model
    # hps = utils.get_hparams_from_file("./configs/any_config_file.json")
    # checkpoint_path = "/path/to/pretrained_model"

    model = models.FlowGenerator(
        len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to("cuda")

    utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

    try:
        cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
    except AttributeError:
        cmu_dict = None 

    if args.f is None:
        tst_stns = [('sample.wav', args.t)]
    else:
        with open(args.f, 'r') as f:
            tst_stns = [line.split('|') for line in f]
    
    for stn in tst_stns: print(stn)

    for file_name, tst_stn in tst_stns:
                
        if getattr(hps.data, "add_blank", False):
            text_norm = text_to_sequence(tst_stn.strip(), [cleaners], cmu_dict)
            text_norm = commons.intersperse(text_norm, len(symbols))
        else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
            tst_stn = " " + tst_stn.strip() + " "
            text_norm = text_to_sequence(tst_stn.strip(), [cleaners], cmu_dict)
        sequence = np.array(text_norm)[None, :]
        print("".join([symbols[c] for c in sequence[0] if c < len(symbols)]))
        # print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
        x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

        with torch.no_grad():
            # noise_scale = .667
            # noise_scale = .333
            noise_scale = args.n
            length_scale = 1.0
            (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)

            # save mel-framescd 
            if not os.path.exists('./hifi-gan/test_mel_files'):
                os.makedirs('./hifi-gan/test_mel_files')

            mel_file_name = file_name.replace('.wav', '.npy')
            np.save(f"./hifi-gan/test_mel_files/{mel_file_name}", y_gen_tst.cpu().detach().numpy())

    python_script = './hifi-gan/inference_e2e.py'
    # options = f'--checkpoint_file ./runs/{}'
    # options = f'--checkpoint_file ./hifi-gan/runs/cp_hifigan/g_00110000' + \
    #           f' --input_mels_dir ./hifi-gan/test_mel_files'
    options = f'--checkpoint_file ./hifi-gan/runs/cp_hifigan_custom/g_00015000' + \
              f' --input_mels_dir ./hifi-gan/test_mel_files'

    os.system(f'python {python_script} {options}')
        # os.rename('./generated_files_from_mel/sample_generated_e2e.wav', \
        #          f'./generated_files_from_mel/{file_name}' )
        # shutil.move('./hifi-gan/generated_files_from_mel/sample_generated_e2e.wav', 'wavs')
    # # "./hifi-gan/generated_files_from_mel/sample_generated_e2e.wav"
