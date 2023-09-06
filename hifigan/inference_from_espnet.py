from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import time

h = None
device = "cpu"


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            print(filname)
            # wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            # wav = wav / MAX_WAV_VALUE
            # wav = torch.FloatTensor(wav).to(device)
            # x = get_mel(wav.unsqueeze(0))
            # print("x is ", x.shape)
            arr2 = torch.load(os.path.join(a.input_wavs_dir, filname))
            print("arr2 type", type(arr2))
            # arr = np.load(os.path.join(a.input_wavs_dir, filname))
            arr = np.array(arr2).astype(float)
            print("arr type", type(arr))
            # arr = np.loadtxt(os.path.join(a.input_wavs_dir, filname),dtype='float')
            if arr.shape[0]!=80:
                arr = arr.T
            print(arr.shape)
            # arr = x.detach().cpu().numpy()
            # print(arr.shape[0],arr.shape[1],arr.shape[2])
            # arr_new = arr.reshape(arr.shape[1],arr.shape[2])
            # print(arr_new.shape)
            arr_new2 = arr.reshape(1,arr.shape[0],arr.shape[1])
            ###x_new = torch.from_numpy(arr_new2).float().to(device)
            x_new = torch.FloatTensor(arr_new2).to(device)
            print("x_new",x_new.shape)
            # x = x_new
            # np.savetxt('tests/' + filname + '.txt', arr_new)
            # y_new = torch.from_numpy(arr.unsqueeze(0))
            # print(y_new.shape)
           
            st = time.time()
            y_g_hat = generator(x_new)
            et = time.time()
            print("Time taken by generator:", (et-st))
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='denorm')
    parser.add_argument('--output_dir', default='wav_folder')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if device is None and torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("device", device)
    inference(a)


if __name__ == '__main__':
    main()

