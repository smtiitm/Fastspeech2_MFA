from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
import numpy as np
import os
import json
from env import AttrDict
import torch
import time

for dev in ("cpu", "cuda"):
        print(f"loading model in {dev}")
        device=torch.device(dev)
        y1 = torch.load("/speech/arun/tts/hifigan/denorm/test_243.npy.pt", map_location=device)
        y2 = torch.concat([y1]*5, dim=1)
        y3 = torch.concat([y1]*10, dim=1)

        config_file = os.path.join('/speech/arun/tts/hifigan/cp_hifigan/config.json')
        with open(config_file) as f:
                data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        generator = Generator(h).to(device)
        state_dict_g = torch.load("/speech/arun/tts/hifigan/cp_hifigan/g_00120000", device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        for i in range(3):
                print("Run ",i)
                for x in [y1, y2, y3]:
                        with torch.no_grad():
                                st = time.time()
                                y_g_hat = generator(x)
                                audio = y_g_hat.squeeze()
                                audio = audio * MAX_WAV_VALUE
                                audio = audio.cpu().numpy().astype('int16')
                                output_file = "gen.wav"
                                write(output_file, h.sampling_rate, audio)
                                et = time.time()
                                elapsed = (et-st)
                                print("Elapsed time:", elapsed)
