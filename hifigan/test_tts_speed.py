from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
import numpy as np
import os
import json
from env import AttrDict
import torch
import time
from espnet2.bin.tts_inference import Text2Speech

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
        text2speech = Text2Speech(train_config="/speech/arun/tts/hifigan/config.yaml",model_file="/var/www/html/IITM_TTS/E2E_TTS_FS2/fastspeech2/models/Hindi_male/train.loss.ave.pth",device=dev)
        for i in range(3):
                print("Run ",i)                
                with torch.no_grad():
                        st = time.time()
                        out = text2speech("EटA sटarakcars औr Elgoridam par pAठyakram par pahlE wyAखyAn mEq")
                        x = out["feat_gen_denorm"].T.unsqueeze(0).to(device)
                        y_g_hat = generator(x)
                        audio = y_g_hat.squeeze()
                        audio = audio * MAX_WAV_VALUE
                        audio = audio.cpu().numpy().astype('int16')
                        output_file = "gen.wav"
                        write(output_file, h.sampling_rate, audio)
                        et = time.time()
                        elapsed = (et-st)
                        print("Elapsed time:", elapsed)
