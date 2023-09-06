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
                        text = "पाइथन में प्रोग्रामिंग, डेटा स्ट्रक्चर्स और एल्गोरिदम पर पाठ्यक्रम पर पहले व्याख्यान में, आपका स्वागत है।"
                        tmp_dir="tmp"
                        lang = "Hindi"
                        timestamp = "1"
                        preprocess_start = time.time()
                        # create tmp directory
                        os.makedirs(tmp_dir, exist_ok = True)
                        textfile_inp = os.path.abspath(f"{tmp_dir}/input.txt")
                        textfile = os.path.abspath(f"{tmp_dir}/input_preprocessed.txt")
                        
                        with open(textfile_inp, "w") as f:
                                f.write(text)
                        # preprocess the text
                        command = f"/var/www/html/IITM_TTS/E2E_TTS_FS2/text_proc/text_proc.sh {textfile_inp} {textfile} {lang} {timestamp} {tmp_dir}"
                        os.system(command)

                        # textfile_oneline = os.path.abspath(f"{tmp_dir}/input_preprocessed_oneline.txt")
                        # command = """awk 'BEGIN{ORS=" "}{for (i=2; i<NF; i++) printf $i " "; print $NF}' """ + textfile + " > " + textfile_oneline
                        # os.system(command)

                        preprocessed_text = []
                        with open(textfile, "r") as f:
                                for line in f.readlines():            
                                        preprocessed_text.append(line.split(" ", 1)[1].strip())
                        preprocess_end = time.time()
                        t2s_start = preprocess_end
                        out = text2speech(" ".join(preprocessed_text))
                        t2s_end = time.time()
                        vocoder_start = t2s_end
                        x = out["feat_gen_denorm"].T.unsqueeze(0).to(device)
                        y_g_hat = generator(x)
                        audio = y_g_hat.squeeze()
                        audio = audio * MAX_WAV_VALUE
                        audio = audio.cpu().numpy().astype('int16')
                        output_file = "gen.wav"
                        write(output_file, h.sampling_rate, audio)
                        vocoder_end = time.time()
                        et = vocoder_end
                        elapsed = (et-st)
                        print(f"Total elapsed time: {elapsed}\nText Preprocess: {(preprocess_end-preprocess_start)}\nText-to-mel: {(t2s_end-t2s_start)}\nMel to wave: {(vocoder_end-vocoder_start)}")
