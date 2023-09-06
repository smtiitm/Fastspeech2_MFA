from espnet2.bin.tts_inference import Text2Speech
import time

for device in ("cpu", "cuda"):
    print(f"loading model in {device}")
    text2speech = Text2Speech(train_config="/speech/arun/tts/hifigan/config.yaml",model_file="/var/www/html/IITM_TTS/E2E_TTS_FS2/fastspeech2/models/Hindi_male/train.loss.ave.pth",device=device)
    for i in range(5):
        print("Run ",i)
        st = time.time()
        out = text2speech("EटA sटarakcars औr Elgoridam par pAठyakram par pahlE wyAखyAn mEq")
        et = time.time()
        elapsed = (et-st)
        print("Elapsed time:", elapsed)
    print("-----------------------------")