import sys
import os
#replace the path with your hifigan path to import Generator from models.py 
sys.path.append("/path/to/the/hifigan")
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from env import AttrDict
import json
from text_preprocess_for_inference import TTSDurAlignPreprocessor

SAMPLING_RATE = 22050

def load_hifigan_vocoder(language, gender, device):
    # Load HiFi-GAN vocoder configuration file and generator model for the specified language and gender
    vocoder_config = f"/path/to/the/vocoder/{gender}/aryan/hifigan/config.json"
    vocoder_generator = f"/path/to/the/vocoder/{gender}/aryan/hifigan/generator"
    # Read the contents of the vocoder configuration file
    with open(vocoder_config, 'r') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    # Move the generator model to the specified device (CPU or GPU)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Return the loaded and prepared HiFi-GAN generator model
    return generator


def load_fastspeech2_model(language, gender, device):
    tts_model = f"/path/to/tts/model/{gender}/{language}/fastspeech2_hs/model.pth"
    tts_config = f"/path/to/tts/model/{gender}/{language}/fastspeech2_hs/config.yaml"

    return Text2Speech(train_config=tts_config, model_file=tts_model, device=device)

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device):
    # Perform Text-to-Speech synthesis
    with torch.no_grad():
        # Load the FastSpeech2 model for the specified language and gender
        
        model = load_fastspeech2_model(language, gender, device)
       
        # Generate mel-spectrograms from the input text using the FastSpeech2 model
        out = model(sample_text, decode_conf={"alpha": 1})
        print("TTS Done")  
        x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
        x = x.to(device)
        
        # Use the HiFi-GAN vocoder to convert mel-spectrograms to raw audio waveforms
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        
        # Return the synthesized audio
        return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Inference")
    parser.add_argument("--language", type=str, required=True, help="Language (e.g., hindi)")
    parser.add_argument("--gender", type=str, required=True, help="Gender (e.g., female)")
    parser.add_argument("--sample_text", type=str, required=True, help="Text to be synthesized")
    args = parser.parse_args()
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the HiFi-GAN vocoder with dynamic language and gender
    vocoder = load_hifigan_vocoder(args.language, args.gender, device)
    preprocessor = TTSDurAlignPreprocessor()

    # Preprocess the sample text
    preprocessed_text, phrases = preprocessor.preprocess(args.sample_text, args.language, args.gender)

    # Call the text_synthesis function with user provided and preprocessed sample_text
    
    audio = text_synthesis(preprocessed_text, vocoder, MAX_WAV_VALUE, device)
    output_file = f"{args.language}_{args.gender}_output.wav"
    write(output_file, SAMPLING_RATE, audio)
