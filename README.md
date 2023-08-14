# Fastspeech2 Model using MFA

This repository contains a Fastspeech2 Model for 8 Indian languages (male and female both) implemented using the Montreal Forced Aligner (MFA) for speech synthesis. The model is capable of generating mel-spectrograms from text inputs and can be used to synthesize speech.

The Repo is large in size: Please download the model which only are needed.

## Model Files

The model for each language includes the following files:

- `config.yaml`: Configuration file for the Fastspeech2 Model.
- `energy_stats.npz`: Energy statistics for normalization during synthesis.
- `feats_stats.npz`: Features statistics for normalization during synthesis.
- `feats_type`: Features type information.
- `pitch_stats.npz`: Pitch statistics for normalization during synthesis.
- `model.pth`: Pre-trained Fastspeech2 model weights.

## Installation

1. Create a conda environment using the provided `environment.yaml` file:

```shell
conda env create -f environment.yaml
```

2. Install PyTorch separately (you can install the specific version based on your requirements):
```shell
conda install pytorch torchvision torchaudio cudatoolkit=<your_cuda_version>
```
## Vocoder
For generating WAV files from mel-spectrograms, you can use a vocoder of your choice. One popular option is the [HIFIGAN](https://github.com/jik876/hifi-gan) vocoder. Please refer to the documentation of the vocoder you choose for installation and usage instructions. (We have provided Vocoder tuned on Aryan and Dravidian languages)

## Usage
Activate the conda environment (check inside environment.yaml file):
```shell
conda activate tts-fs-hifigan
```
Use the inference file to synthesize speech from text inputs:
```shell
python inference.py --input_text "Your input text here" --language <language> --gender <gender> --output_wav output.wav
```

### Citation
If you use this Fastspeech2 Model in your research or work, please consider citing:

“
COPYRIGHT
2016 TTS Consortium,
TDIL, Meity represented by Hema A Murthy & S Umesh,
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
and
ELECTRICAL ENGINEERING,
IIT MADRAS. ALL RIGHTS RESERVED "



Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
