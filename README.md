# Fastspeech2 Model using MFA

This repository contains a Fastspeech2 Model implemented using the Montreal Forced Aligner (MFA) for speech synthesis. The model is capable of generating mel-spectrograms from text inputs and can be used to synthesize speech.

## Model Files

The model includes the following files:

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

1. Install PyTorch separately (you can install the specific version based on your requirements):

conda install pytorch torchvision torchaudio cudatoolkit=<your_cuda_version>


Usage
Activate the conda environment:

conda activate <environment_name>

Use the inference file to synthesize speech from text inputs:

python inference.py --input_text "Your input text here" --output_wav output.wav

Vocoder
For generating WAV files from mel-spectrograms, you can use a vocoder of your choice. One popular option is the HIFIGAN vocoder. Please refer to the documentation of the vocoder you choose for installation and usage instructions.

Citation
If you use this Fastspeech2 Model in your research or work, please consider citing:

“
COPYRIGHT
2016
TTS
Consortium,
TDIL,
Meity
represented by Hema A Murthy & S Umesh, DEPARTMENT OF
Computer
Science
and
Engineering
and
Electrical
Engineering, IIT Madras. ALL RIGHTS RESERVED”






Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
