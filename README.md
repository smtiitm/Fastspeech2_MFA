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






Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
