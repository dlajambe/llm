# Large Language Model from Scratch

In this project, a generative transformer-based large language model (LLM) was implemented from scratch using PyTorch. This project was inspired by Andrej Karpathy's excellent [LLM tutorial](https://youtu.be/kCc8FmEb1nY?si=umG3WZzNv6TyMz43), although several features were added:

- Parameterization of all global variables
- Refactoring of training script into functional components to facilitate debugging
- Implementation of layer normalization component of multi-head attention blocks from scratch
- Implementation of a character tokenizer class to assist with data preprocessing
- Full modularization (without global variables) of supporting classes and functions to allow for reuse in other projects

## Description

This project contains several modules that can be used to build generative transformer models in PyTorch.

The classes included in the llm.py file are based on the Decoder portion of the transformer model architecture 
decsribed in the famous [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. By intelligently stacking multiple transformer blocks, a powerful language model can be trained in a more computationally efficient fashion than recursive language models.

Although higher performance can be achieved using word- or syllable-level embeddings, character-level embeddings were used in this project to reduce the dimensionality of the dataset and make training feasible with a household computer.

## Getting Started

The `train_llm.py` script contains an example of how to use the project to build a generative large language model from an initial user-provided character sequence. A sample dataset is provided (Shakespearean scripts), but it can be substituted with any user-provided text file.

### Dependencies

All that is required is Python and a package manager such as Virtualenv or Conda. The required Python packages are listed in the `requirements.txt` file.

### Installation

To install, simply clone the repository onto your machine and use your package manager to recreate the environment with the provided `requirements.txt` file. If you have Anaconda or Miniconda, you can instead use the `environment.yml` to reproduce the environment.

### Execution

This program can be run through the command line. With your environment activated, open a command prompt in the `app/` directory and execute the following command: 
```
python train_model.py
```

All hyperparameters are stored in `app/config/hyperparameters.py`. These values can be adjusted to change the model architecture, training, and evaluation characteristics.

## License

This project is licensed under the Apache version 2.0 License - see the LICENSE.md file for details