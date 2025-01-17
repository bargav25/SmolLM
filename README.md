# SmolLM: A Lightweight Language Model Implementation

This repository contains a lightweight implementation of a causal language model, **smolLM**, using PyTorch. The model is designed to mimic the architecture of modern transformer-based models, including features like rotary embeddings, multi-head attention, and RMS normalization.

## Features

- **Rotary Positional Embeddings**: Efficiently encode positional information in queries and keys.
- **Layer Normalization**: RMSNorm for better numerical stability.
- **Modular Design**: Easy-to-understand, well-commented codebase for each component.
- **Hugging Face Weight Compatibility**: Pretrained weights can be loaded from Hugging Face for evaluation.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/bargav25/smolLM.git
cd smolLM 
```

### Install Dependencies


```bash
pip install torch transformers
```

## Download Pretrained Weights

Download the pretrained weights from Hugging Face using Git LFS:

```bash
git lfs install
git clone https://huggingface.co/dsouzadaniel/C4AI_SMOLLM135
mv C4AI_SMOLLM135/BareBones_SmolLM-135M.pt ./
```

### Testing the Model

```bash
python test_model.py
```


## Disclaimer

This implementation of **smolLM** is an independent project and is not affiliated with or endorsed by the original authors or organizations behind the pretrained model hosted on Hugging Face. The pretrained weights (`BareBones_SmolLM-135M.pt`) and associated tokenizer were downloaded from [Hugging Face](https://huggingface.co/dsouzadaniel/C4AI_SMOLLM135), and all credits for the training and development of these weights go to their respective authors.

If you are one of the authors and have concerns or suggestions regarding this project, please feel free to open an issue or reach out.

