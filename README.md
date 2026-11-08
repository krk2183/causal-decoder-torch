# PyTorch 120M/236M Custom Decoder-Only LLM

A custom-built, from-scratch causal language model trained on a 24-billion token pipeline using PyTorch, with a custom tokenizer trained on a sample from the fineweb-edu dataset's CC-MAIN-2025-26 subset.

## Architecture Highlights
* **Parameters:** ~120M/236M (8 layers, $d_{\text{model}}=1024$, 16 attention heads)
* **Activations & Norm:** Custom SwiGLU FFN and Pre-LN RMSNorm
* **Data Pipeline:** On-t# PyTorch 120M/236M Custom Decoder-Only LLM

## ⚙️ Hardware Requirements

This training script is optimized for memory efficiency using **8-bit AdamW (`bitsandbytes`)**, **bfloat16 mixed-precision**, **gradient checkpointing**, and a sequence length of **512**.

| Component | Baseline Spec (Absolute Minimum to Run) | Recommended Spec (Fast & Stable Run) |
| :--- | :--- | :--- |
| **GPU** | **8 GB VRAM**<br>*(e.g., NVIDIA RTX 2060 Super, RTX 3060)* | **16 GB – 24 GB VRAM**<br>*(e.g., NVIDIA RTX 3090 / 4090, Cloud A10G)* |
| **System RAM** | **16 GB** | **32 GB** |
| **CPU** | **2+ Core(s)** | **4+ Cores** *(data loading is not parallelized, multi-core CPU recommended for system and training stability)* |
| **Storage** | **20 GB+ Free Space** | **50 GB+ NVMe SSD** *(Hugging Face caching)* |

> ℹ️ **Note:** An **NVIDIA CUDA-enabled GPU** is recommended to enable the `bitsandbytes` 8-bit optimizer for memory efficiency. 

## Quickstart
### Initial setup
```bash
git clone https://github.com/krk2183/decoder-lm-120m-236m
cd pytorch-llm-scratch
pip install -r requirements.txt
```
### Training the tokenizer
```bash
python train_tokenizer.py
```
### Training the model
```bash
python main.py
```
he-fly streaming via Hugging Face (`fineweb-edu`) with custom token-counting guards and rolling token-packing.

## Quickstart
```bash
git clone [https://github.com/krk2183/decoder-lm-120m-236m](https://github.com/krk2183/decoder-lm-120m-236m)
cd pytorch-llm-scratch
pip install -r requirements.txt
python main.py