# PyTorch 189M/480M Custom Decoder-Only LLM

A custom-built, from-scratch causal language model trained on a 1.75-billion token pipeline using PyTorch, with a custom tokenizer trained on a sample from the fineweb-edu dataset's CC-MAIN-2025-26 subset.

<br>

---

## Architecture & Specifications

- **Parameters:** 189M baseline (scalable up to 480M)
- **Dimensions (189M):** 16 Layers, $d_{\text{model}} = 768$, 16 Attention Heads, SwiGLU FFN ($d_{\text{ffn}} = 2048$)
- **Activations & Norm:** SwiGLU FFN and Pre-LN RMSNorm
- **Attention:** PyTorch Scaled Dot-Product Attention (`F.scaled_dot_product_attention`) for memory-efficient FlashAttention dispatch.

---


<br>

## Hardware Requirements

This training script is optimized for memory efficiency using **8-bit AdamW (`bitsandbytes`)**, **bfloat16 mixed-precision**, **gradient checkpointing**, and a sequence length of **512**.

| Component | Baseline Spec (Absolute Minimum to Run) | Recommended Spec (Fast & Stable Run) |
| :--- | :--- | :--- |
| **GPU** | **8 GB VRAM**<br>*(e.g., NVIDIA RTX 2060 Super, RTX 3060)* | **16 GB – 24 GB VRAM**<br>*(e.g., NVIDIA RTX 3090 / 4090, Cloud A10G)* |
| **System RAM** | **16 GB** | **32 GB** |
| **CPU** | **2+ Core(s)** | **4+ Cores** *(data loading is not parallelized, multi-core CPU recommended for system and training stability)* |
| **Storage** | **20 GB+ Free Space** | **50 GB+ NVMe SSD** *(Hugging Face caching)* |

> ℹ️ **Note:** An **NVIDIA CUDA-enabled GPU** is recommended to enable the `bitsandbytes` 8-bit optimizer for memory efficiency. 

<br>

## Project Structure
```text
├── model.py              # Transformer architecture, SwiGLU, and SDPA Decoder layers
├── main.py               # Training loop with 8-bit AdamW, AMP, and dataset streaming
├── train_tokenizer.py    # Custom tokenizer training script
├── batch_params.py       # Batch size picker
├── requirements.txt      # Python dependencies
└── model_checkpoints/    # Saved weights and best model artifacts
```

<br>

## Quickstart
### Initial setup
```bash
git clone https://github.com/krk2183/causal-decoder-torch
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

<br>

## Future Roadmap

The codebase is fully forward-compatible up to **480M parameters**. Current hyperparameter optimization efforts are focused on stabilizing and maximizing the efficiency of the **189M baseline**, with plans to scale parameter tiers in upcoming iterations.

<br>

In the meantime, feel free to scale up the configuration yourself using this setup:
```python
Transformer(
    d_model=1024,
    dim_feedforward=2730,
    nheads=16,
    n_layers=24,
    vocab_size=49267
)
```