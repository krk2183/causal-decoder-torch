# PyTorch 120M/236M Custom Decoder-Only LLM

A custom-built, from-scratch causal language model trained on a 24-billion token pipeline using PyTorch, with a custom tokenizer trained on a sample from the fineweb-edu dataset's CC-MAIN-2025-26 subset.

## Architecture Highlights
* **Parameters:** ~120M/236M (8 layers, $d_{\text{model}}=1024$, 16 attention heads)
* **Activations & Norm:** Custom SwiGLU FFN and Pre-LN RMSNorm
* **Data Pipeline:** On-t# PyTorch 120M/236M Custom Decoder-Only LLM

A custom-built, from-scratch causal language model trained on a 24-billion token pipeline using PyTorch, with a custom tokenizer trained on a sample from the fineweb-edu dataset's CC-MAIN-2025-26 subset.

## Architecture Highlights
* **Parameters:** ~120M/236M (8 layers, embedding size 1024, 16 attention heads)
* **Activations & Norm:** Custom SwiGLU FFN and Pre-LN RMSNorm
* **Data Pipeline:** On-the-fly streaming via Hugging Face (`fineweb-edu`) with custom token-counting guards and rolling token-packing.

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