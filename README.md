# The Illustrated Llama-Based Decoder — with PyTorch Code

A picture-first, build-it-from-scratch guide to how a modern **Llama-based decoder**
language model works, implemented entirely in **PyTorch** and runnable on a tiny toy model.

This is a spiritual sequel to *The Illustrated GPT-2*. It keeps the same decoder-only
Transformer backbone but explains and implements the four upgrades that define a
Llama-based decoder:

| Component | GPT-2 (classic) | Llama-based decoder (this notebook) |
|---|---|---|
| Normalization | LayerNorm (post-norm) | **RMSNorm (pre-norm)** |
| Position info | Learned absolute embeddings | **Rotary Position Embeddings (RoPE)** |
| Attention | Multi-Head Attention | **Grouped-Query Attention (GQA)** |
| Feed-forward | GELU MLP (2 matrices) | **SwiGLU (3 matrices, gated)** |
| Linear bias | Yes | **No bias anywhere** |

## What's inside `Transformer_llama_based_decoder.ipynb`

Each concept is explained with illustrations/GIFs, the math, runnable PyTorch code, and a
small worked example with printed outputs and matplotlib visualizations:

1. Tiny configuration (`ModelArgs`)
2. The big picture: a stack of decoder blocks (pre-norm)
3. Token embeddings
4. **RMSNorm** (+ LayerNorm vs RMSNorm visual)
5. **RoPE** (+ rotation, frequency, and relative-distance visuals)
6. Self-attention recap + causal masking + scaled dot-product attention
7. **Grouped-Query Attention** (+ MHA vs GQA vs MQA diagram)
8. **SwiGLU** feed-forward (+ activation-function plot)
9. The decoder block (`TransformerBlock`)
10. The full `LlamaBasedDecoder` model (+ real attention maps)
11. Auto-regressive generation and a tiny overfit-to-prove-it-works training loop

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib numpy jupyter
jupyter notebook Transformer_llama_based_decoder.ipynb
```

The notebook is already executed with outputs embedded, so you can also just read it
top-to-bottom. Images live in `./image`.
