# Phase 2: GPT-2 Manual Forward Pass

**Goal:** Reconstruct the complete GPT-2 forward pass from raw weights without using high-level `nn.Module` abstractions.

## What we did here:
- Loaded raw `.safetensors` files manually, bypassing HuggingFace's model class.
- Implemented Token & Position embeddings.
- Hand-wrote the `LayerNorm` function.
- Extracted Q, K, V projections and implemented **Causal Self-Attention** from scratch.
- Reconstructed the GELU-based MLP block and residual connections.
- Verified mathematical correctness by comparing our manual logits with HuggingFace's output (achieving 6.1e-5 max abs diff).
- Implemented a naive (non-KV-cached) generation loop to establish a slow baseline.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | ,- |
| **Model Size** | 124.4 M Parameters |
| **Total fp32 Memory Footprint** | 497.8 MB |
| **Full-Model Correctness** | 6.1e-5 max absolute difference vs HuggingFace |
| **Naive Generation Speed (Uncached)** | ~10-11 ms per step (for very short sequences) |
