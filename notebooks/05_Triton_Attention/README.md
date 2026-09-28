# Phase 5: Custom Triton Kernels - Causal Attention

**Goal:** Optimize the most memory-intensive part of the Transformer using Kernel Fusion.

## What we did here:
- Implemented a custom Fused Causal Attention kernel in Triton.
- Avoided materializing the large $N \times N$ attention matrix in HBM (High Bandwidth Memory), keeping operations in fast SRAM (similar to FlashAttention principles).
- Debugged real-world hardware constraints (e.g., minimum block sizes for Tensor Cores).
- Achieved **2.35x isolated speedup** and further optimized the end-to-end decode pipeline.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | ,- |
| **Isolated Kernel Speedup** | 2.35x faster than standard manual attention |
| **End-to-End Pipeline Speedup** | 1.32x faster overall |
| **Precision** | Exact match to fp32 precision floor |
