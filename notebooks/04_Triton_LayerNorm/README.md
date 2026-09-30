# Phase 4: Custom Triton Kernels - LayerNorm

**Goal:** Dive into GPU kernel programming using OpenAI's Triton to fuse operations and save memory bandwidth.

## What we did here:
- Introduced Triton and wrote our first custom GPU kernel.
- Built a **Fused LayerNorm** kernel, calculating mean, variance, and normalization in a single GPU pass.
- Benchmarked isolated performance, achieving **2.47x speedup** over our manual PyTorch LayerNorm.
- Integrated the kernel into the full forward pass, observing Amdahl's Law: a massive isolated speedup only yielded a 1.13x end-to-end pipeline speedup because matmuls dominate the runtime.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | --- |
| **Isolated Kernel Speedup** | 2.47x faster than PyTorch Manual LayerNorm |
| **End-to-End Pipeline Speedup** | 1.13x faster overall |
| **Observation** | Amdahl's Law limit reached; LayerNorm is a minority of total execution time. |
