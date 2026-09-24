# Phase 1: GPU Basics & Performance Profiling

**Goal:** Understand the baseline performance characteristics of the Tesla T4 GPU before writing complex inference code.

## What we did here:
- Checked PyTorch GPU access and exact VRAM capacity.
- Measured CPU to GPU memory transfer bandwidth (~4.8 GB/s).
- Demonstrated the **Async Timing Trap**: why you must use `torch.cuda.synchronize()` to get real GPU execution times.
- Ran a size sweep on Matrix Multiplication (Matmul) to find the crossover point where GPU compute starts beating CPU compute (n~128).
- Evaluated memory bandwidth limits on simple operations to establish a baseline for optimization.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | ,- |
| **CPU -> GPU Transfer Bandwidth** | ~4.8 GB/s (pageable memory) |
| **GPU VRAM Bandwidth** | ~230 - 245 GB/s (72% of T4 peak) |
| **CPU vs GPU Matmul (n=4096)** | 17.2x speedup on GPU |
| **Compute Crossover Point** | GPU beats CPU starting at Matrix Size ~128 |
| **Async Timing Trap** | Without sync: fake 261 TFLOPS. With sync: real 4.2 TFLOPS. |
