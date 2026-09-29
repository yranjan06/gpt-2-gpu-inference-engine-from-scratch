# Phase 6: INT8 Weight Quantization

**Goal:** Reduce memory footprint and memory bandwidth requirements by compressing model weights from 32-bit float to 8-bit integers.

## What we did here:
- Implemented Symmetric Per-Tensor INT8 Quantization.
- Verified theoretical error bounds (max error = scale/2, MSE ~= scale^2/12) exactly.
- Tested a naive "dequantize-then-matmul" approach and found it 3.7x slower than fp32 due to memory read/write overheads.
- Built a **Fused INT8 Dequantize+Matvec Kernel** in Triton, which was 2.28x faster than naive, though still slower than cuBLAS fp32 (proving that bandwidth savings alone don't win without utilizing hardware Tensor Cores).
- Tracked how quantization error compounds across reductions, noting a 6.24% relative error but maintaining a high 0.997 cosine similarity.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | ,- |
| **Memory Reduction** | Exactly 4.00x smaller weights |
| **Weight Error (Max Relative)** | 0.39% |
| **Output Error (Compounded)** | 6.24% relative error |
| **Output Cosine Similarity** | 0.997 (Maintained vector direction) |
| **Naive Dequantize-then-Matmul** | 3.7x slower than fp32 (due to memory R/W overhead) |
| **Fused INT8 Matvec Kernel** | 2.28x faster than naive approach (but still 0.62x vs cuBLAS fp32) |
