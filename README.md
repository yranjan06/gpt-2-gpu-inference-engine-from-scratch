# GPT-2 GPU Inference Engine (from scratch)

### Run on Google Colab (Stage-by-Stage)
| Phase | Topic | Colab Link |
| :--- | :--- | :--- |
| **1** | GPU Basics & Profiling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/01_GPU_Basics/01_GPU_Basics.ipynb) |
| **2** | Manual Forward Pass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/02_Manual_Forward_Pass/02_Manual_Forward_Pass.ipynb) |
| **3** | KV Cache Implementation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/03_KV_Cache/03_KV_Cache.ipynb) |
| **4** | Custom Triton LayerNorm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/04_Triton_LayerNorm/04_Triton_LayerNorm.ipynb) |
| **5** | Custom Triton Attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/05_Triton_Attention/05_Triton_Attention.ipynb) |
| **6** | INT8 Weight Quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/06_INT8_Quantization/06_INT8_Quantization.ipynb) |
| **7** | Advanced Sampling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yranjan06/gpt-2-gpu-inference-engine-from-scratch/blob/main/notebooks/07_Sampling/07_Sampling.ipynb) |

A hand-written GPU inference engine for GPT-2, built to understand what production LLM serving frameworks abstract away: memory movement, kernel-launch overhead, floating-point precision, Triton kernel fusion, and quantization tradeoffs. No `model.generate()`, no `torch.nn.MultiheadAttention`, every operation from token embedding to the final logit was implemented by hand and verified against a HuggingFace reference model.

**Model:** GPT-2 (124M params, fp32) &nbsp;|&nbsp; **Hardware:** Tesla T4 (40 SMs, 2560 CUDA cores, 15.6 GB VRAM)

## Performance Metrics

| Metric | Result |
| --- | --- |
| **TTFT** (5-token prompt) | 10.18 ms |
| **TPOT** (decode step) | 9.63 ms |
| **Decode throughput** | 103.8 tokens/sec |
| **Memory-bandwidth ceiling** | ~480 tokens/sec (22% reached, overhead-bound, not bandwidth-bound) |
| **KV cache growth** | ~0.074 MB/token |
| **Naive-recompute vs KV-cache crossover** | ~128 tokens of context |
| **Full-model correctness (fp32, 12 layers)** | 6.1e-5 max abs diff vs HuggingFace |
| **Fused LayerNorm (Triton)** | 2.47x isolated / 1.13x full-pipeline |
| **Fused attention (Triton)** | 2.35x isolated / 1.32x full-pipeline |
| **INT8 weight quantization** | 4.00x smaller, 0.39% max relative weight error |
| **Fused INT8 matvec kernel** | 2.28x faster than naive dequant, still 0.62x vs cuBLAS fp32 |

## Engineering Findings

- **Kernel-launch overhead, not compute, dominates at small model scale.** A T4 has a fixed ~30 µs floor per kernel launch. GPT-2's forward pass issues ~150-200 launches per token, which is why TTFT and TPOT came out nearly identical (10.18 ms vs 9.63 ms) even though they are processing very different amounts of work.
- **Floating-point error accumulates sub-linearly.** Stacking the same operation across 12 layers grew the error from 1.14e-5 to 6.1e-5, about 5x, not 12x, consistent with independent rounding errors partially cancelling ($\sqrt{n}$ growth).
- **A "faster" kernel in isolation doesn't guarantee a faster pipeline.** Fused LayerNorm gave 2.47x alone but only 1.13x end-to-end (Amdahl's Law in action).
- **Reading less data doesn't automatically mean going faster.** The INT8 fused kernel reads 4x fewer weight bytes but still lost to cuBLAS's fp32 matmul (0.62x) because it uses a plain reduction loop instead of Tensor Cores.
- **Quantization error compounds through a matmul's reduction dimension.** A 0.39% weight-level error became a 6.24% output-level relative error after summing across 768 terms, while cosine similarity stayed high (0.997).

## How to Run (Local or Google Colab)

1. **Option A: Google Colab**
   Click any of the "Open in Colab" badges at the top of this README to open a specific phase. Ensure you select a **T4 GPU** runtime (`Runtime > Change runtime type > Hardware accelerator > T4 GPU`) when running them.

2. **Option B: Local Setup**
   Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/yranjan06/gpt-2-gpu-inference-engine-from-scratch.git
   cd gpt-2-gpu-inference-engine-from-scratch
   pip install -r requirements.txt
   ```
   Launch Jupyter to explore the notebooks phase by phase:
   ```bash
   jupyter notebook notebooks/
   ```

## Limitations & Next Steps

- Single sequence, batch size 1 only, no continuous batching (yet).
- Quantization is naive per-tensor symmetric INT8, no per-channel scales, no calibration (GPTQ/AWQ-style), and the Triton kernel doesn't use Tensor Cores yet.
- **Next Step:** Update the Triton INT8 kernel to utilize Tensor Cores (`tl.dot`) to achieve actual speedups over cuBLAS fp32.
