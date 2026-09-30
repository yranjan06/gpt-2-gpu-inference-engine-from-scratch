# Phase 3: KV Cache Implementation

**Goal:** Drastically speed up token generation by caching past Key and Value states.

## What we did here:
- Built a `KVCache` class to store $K$ and $V$ tensors layer-by-layer.
- Split the attention mechanism into **Prefill** (processing the whole prompt) and **Decode** (processing one token at a time).
- Measured formal LLM metrics: **TTFT** (Time To First Token) and **TPOT** (Time Per Output Token).
- Performed a Context Length Sweep, proving that the overhead of KV Cache pays off only after ~128 tokens, matching our earlier compute vs overhead crossover findings.

## 📊 Key Metrics & Findings

| Metric | Measurement / Result |
| --- | --- |
| **TTFT (Time To First Token)** | 10.18 ms |
| **TPOT (Time Per Output Token)** | 9.63 ms |
| **Decode Throughput** | 103.8 tokens/sec |
| **Cache Growth Rate** | ~0.074 MB/token |
| **KV Cache Crossover Point** | Caching beats naive recompute after ~128 tokens |
| **Bandwidth Ceiling Reached** | ~22% (Bottlenecked by ~30µs kernel launch overhead) |
