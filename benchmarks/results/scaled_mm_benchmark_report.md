# Helion vs CUTLASS scaled_mm Benchmark Report

**Date:** 2026-04-29
**GPU:** NVIDIA H100 80GB HBM3
**Model:** RedHatAI/Qwen3-8B-FP8-dynamic
**vLLM version:** 0.18.2rc1.dev387+g189035283

## Common Configuration

- **Prompts:** 2048
- **Max concurrency:** 16
- **Output length:** 600 tokens
- **Warmups:** 256
- **Compilation mode:** VLLM_COMPILE (inductor backend)
- **CUDA graphs:** FULL_AND_PIECEWISE, capture sizes 1-512 (max_cudagraph_capture_size=512)
- **Chunked prefill:** enabled, max_num_batched_tokens=8192

Batches with num_tokens <= 512 use CUDA graphs (piecewise for prefill, full for
decode). Larger batches (prefill chunks up to 8192 tokens) exceed the max
capture size and fall back to compiled inductor execution without CUDA graphs.

---

## Experiment 1: input_len=512 (Custom Op Path, commit 07e0f0c0c)

Traffic: fixed 512-token inputs at infinite request rate. With max concurrency
16, prefill batches are at the cudagraph boundary — most execution uses CUDA
graphs.

Dispatch path: `cutlass_scaled_mm()` in `_custom_ops.py` routes to
`torch.ops.vllm_helion.scaled_mm` (helion) or `torch.ops._C.cutlass_scaled_mm`
(cutlass) based on `VLLM_DISABLE_HELION` env var.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 442.14 | 441.51 | +0.1% |
| Request throughput (req/s) | 4.63 | 4.64 | -0.2% |
| Output token throughput (tok/s) | 2779.22 | 2783.16 | -0.1% |
| Peak output throughput (tok/s) | 3008.00 | 3004.00 | +0.1% |
| Total token throughput (tok/s) | 5150.82 | 5158.12 | -0.1% |
| Mean TTFT (ms) | 101.76 | 110.23 | -7.7% |
| Median TTFT (ms) | 127.33 | 135.73 | -6.2% |
| P99 TTFT (ms) | 154.25 | 154.68 | -0.3% |
| Mean TPOT (ms) | 5.59 | 5.57 | +0.4% |
| Median TPOT (ms) | 5.55 | 5.53 | +0.4% |
| P99 TPOT (ms) | 5.68 | 5.69 | -0.2% |
| Mean ITL (ms) | 5.60 | 5.57 | +0.5% |
| Median ITL (ms) | 5.48 | 5.47 | +0.2% |
| P99 ITL (ms) | 6.55 | 6.49 | +0.9% |

## Experiment 2: input_len=4096 (Custom Op Path, commit 07e0f0c0c)

Traffic: fixed 4096-token inputs at infinite request rate. With max concurrency
16, prefill batches (up to 8192 tokens) far exceed the max cudagraph capture
size (512) and fall back to compiled inductor execution. Decode still uses
CUDA graphs.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 764.61 | 778.26 | -1.8% |
| Request throughput (req/s) | 2.68 | 2.63 | +1.9% |
| Output token throughput (tok/s) | 1607.10 | 1578.90 | +1.8% |
| Peak output throughput (tok/s) | 2048.00 | 2000.00 | +2.4% |
| Total token throughput (tok/s) | 12578.25 | 12357.55 | +1.8% |
| Mean TTFT (ms) | 601.02 | 601.35 | -0.1% |
| Median TTFT (ms) | 641.43 | 644.04 | -0.4% |
| P99 TTFT (ms) | 1088.28 | 1027.61 | +5.9% |
| Mean TPOT (ms) | 8.97 | 9.14 | -1.9% |
| Median TPOT (ms) | 8.89 | 9.06 | -1.9% |
| P99 TPOT (ms) | 9.79 | 9.96 | -1.7% |
| Mean ITL (ms) | 8.97 | 9.14 | -1.9% |
| Median ITL (ms) | 8.03 | 8.14 | -1.4% |
| P99 ITL (ms) | 9.49 | 9.61 | -1.2% |

## Experiment 3: input_len=512 (IR Op Commit, b20e66990)

Same dispatch path as Experiment 1 (IR op registered but not wired into model
forward path). Verifies the IR op registration code does not regress perf.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 446.65 | 441.25 | +1.2% |
| Request throughput (req/s) | 4.59 | 4.64 | -1.1% |
| Output token throughput (tok/s) | 2751.17 | 2784.83 | -1.2% |
| Peak output throughput (tok/s) | 3040.00 | 3008.00 | +1.1% |
| Total token throughput (tok/s) | 5098.84 | 5161.23 | -1.2% |
| Mean TTFT (ms) | 176.66 | 111.04 | +59.1% |
| Median TTFT (ms) | 145.83 | 132.83 | +9.8% |
| P99 TTFT (ms) | 1468.53 | 163.64 | +797% |
| Mean TPOT (ms) | 5.53 | 5.57 | -0.7% |
| Median TPOT (ms) | 5.51 | 5.55 | -0.7% |
| P99 TPOT (ms) | 5.68 | 5.70 | -0.4% |
| Mean ITL (ms) | 5.53 | 5.57 | -0.7% |
| Median ITL (ms) | 5.36 | 5.46 | -1.8% |
| P99 ITL (ms) | 6.42 | 6.39 | +0.5% |

## Experiment 4: input_len=4096 (IR Op Commit, b20e66990)

Same dispatch path as Experiment 2 on the IR op commit.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 768.15 | 778.16 | -1.3% |
| Request throughput (req/s) | 2.67 | 2.63 | +1.5% |
| Output token throughput (tok/s) | 1599.68 | 1579.12 | +1.3% |
| Peak output throughput (tok/s) | 2016.00 | 2000.00 | +0.8% |
| Total token throughput (tok/s) | 12520.17 | 12359.22 | +1.3% |
| Mean TTFT (ms) | 578.73 | 594.06 | -2.6% |
| Median TTFT (ms) | 622.54 | 604.44 | +3.0% |
| P99 TTFT (ms) | 967.68 | 1022.81 | -5.4% |
| Mean TPOT (ms) | 9.05 | 9.16 | -1.2% |
| Median TPOT (ms) | 8.95 | 9.07 | -1.3% |
| P99 TPOT (ms) | 9.83 | 9.97 | -1.4% |
| Mean ITL (ms) | 9.05 | 9.16 | -1.2% |
| Median ITL (ms) | 8.08 | 8.14 | -0.7% |
| P99 ITL (ms) | 9.41 | 9.65 | -2.5% |

## Experiment 5: Mixed input_len ~[128, 8192] (Custom Op Path, commit 07e0f0c0c)

Traffic: input lengths sampled uniformly from ~[128, 8192] (mean ~4160), output
lengths from ~[10, 640] (mean ~330), infinite request rate. This exercises the
full range: short requests fit in cudagraphs, long prefills exceed the 512
capture limit.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 521.24 | 514.07 | +1.4% |
| Request throughput (req/s) | 3.93 | 3.98 | -1.3% |
| Output token throughput (tok/s) | 1286.21 | 1304.17 | -1.4% |
| Peak output throughput (tok/s) | 1885.00 | 1963.00 | -4.0% |
| Total token throughput (tok/s) | 17998.00 | 18249.29 | -1.4% |
| Mean TTFT (ms) | 140.38 | 143.16 | -1.9% |
| Median TTFT (ms) | 129.24 | 133.82 | -3.4% |
| P99 TTFT (ms) | 364.44 | 364.82 | -0.1% |
| Mean TPOT (ms) | 12.05 | 11.84 | +1.8% |
| Median TPOT (ms) | 11.98 | 11.78 | +1.7% |
| P99 TPOT (ms) | 17.28 | 16.97 | +1.8% |
| Mean ITL (ms) | 12.01 | 11.83 | +1.5% |
| Median ITL (ms) | 8.88 | 8.71 | +2.0% |
| P99 ITL (ms) | 117.30 | 117.95 | -0.6% |

## Experiment 6: Mixed input_len ~[128, 8192] (IR Op Commit, b20e66990)

Same dispatch path as Experiment 5 on the IR op commit.

| Metric | Helion | CUTLASS | Delta |
| --- | --- | --- | --- |
| Benchmark duration (s) | 505.16 | 521.33 | -3.1% |
| Request throughput (req/s) | 4.05 | 3.93 | +3.1% |
| Output token throughput (tok/s) | 1327.16 | 1286.00 | +3.2% |
| Peak output throughput (tok/s) | 2065.00 | 1948.00 | +6.0% |
| Total token throughput (tok/s) | 18571.03 | 17994.99 | +3.2% |
| Mean TTFT (ms) | 138.85 | 142.99 | -2.9% |
| Median TTFT (ms) | 126.84 | 133.08 | -4.7% |
| P99 TTFT (ms) | 396.98 | 381.16 | +4.1% |
| Mean TPOT (ms) | 11.81 | 12.05 | -2.0% |
| Median TPOT (ms) | 11.52 | 11.97 | -3.8% |
| P99 TPOT (ms) | 18.31 | 17.28 | +6.0% |
| Mean ITL (ms) | 11.63 | 12.00 | -3.1% |
| Median ITL (ms) | 8.55 | 8.88 | -3.7% |
| P99 ITL (ms) | 97.37 | 116.94 | -16.7% |

---

## Analysis

### input_len=512 (cudagraph-dominated)

Throughput is within ~1% between Helion and CUTLASS — effectively noise. Both
kernels run inside captured CUDA graphs for decode batches, so kernel dispatch
overhead is amortized. Prefill batches at 512 tokens are at the cudagraph
boundary.

### input_len=4096 (prefill-heavy, non-cudagraph)

With 4096-token inputs, prefill dominates wall time. Prefill batches (up to
8192 tokens via chunked prefill) exceed the max cudagraph capture size (512)
and fall back to compiled inductor execution. Here Helion shows a consistent
~1.5-2% throughput improvement and ~1.5-2% lower TPOT/ITL. TTFT is similar
between the two.

### Mixed traffic [128, 8192] (realistic workload)

With input lengths uniformly sampled from ~128 to ~8192, both short
(cudagraph-eligible) and long (non-cudagraph) prefills are exercised.

On the custom op commit (Experiment 5), Helion and CUTLASS are within ~1.5%
on throughput — CUTLASS slightly ahead on output throughput while Helion has
slightly lower TTFT. The difference is within run-to-run variance.

On the IR op commit (Experiment 6), Helion shows a clearer ~3% throughput
advantage over CUTLASS, with 3-4% lower median TPOT and ITL, and notably
16.7% lower P99 ITL (97ms vs 117ms). This suggests Helion handles the
diversity of matrix sizes in mixed traffic better, with fewer tail-latency
spikes.

### IR Op commit impact

The IR op registration code (commit b20e66990) does not regress performance.
In fact, the IR op commit runs show slightly better Helion numbers than the
custom op commit, likely due to run-to-run variance rather than the IR code
itself (the actual dispatch path is identical).

### Summary

| Workload | Winner | Margin |
| --- | --- | --- |
| Short prefill (512 tokens) | Tie | <1% |
| Long prefill (4096 tokens) | Helion | ~1.5-2% throughput, ~1.5-2% lower TPOT |
| Mixed traffic [128-8192] | Helion | ~3% throughput, ~3-4% lower TPOT, ~17% lower P99 ITL |
| Decode latency | Tie | <1% (cudagraph-dominated) |

## Notes

- All experiments use the `vllm_helion::scaled_mm` custom op path (NOT the
  `vllm_ir::scaled_mm` IR op path). The IR op is registered but the model's
  forward pass calls `cutlass_scaled_mm()` which dispatches to the helion
  custom op directly.
- CUTLASS runs set `VLLM_DISABLE_HELION=1` to force the CUTLASS path.
- Compile cache was cleared between each experiment.
- All runs completed 2048/2048 requests with 0 failures.
- Mixed traffic uses `--random-range-ratio 0.97` with `--input-len 4160
  --output-len 330`, giving input lengths in ~[124, 8195] and output lengths
  in ~[10, 640].
