# Qwen Temporal-Handoff Development Pilot

This is a 15-example development kill test, not held-out confirmatory evidence.
The intervention physically compacts the decoder sequence after layer 8; it is not attention masking or KV-cache reuse.

Gate decision: **INCONCLUSIVE**.

| Condition | Accuracy | Mean Δ logp | Mean Δ margin | Median stack speedup | Median FLOP reduction |
|---|---:|---:|---:|---:|---:|
| dense_custom | 0.200 | 0.000000 | 0.000000 | 1.000 | 0.000 |
| handoff_mean | 0.200 | 0.042232 | 0.225000 | 1.374 | 0.489 |
| hard_evict | 0.200 | 0.016588 | 0.183333 | 1.408 | 0.490 |
| random_handoff | 0.267 | 0.138984 | 0.233333 | 1.324 | 0.489 |

Profiling excludes video decoding, preprocessing, model loading, artifact serialization and route selection.
All intervention comparisons use `dense_custom` as the paired dense control.
