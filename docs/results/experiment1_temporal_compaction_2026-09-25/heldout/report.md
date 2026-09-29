# Qwen Temporal Compaction Held-Out Evaluation

This is the frozen 56-example held-out confirmatory evaluation for retained native temporal cells `[1, 3]`.
The primary condition is `hard_evict`; `handoff_mean` is a secondary mechanistic comparison on identical examples.

Primary gate: **FAIL**.

| Condition | Accuracy | Mean Δ logp | Mean Δ margin | Median FLOP reduction | Profiled examples |
|---|---:|---:|---:|---:|---:|
| dense_custom | 0.411 | 0.000000 | 0.000000 | 0.000 | 8 |
| hard_evict | 0.411 | -0.016965 | -0.044643 | 0.490 | 8 |
| handoff_mean | 0.411 | -0.008106 | -0.033482 | 0.489 | 8 |

Latency profiling is limited to the frozen profiling subset and excludes video decoding, preprocessing, model loading, route selection and serialization.
The example is the independent statistical unit; pair rows are not used in this held-out evaluation.
