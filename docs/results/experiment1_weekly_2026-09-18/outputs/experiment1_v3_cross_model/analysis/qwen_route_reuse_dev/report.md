# Qwen Development Route-Replay Pilot

This is a 15-example development kill test. Routes were replayed from dense baseline artifacts.
It does not establish online routing and does not measure actual latency savings; runtimes are instrumentation runtimes.
Category breakdowns are exploratory because the cells are small.

Causal gate: **INCONCLUSIVE**.

PASS if adaptive routing preserves correct-answer log probability and answer margin better than both equal-budget random and uniform controls; FAIL if adaptive is no better than either control; INCONCLUSIVE if point estimates favor adaptive but uncertainty is too large.

## Condition Summary

- adaptive: mean Δ logp 0.1487 [-0.0367, 0.4299], mean Δ margin 0.2917 [0.0179, 0.6979], accuracy 0.267, flip rate 0.067.
- random: mean Δ logp -0.0918 [-0.3817, 0.2183], mean Δ margin -0.0333 [-0.3270, 0.3304], accuracy 0.200, flip rate 0.133.
- uniform: mean Δ logp 0.0824 [-0.1980, 0.3084], mean Δ margin 0.2333 [-0.0577, 0.5312], accuracy 0.200, flip rate 0.200.

## Outputs

- `per_example.csv`: exact paired per-example table.
- `summary.json`: validation, aggregate statistics, pairwise adaptive-control differences, and gate decision.
- `quality_delta.png`, `margin_delta.png`, `accuracy_and_flip_rates.png`: descriptive development-set plots.
