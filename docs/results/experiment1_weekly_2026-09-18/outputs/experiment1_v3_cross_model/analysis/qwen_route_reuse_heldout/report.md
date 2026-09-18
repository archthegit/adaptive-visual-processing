# Qwen Heldout Confirmatory Route-Replay Analysis

This is a 56-example heldout_confirmatory route-replay analysis. Routes were replayed from dense baseline artifacts.
It does not establish online routing and does not measure actual latency savings; runtimes are instrumentation runtimes.
Category breakdowns are exploratory because the cells are small.

Causal gate: **FAIL**.

PASS if adaptive routing preserves correct-answer log probability and answer margin better than both equal-budget random and uniform controls; FAIL if adaptive is no better than either control; INCONCLUSIVE if point estimates favor adaptive but uncertainty is too large.

## Condition Summary

- adaptive: mean Δ logp -0.0107 [-0.1185, 0.0886], mean Δ margin -0.0714 [-0.2292, 0.0625], accuracy 0.321, flip rate 0.232.
- random: mean Δ logp 0.0071 [-0.1094, 0.1230], mean Δ margin 0.0067 [-0.1932, 0.1887], accuracy 0.375, flip rate 0.214.
- uniform: mean Δ logp 0.0146 [-0.1005, 0.1585], mean Δ margin -0.0022 [-0.1613, 0.2212], accuracy 0.339, flip rate 0.250.

## Outputs

- `per_example.csv`: exact paired per-example table.
- `summary.json`: validation, aggregate statistics, pairwise adaptive-control differences, and gate decision.
- `quality_delta.png`, `margin_delta.png`, `accuracy_and_flip_rates.png`: descriptive development-set plots.
