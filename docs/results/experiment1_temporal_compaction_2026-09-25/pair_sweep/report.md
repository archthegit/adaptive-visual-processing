# Qwen Temporal-Handoff Retained-Pair Sweep

This is a 15-example development analysis. It is not held-out evidence and does not claim end-to-end generation speedup.

The sweep evaluates all six retained native-cell pairs for `handoff_mean` and `hard_evict`, using the example as the independent unit.

## Development Conclusions

1. Compact memory versus hard deletion: see `memory_benefit` in `analysis_summary.json`.
2. Attention-ranked selection versus arbitrary routes: see `selection_benefit`.
3. Retained attention mass predictiveness: see per-example Spearman summaries.
4. Extreme random-route wins: inspect `route_quality_range` and per-example selected-pair ranks.
