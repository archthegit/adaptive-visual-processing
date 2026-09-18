# Qwen Route-Replay Failure-Mode Analysis

This CPU-only analysis explains the completed route-replay results. It does not change inference, routing, manifests, artifacts, or causal outcomes.
Development and held-out cohorts are reported separately; pooled 71-example statistics are not treated as confirmatory.

## Development

Examples: 15; routed-layer rows: 225.
Mean adaptive-minus-dense log probability: 0.1487 [-0.0367, 0.4299].
Mean adaptive-minus-dense margin: 0.2917 [0.0250, 0.7019].

## Heldout

Examples: 56; routed-layer rows: 840.
Mean adaptive-minus-dense log probability: -0.0107 [-0.1138, 0.0878].
Mean adaptive-minus-dense margin: -0.0714 [-0.2310, 0.0655].

## Failure Hypotheses

H1 tests whether harm tracks high anchor entropy or low retained target-layer mass.
H2 tests whether harm tracks adjacent selections or low temporal coverage.
H3 tests whether harm tracks declining route agreement. Anchor distance is summarized only as route staleness because distances 1, 2, and 3 are applied together and yield one final answer.
H4 tests whether apparently stable, high-mass, well-covered routes still cause harm, which would implicate hard deletion itself.

Observed causal outcome is the paired answer-quality delta from the completed route-replay intervention. Failure-mode associations are exploratory. Causal harm by anchor distance is not identifiable in this design.

## Recommendation: shorter/dynamic refresh
