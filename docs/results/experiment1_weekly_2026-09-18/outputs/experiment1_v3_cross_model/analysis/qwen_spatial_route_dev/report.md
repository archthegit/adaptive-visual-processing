# Qwen spatial-route development kill test

Gate: **REJECT**

This is a 15-example development kill test. Routes are replayed from dense baseline artifacts; this does not establish online routing or measured latency/FLOP savings.

The gate tests whether dense decoder question-to-visual attention identifies spatial visual tokens that preserve answer quality better than equal-budget random and uniform spatial controls while preserving all sampled inputs and all Qwen native temporal cells.
