# Experiment 1 v3: VILA Baseline Decoder Analysis

This document is produced by `scripts/plot_vila_temporal_baseline.py` for the
VILA-Llama3 baseline run:

- Input: `outputs/experiment1_v3_cross_model/runs/vila_baseline`
- Output: `outputs/experiment1_v3_cross_model/preliminary/vila_baseline`
- Diagnostics: `outputs/experiment1_v3_cross_model/preliminary/vila_baseline/vila_baseline_diagnostics.json`

The script updates this file with the current numerical results when run on the
real artifacts.

## Scope

This is a decoder-only cross-architecture baseline for
`Efficient-Large-Model/Llama-3-VILA1.5-8B`. VILA uses exactly eight temporal
bins and one deterministic center frame per bin. Failed sampling-ineligible
records are reported separately and are not included in denominators.

No VILA encoder measurements are generated or implied.

## Decoder Temporal Allocation

![Decoder attention lift heatmap](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_attention_lift_heatmap.png)

The heatmap displays `8 * p(bin) - 1`, so uniform temporal allocation is zero.
This answers whether VILA decoder attention is nonuniform and how large the
maximum deviation from uniform is. The exact maximum is stored in
`vila_baseline_diagnostics.json`.

## Entropy By Question Type

![Decoder entropy by question type](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_entropy_by_question_type.png)

Normalized entropy versus decoder depth shows whether temporal allocation
changes non-monotonically. Curves are shown overall and for question types with
at least five completed examples.

## Absolute Visual Access

![Decoder absolute visual mass](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_absolute_visual_mass.png)

Absolute question-token-to-visual-token attention mass is kept separate from
conditional temporal allocation. It is not normalized across temporal bins.

## Early Versus Late Bins

![First and last bin mass](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_first_last_bin_mass.png)

This plot answers whether early layers prefer early bins and whether later
layers shift toward later bins.

![Top bin position](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/decoder_top_bin_position.png)

The top-bin position heatmap shows the fraction of examples whose top-ranked bin
is each of bins 0 through 7 at every decoder layer.

## Accuracy

![Accuracy by category](../outputs/experiment1_v3_cross_model/preliminary/vila_baseline/accuracy_by_category.png)

Accuracy is descriptive only. Exact numerators and denominators are included in
the figure and diagnostics.

## Interpretation Guardrails

This VILA baseline can qualitatively reproduce the Qwen pattern only if it shows
depth-dependent, nonuniform decoder temporal allocation with separate absolute
visual-access dynamics. It does not by itself establish content relevance,
causal importance, or pruning utility.

Repeated-frame, reversed-video, and matched-Qwen controls are still required.
