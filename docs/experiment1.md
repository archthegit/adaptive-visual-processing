# Experiment 1

## Hypotheses

Experiment 1 tests whether Qwen2.5-VL allocates temporal information unevenly across HD-EPIC video clips, whether decoder temporal allocation is query-conditioned rather than only positional or generic video saliency, and whether high-ranked temporal bins are causally useful for VQA answers.

High attention alone is not treated as proof of importance. The experiment separates temporal non-uniformity, positional bias, video saliency, query-conditioned relevance, causal necessity, causal sufficiency, and actual pruning efficiency.

## Experimental Unit

The independent unit is the source video. Bins, heads, layers, questions, and repeated conditions are measurements nested under source videos and are not counted as independent samples.

The v3 manifest freezes the existing Experiment 1 question/video cohort and
uses only single-video examples from:

- `fine_grained`
- `gaze`
- `ingredient`
- `object_motion`

Image-only, video-plus-image, multi-video, unavailable, and corrupt/incomplete-video examples are excluded with explicit reasons.

## Dataset Construction

The old `outputs/experiment1_v2/runs/baseline` pilot used an invalid
17-second realtime policy. It is retained only as a failed engineering pilot
and must not be cited as final results.

Prepare the corrected v3 manifests and CPU-only sampling audit with:

```bash
python scripts/prepare_experiment1_v3_sampling.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --frozen-primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --output-dir outputs/experiment1_v3
```

This writes:

```text
outputs/experiment1_v3/duration_inventory.jsonl
outputs/experiment1_v3/primary_manifest.jsonl
outputs/experiment1_v3/additional_questions.jsonl
outputs/experiment1_v3/mismatched_queries.json
outputs/experiment1_v3/split_summary.json
outputs/experiment1_v3/exclusions.jsonl
outputs/experiment1_v3/sampling_audit.json
outputs/experiment1_v3/sampling_audit_per_example.jsonl
```

Duration groups are short/medium/long tertiles computed from
`analyzed_duration_seconds`, the exact interval presented to Qwen. The exact
thresholds are saved in `split_summary.json` with `duration_tertile_basis`.
Each primary record separately preserves `source_video_duration_seconds`,
`analyzed_duration_seconds`, analyzed start/end, and whether the analyzed input
is an unbounded/full-video input.

The primary manifest enforces at most one primary question per source video. The development/test split is source-video-level and approximately 20/80, stratified by category and duration group where possible. Mismatched queries are deterministic derangements within category and duration group, using a different source video and closest available token length.

For v3, the frozen question/video cohort is preserved exactly, but the
development/test split and `mismatched_queries.json` are recomputed after
assigning the corrected analyzed-duration groups. The adaptive realtime
`target_delta_t` is computed only after that corrected development split is
frozen.

## Sampling Policies

Primary policy:

```text
target_delta_t = median(development analyzed durations) / 16
desired_bins = ceil(analyzed_duration / target_delta_t)
bins = clip(desired_bins, 8, 64)
frames_per_bin = 2
max_frames = 128
```

The complete analyzed interval is partitioned into equal contiguous bins for
each example. The first bin begins at the effective analyzed start and the final
bin ends at the effective analyzed end, so long clips are compressed into wider
bins instead of being truncated. Decord length is the authoritative decodable
frame count; if the annotated end exceeds the decodable boundary, the effective
analyzed end is adjusted before constructing bins and the adjustment is recorded.
Each bin samples exactly two chronological frames. Frames are not duplicated
when enough unique source frames exist.

Robustness policy:

```text
frames = 128
bins = 16 equal-relative-time bins
frames_per_bin = 8
```

Every output record must preserve source frame index, timestamp, bin start/end, frame-to-bin mapping, effective FPS, original temporal position, presented temporal position, `second_per_grid_ts`, and Qwen temporal-position interval.

## Measurements

Encoder attention is query-agnostic and must never be labeled query relevance. For every Qwen vision block, aggregate incoming visual self-attention by temporal analysis bin and preserve per-layer and per-head measurements.

Decoder relevance is question-conditioned. During prefill, use natural-language question-token query rows only and visual-token key columns only. Store absolute question-to-vision attention mass separately from the normalized temporal distribution.

For encoder and decoder temporal distributions, record entropy, top-20% mass, uniform top-20% expectation, Gini coefficient, bins required for 80% mass, first/last-bin mass, top-bin relative temporal position, layer-to-layer Jensen-Shannon divergence, layer-to-layer Spearman correlation, and head agreement.

## Controls

Baseline uses original frames and the correct query.

Repeated-frame control repeats one deterministic frame at every temporal position while preserving sequence length and temporal positions. It measures positional bias.

Reversed-video control reverses presented frame order and maps scores back to original timestamps. It compares content-following and position-following correlations.

Mismatched-query control uses the same video with its frozen same-category, same-duration mismatched question.

Same-video/different-query analysis compares temporal distributions for additional questions about the same source video where available.

## Interventions

Use development videos only to select one decoder reference layer. Freeze the selected layer before held-out test interventions. Temporal ground-truth alignment is unavailable, so the predeclared deterministic selection rule is:

1. exclude decoder layers whose mean absolute question-to-visual attention mass is below the frozen threshold `0.05`
2. maximize correct-query versus mismatched-query temporal Jensen-Shannon divergence
3. break ties by higher mean absolute question-to-visual attention mass
4. then higher top-20% temporal mass
5. then shallower decoder layer index

Held-out test interventions derive bin sets from baseline outputs:

- correct-query top 20%
- bottom 20%
- position-matched random 20%
- mismatched-query top 20%
- contiguous high-attention cluster

Necessity interventions preserve token count and temporal positions with neutral replacement or a justified attention mask. Sufficiency/pruning interventions reduce temporal input and report token count and latency separately.

Fusion-depth intervention blocks direct question-to-selected-visual-token access after decoder layers `0, 4, 8, 12, 16, 20, 24, 27`.

## Statistics

Use paired per-video comparisons. Report means, paired differences, 95% confidence intervals from 10,000 hierarchical bootstrap replicates, paired permutation tests, Benjamini-Hochberg correction for layer-wise tests, and effect sizes. Stratify by duration group and category.

## Success Criteria

The experiment can support temporal pruning only if temporal distributions are non-uniform, query-conditioned effects exceed positional/video-saliency controls, top-bin interventions harm answer log probability or margin more than bottom/random/mismatched controls, and pruning retains performance while reducing visual tokens or latency.

## Current Implementation Status

Implemented:

- source-video-level v3 frozen-cohort manifest/audit builder
- complete local MP4 inventory through `ffprobe`
- duration tertiles from analyzed model-input durations
- separate preservation of source MP4 duration and analyzed question duration
- primary source-video manifest with one primary question per video and deterministic global category-duration balancing
- deterministic source-level development/test split
- deterministic same-category/same-duration mismatched query derangement
- additional same-video question manifest
- explicit exclusions
- primary and robustness sampling policy helpers
- runner-connected `--sampling-mode legacy|realtime|fixed_budget`
- artifact-level sampled frame indices, timestamps, sampling metadata and frame/bin mappings
- reversal and repeated-frame sampling metadata controls
- expected run matrix generation
- completeness checker and initial final-report artifact writer
- shared temporal metrics and clustered bootstrap helpers
- deterministic intervention manifest creation from baseline artifacts
- actual Qwen vision-block attention capture path for eager attention backends
- online canonical temporal pooling for captured vision attention chunks without retaining full all-layer token x token attention on CPU
- one-example stage profiler with elapsed time, peak CPU RSS, CUDA memory counters, and reduced tensor shapes
- repeated-frame and reversed-video control manifests
- mismatched-query manifests with runner-side question overrides
- same-video/different-query manifests with runner-side question overrides
- pre-encoder keep/pruning support distinct from pre-encoder masking
- frozen decoder reference-layer selection from development artifacts using the predeclared JSD-first rule and a frozen absolute-visual-mass threshold
- confirmatory intervention manifest generation restricted to `split=test` and to the frozen reference layer
- final condition-summary tables, average encoder/decoder heatmap generation,
  and figure manifest generation when plotting dependencies are installed
- paired per-video answer deltas, temporal control layer deltas, reversed-video
  content-versus-position summaries, hierarchical participant/video bootstrap
  CIs, paired permutation tests, paired effect sizes, Benjamini-Hochberg
  correction, duration/category stratification, same-video comparison summaries,
  encoder layer summaries, encoder representation summaries, and aggregate
  encoder-decoder alignment summaries when artifacts contain the required fields

Pending:

- real GPU execution of the complete condition matrix
- visual inspection/validation of generated paper figures
- representative-frame extraction currently remains summary-level; final
  publication panels still require choosing concrete completed examples

## Engineering Commands

Corrected v3 manifest and CPU-only sampling audit:

```bash
python scripts/prepare_experiment1_v3_sampling.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --frozen-primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --output-dir outputs/experiment1_v3
```

After manifests exist, run a small engineering validation before any pilot:

```bash
python scripts/run_experiment1.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --limit 3 \
  --num-frames 128 \
  --sampling-mode realtime \
  --sampling-policy-json outputs/experiment1_v3/split_summary.json \
  --resolution-config low \
  --attention-extraction reduced_sdpa \
  --query-scope question \
  --resume \
  --allow-7b-inference \
  --output-dir outputs/experiment1_v3/engineering_low_f32_baseline
```

The realtime policy freezes `target_delta_t = median(dev analyzed durations) /
16`, clips each example to 8-64 full-coverage bins, samples two chronological
frames per bin, and records exact frame/bin mappings in each artifact. The
frozen policy is stored in `split_summary.json` under
`realtime_sampling_policy`; every realtime run, including test-only intervention
manifests, must pass that file with `--sampling-policy-json` because
intervention manifests do not contain development records.

The robustness policy is run separately with fixed-budget sampling:

```bash
python scripts/run_experiment1.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --num-frames 128 \
  --sampling-mode fixed_budget \
  --condition baseline_fixed_budget \
  --resolution-config low \
  --attention-extraction reduced_sdpa \
  --query-scope question \
  --resume \
  --allow-7b-inference \
  --output-dir outputs/experiment1_v3/runs/baseline_fixed_budget
```

Realtime runs use the frozen policy:

```bash
python scripts/run_experiment1.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --num-frames 128 \
  --sampling-mode realtime \
  --sampling-policy-json outputs/experiment1_v3/split_summary.json \
  --condition baseline \
  --resolution-config low \
  --attention-extraction reduced_sdpa \
  --query-scope question \
  --resume \
  --allow-7b-inference \
  --output-dir outputs/experiment1_v3/runs/baseline
```

To construct manifests and print the ordered GPU commands without starting
expensive inference:

```bash
python scripts/prepare_experiment1_v2.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --output-root outputs/experiment1_v3 \
  --run-root outputs/experiment1_v3/runs
```

Generate the expected matrix and completeness report:

```bash
python scripts/analyze_experiment1_v2.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --output-root outputs/experiment1_v3/runs \
  --final-dir outputs/experiment1_v3/final \
  --bootstrap-replicates 10000
```

Create held-out intervention manifests from completed baseline artifacts:

```bash
python scripts/create_experiment1_v2_intervention_manifest.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --baseline-output-dir outputs/experiment1_v3/runs/baseline \
  --output-jsonl outputs/experiment1_v3/interventions/mask_top20.jsonl \
  --condition mask_top20 \
  --strategy top \
  --removal-fraction 0.2 \
  --frozen-reference-layer-json outputs/experiment1_v3/frozen_reference_layer.json \
  --seed 20260830
```

`random` intervention selection is position-matched: it first finds the
same-budget top-relevance bins, groups those bins by coarse relative temporal
tercile, and samples deterministic replacement bins from the same terciles where
possible. `uniform` is a separate strategy using evenly spaced temporal bins, so
`keep_uniform20` and `keep_random20` are intentionally different controls.

Fusion-depth manifests use the same baseline-derived bin selection, but encode
the decoder boundary in the condition name. For example, this allows top-bin
direct access through layer 8 and blocks direct question-to-selected-visual-bin
attention only after layer 8:

```bash
python scripts/create_experiment1_v2_intervention_manifest.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --baseline-output-dir outputs/experiment1_v3/runs/baseline \
  --output-jsonl outputs/experiment1_v3/interventions/fusion_block_top20_after_layer_8.jsonl \
  --condition fusion_block_top20_after_layer_8 \
  --strategy top \
  --removal-fraction 0.2 \
  --frozen-reference-layer-json outputs/experiment1_v3/frozen_reference_layer.json \
  --seed 20260830
```

When running a fusion-depth manifest, `scripts/run_experiment1.py` reads the
stored `decoder_direct_access_through_layer`. A CLI
`--decoder-direct-access-through-layer` value overrides the manifest for
engineering checks.

Fixed-budget causal masks are scheduled separately from realtime masks. Their
intervention manifests must be built from `runs/baseline_fixed_budget`, and the
corresponding run commands use `--sampling-mode fixed_budget`; fixed-budget
attention artifacts are expected to contain exactly 16 analysis bins.

Create control manifests:

```bash
python scripts/create_experiment1_v2_control_manifest.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --mismatched-queries outputs/experiment1_v3/mismatched_queries.json \
  --output-jsonl outputs/experiment1_v3/controls/mismatched_query.jsonl \
  --control mismatched_query
```

Same-video/different-query controls use `additional_questions.jsonl`:

```bash
python scripts/create_experiment1_v2_control_manifest.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --mismatched-queries outputs/experiment1_v3/mismatched_queries.json \
  --additional-questions outputs/experiment1_v3/additional_questions.jsonl \
  --output-jsonl outputs/experiment1_v3/controls/same_video_different_query.jsonl \
  --control same_video_different_query
```

Freeze the decoder reference layer after development baseline and mismatched
control artifacts exist:

```bash
python scripts/select_experiment1_v2_reference_layer.py \
  --primary-manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --baseline-output-dir outputs/experiment1_v3/runs/baseline \
  --mismatched-output-dir outputs/experiment1_v3/runs/mismatched_query \
  --output-json outputs/experiment1_v3/frozen_reference_layer.json
```

Medium-resolution profiling gate for the longest realtime engineering stress
case. This keeps all layers, heads, prompts, frames and medium resolution intact
while reporting stage-specific CPU/GPU peaks:

```bash
python scripts/run_experiment1.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3/primary_manifest.jsonl \
  --question-id ingredient_ingredient_adding_localization_16 \
  --num-frames 128 \
  --sampling-mode realtime \
  --sampling-policy-json outputs/experiment1_v3/split_summary.json \
  --resolution-config medium \
  --attention-extraction reduced_sdpa \
  --query-scope question \
  --condition baseline \
  --resume \
  --allow-7b-inference \
  --profile-one-example \
  --profile-output-json outputs/experiment1_v3/profiles/medium_128_ingredient_ingredient_adding_localization_16.json \
  --output-dir outputs/experiment1_v3/profile_medium_128_ingredient_ingredient_adding_localization_16
```
