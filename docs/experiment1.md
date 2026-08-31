# Experiment 1

## Hypotheses

Experiment 1 tests whether Qwen2.5-VL allocates temporal information unevenly across HD-EPIC video clips, whether decoder temporal allocation is query-conditioned rather than only positional or generic video saliency, and whether high-ranked temporal bins are causally useful for VQA answers.

High attention alone is not treated as proof of importance. The experiment separates temporal non-uniformity, positional bias, video saliency, query-conditioned relevance, causal necessity, causal sufficiency, and actual pruning efficiency.

## Experimental Unit

The independent unit is the source video. Bins, heads, layers, questions, and repeated conditions are measurements nested under source videos and are not counted as independent samples.

The v2 manifest uses only single-video examples from:

- `fine_grained`
- `gaze`
- `ingredient`
- `object_motion`

Image-only, video-plus-image, multi-video, unavailable, and corrupt/incomplete-video examples are excluded with explicit reasons.

## Dataset Construction

Build the source-video-level manifest with:

```bash
python scripts/create_experiment1_v2_manifest.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --output-dir outputs/experiment1_v2 \
  --seed 20260830 \
  --dev-fraction 0.2
```

This writes:

```text
outputs/experiment1_v2/duration_inventory.jsonl
outputs/experiment1_v2/primary_manifest.jsonl
outputs/experiment1_v2/additional_questions.jsonl
outputs/experiment1_v2/mismatched_queries.json
outputs/experiment1_v2/split_summary.json
outputs/experiment1_v2/exclusions.jsonl
```

Duration groups are short/medium/long tertiles computed from eligible locally available analyzed durations. The exact thresholds are saved in `split_summary.json`.

The primary manifest enforces at most one primary question per source video. The development/test split is source-video-level and approximately 20/80, stratified by category and duration group where possible. Mismatched queries are deterministic derangements within category and duration group, using a different source video and closest available token length.

## Sampling Policies

Primary policy:

```text
delta_t = ceil(P95(development analyzed durations) / 64)
bins = min(64, ceil(duration / delta_t))
frames_per_bin = 2
max_frames = 128
```

Each real-time bin samples two chronological frames. Frames are not duplicated when enough unique source frames exist.

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

Use development videos only to select one decoder reference layer. Freeze the selected layer before held-out test interventions. Selection priority:

1. temporal evidence alignment when annotations permit it
2. correct-query versus mismatched-query separation
3. sufficient absolute visual attention mass
4. temporal concentration

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

- source-video-level v2 manifest builder
- complete local MP4 inventory through `ffprobe`
- duration tertiles from eligible local analyzed durations
- primary source-video manifest with one primary question per video
- deterministic source-level development/test split
- deterministic same-category/same-duration mismatched query derangement
- additional same-video question manifest
- explicit exclusions
- primary and robustness sampling policy helpers
- reversal and repeated-frame sampling metadata controls
- expected run matrix generation
- completeness checker and initial final-report artifact writer
- shared temporal metrics and clustered bootstrap helpers
- deterministic intervention manifest creation from baseline artifacts
- actual Qwen vision-block attention capture path for eager attention backends
- canonical temporal pooling for captured vision attention chunks
- repeated-frame and reversed-video control manifests
- mismatched-query manifests with runner-side question overrides
- pre-encoder keep/pruning support distinct from pre-encoder masking
- frozen decoder reference-layer selection from development artifacts

Pending:

- causal intervention run matrix
- full statistical analysis and final paper figures

## Engineering Commands

Manifest dry engineering path on the GPU Pod:

```bash
python scripts/create_experiment1_v2_manifest.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --output-dir outputs/experiment1_v2 \
  --seed 20260830 \
  --dev-fraction 0.2
```

After manifests exist, run a small engineering validation before any pilot:

```bash
python scripts/run_experiment1.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --limit 3 \
  --num-frames 32 \
  --resolution-config low \
  --attention-extraction reduced_sdpa \
  --query-scope question \
  --resume \
  --allow-7b-inference \
  --output-dir outputs/experiment1_v2/engineering_low_f32_baseline
```

Do not run the full pilot until v2 encoder attention capture, controls, interventions, and completeness checks are implemented.

Generate the expected matrix and completeness report:

```bash
python scripts/analyze_experiment1_v2.py \
  --primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --output-root outputs/experiment1_v2/runs \
  --final-dir outputs/experiment1_v2/final \
  --bootstrap-replicates 10000
```

Create held-out intervention manifests from completed baseline artifacts:

```bash
python scripts/create_experiment1_v2_intervention_manifest.py \
  --primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --baseline-output-dir outputs/experiment1_v2/runs/baseline \
  --output-jsonl outputs/experiment1_v2/interventions/mask_top20.jsonl \
  --condition mask_top20 \
  --strategy top \
  --removal-fraction 0.2 \
  --ranking-layer -1 \
  --seed 20260830
```

Create control manifests:

```bash
python scripts/create_experiment1_v2_control_manifest.py \
  --primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --mismatched-queries outputs/experiment1_v2/mismatched_queries.json \
  --output-jsonl outputs/experiment1_v2/controls/mismatched_query.jsonl \
  --control mismatched_query
```

Freeze the decoder reference layer after development baseline and mismatched
control artifacts exist:

```bash
python scripts/select_experiment1_v2_reference_layer.py \
  --primary-manifest outputs/experiment1_v2/primary_manifest.jsonl \
  --baseline-output-dir outputs/experiment1_v2/runs/baseline \
  --mismatched-output-dir outputs/experiment1_v2/runs/mismatched_query \
  --output-json outputs/experiment1_v2/frozen_reference_layer.json
```
