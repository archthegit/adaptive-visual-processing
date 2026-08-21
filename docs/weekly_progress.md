# Weekly Progress Tracker

Use this file to keep a concise record of research progress and make weekly updates easier to assemble.

## Current Status

Project: query-relevant adaptive visual processing for HD-EPIC VQA  
Baseline VLM: Qwen2.5-VL-7B-Instruct  
Dataset: HD-EPIC VQA  
Current phase: Experiment 1 temporal engineering run analysis, encoder temporal representation analysis, and causal intervention preparation

## Week of 2026-08-17

### Completed

- Pulled in the first 32-frame temporal engineering run for Experiment 1:
  - output bundle: `results/experiment1_temporal_engineering_f32`
  - manifest: `results/experiment1_temporal_engineering_f32/manifest.jsonl`
  - run summary: `results/experiment1_temporal_engineering_f32/run_summary.json`
  - decoder temporal summary: `results/experiment1_temporal_engineering_f32/analysis_summary.json`
  - encoder temporal analysis: `results/experiment1_temporal_engineering_f32/encoder_analysis/encoder_analysis.json`
- Ran the 8-example engineering set with Qwen2.5-VL-7B-Instruct:
  - 2 examples per category: `fine_grained`, `gaze`, `ingredient`, `object_motion`
  - 32 sampled frames per example
  - low resolution
  - `reduced_sdpa` attention extraction
  - no visual-access cutoff and no temporal-bin masking
  - 16 Qwen temporal bins and 4096 visual tokens per example
- Generated decoder temporal plots:
  - aggregate entropy by category
  - aggregate top-1 temporal-bin mass by category
  - aggregate bins-to-80%-mass by category
  - per-example layer x temporal-bin heatmaps
  - per-example temporal-bin rank trajectories
- Added and ran encoder temporal representation analysis:
  - script: `scripts/analyze_encoder_temporal.py`
  - per-example metrics: `encoder_metrics_per_example.csv`
  - temporal-lag profiles: `encoder_lag_profiles.csv`
  - aggregate plots for pairwise cosine, lag cosine, local temporal advantage, and effective rank
- Confirmed encoder canonicalization is being recorded in real artifacts:
  - early/middle/late vision blocks report `canonical_order_recovered: true`
  - pre-reverse merger reports `canonical_order_recovered: true`
  - final canonical visual output reports `canonical_order_recovered: false`, as expected
- Added follow-up correctness fixes before using these outputs for the pilot:
  - Qwen vision reverse-index recovery now searches nested `*.visual` modules and fails loudly for video input if reverse indices cannot be recovered.
  - Decoder direct-access masking test now uses non-visual query rows, preserving the rule that only non-visual query rows lose access to selected visual columns.
  - Answer scoring now accepts mapping-like tokenizer outputs, matching Hugging Face `BatchEncoding` behavior.

### Experiments / Results

- Engineering run accuracy:
  - completed: 8/8
  - failed: 0
  - correct: 2/8
  - accuracy: 25%
  - category accuracy: `fine_grained` 1/2, `gaze` 1/2, `ingredient` 0/2, `object_motion` 0/2
- Decoder temporal relevance becomes more concentrated across depth:
  - layer 0: top-1 temporal-bin mass 0.079, entropy 0.997, bins-to-80% mass 12.88, absolute visual mass 0.347
  - layer 7: top-1 0.168, entropy 0.968, bins-to-80% 11.88, absolute visual mass 0.071
  - layer 19: top-1 0.150, entropy 0.955, bins-to-80% 11.00, absolute visual mass 0.247
  - layer 27: top-1 0.207, entropy 0.937, bins-to-80% 11.12, absolute visual mass 0.175
  - peak top-1 temporal-bin mass and lowest entropy both occur at layer 27.
- Final-layer category patterns are descriptive only because this is 2 examples/category:
  - `fine_grained`: top-1 0.192, entropy 0.927, bins-to-80% 10.5
  - `gaze`: top-1 0.181, entropy 0.956, bins-to-80% 11.5
  - `ingredient`: top-1 0.191, entropy 0.949, bins-to-80% 11.5
  - `object_motion`: top-1 0.265, entropy 0.917, bins-to-80% 11.0
- Encoder temporal representation analysis:
  - early vision block representations are highly similar in raw cosine space but still show local temporal structure after centering:
    - raw adjacent cosine 0.9954 vs raw non-adjacent cosine 0.9919
    - centered adjacent mean 0.3285 vs centered non-adjacent mean -0.1126
    - centered adjacent advantage 0.4410
    - effective rank 3.52
  - middle vision block has the clearest distributed temporal structure:
    - raw adjacent cosine 0.9649 vs raw non-adjacent cosine 0.9490
    - centered adjacent advantage 0.3305
    - nearest-neighbor-is-adjacent fraction 0.453
    - effective rank 9.18
  - late vision block collapses toward a dominant direction:
    - raw adjacent/non-adjacent cosines are both ~1.0
    - effective rank 1.14
    - PC1 variance fraction 0.982
  - merger/final representations recover broader temporal structure:
    - raw adjacent cosine 0.9326 vs raw non-adjacent cosine 0.9060
    - centered adjacent advantage 0.3115
    - effective rank 9.28
    - PC1 variance fraction 0.312
- Interpretation from the engineering run:
  - Decoder query-to-visual relevance does not strongly select one temporal bin early; it gradually concentrates and is most selective by the final decoder layer.
  - Encoder temporal structure is not monotonic across vision depth: middle blocks and final/merged visual outputs preserve richer temporal variation, while late block outputs look highly collapsed before merger/final projection.
  - The engineering results are enough to justify causal temporal-bin interventions, but not enough for task/category conclusions.

### Code / Infrastructure

- Added `scripts/analyze_encoder_temporal.py` for encoder-stage temporal analysis and bootstrap summaries.
- Stored engineering results under `results/experiment1_temporal_engineering_f32` so the analysis is versioned separately from transient Colab `outputs`.
- Generated plots:
  - `results/experiment1_temporal_engineering_f32/temporal_plots/*.png`
  - `results/experiment1_temporal_engineering_f32/encoder_analysis/plots/*.png`
- Answer-score comparison semantics are now explicit:
  - decoder direct-access masking uses same-artifact `intervention_answer_choice_scores`
  - pre-encoder masking compares masked artifact `answer_choice_scores` against the matching baseline artifact

### Blockers / Dependencies

- This is an engineering set, not a confirmatory run.
- The run is only 8 examples, so category-level differences are descriptive and should not be treated as evidence.
- Accuracy remains low at 32 frames/low resolution, especially for `ingredient` and `object_motion`.
- Pre-encoder masking zeros sampled frames and tests evidence necessity; it does not yet test computational savings from skipping encoding.
- No 48-example pilot should be treated as final until decoder direct-access and pre-encoder temporal intervention comparisons are run against the engineering baseline.

### Next Steps

- Create intervention manifests from the engineering baseline:
  - top final-layer temporal bins
  - bottom final-layer temporal bins
  - random temporal bins with fixed seed
- Run decoder direct-access masking on the 8 engineering examples and compare same-artifact intervention answer scores.
- Run pre-encoder temporal masking on the 8 engineering examples and compare masked artifacts to baseline artifacts.
- Use the engineering intervention deltas to decide whether the 48-example, 128-frame pilot should use final-layer ranking, middle-layer ranking, or both.
- Only after the intervention sanity checks, launch the 48-example non-confirmatory pilot.

## Week of 2026-08-09

### Completed

- Ran real Qwen2.5-VL-7B-Instruct Experiment 1 debug jobs on Colab A100.
- Fixed Colab/GPU execution issues:
  - Qwen video processor `fps` validation for one-video and repeated multi-video inputs
  - question-token layout detection after Qwen visual-token expansion
  - `bfloat16` attention conversion before NumPy aggregation
  - truncated answer generation by increasing `max_new_tokens` and tightening the prompt
- Added a video-aware Experiment 1 mini-pilot builder to avoid downloading the full 70.1 GiB balanced-pilot video set.
- Generated and ran a 12-example video-aware mini pilot using 3 already-downloaded HD-EPIC MP4s.
- Added notebook plots for query-to-visual fusion across decoder depth:
  - top-1 temporal-bin relevance and entropy by layer
  - layer x temporal-bin relevance heatmap
  - category-wise temporal concentration curves
- Added a maintained Experiment 1 output summarizer script to replace fragile notebook one-liners.
- Refactored relevance construction so reduced token scores can be converted into full layer/frame/spatial summaries without requiring full attention tensors at that boundary.
- Implemented an opt-in `reduced_sdpa` Qwen attention extraction mode that computes normal attention output with SDPA and captures only question-token rows x visual-token columns for relevance.
- Connected `--vision-access-through-layer` to Qwen decoder attention during both prefill and generation using an additive text-to-visual key mask after the cutoff layer.
- Fixed the critical causal-mask registration bug for custom Qwen attention backends. Before this fix, `reduced_sdpa` and cutoff runs could attend to future prompt tokens during prefill and are invalid.
- Changed Experiment 1 frame budgeting so `--num-frames` is a total budget split across visual inputs by default, preventing two-video object-motion questions from receiving double the frames/tokens. Legacy per-input behavior remains available via `--frame-budget-mode per-input`.
- Fixed mixed HD-EPIC inputs: `time` inputs are now sampled once and sent to Qwen as images, while interval inputs are sent as videos and consume the video-frame budget.
- Updated Experiment 1 summaries and artifact comparison to use per-input frame relevance and compare compact prefill top-k logits.
- Added absolute visual-attention mass by decoder layer, in addition to visual-token-normalized relevance.
- Preserved visual-token layout metadata with explicit `video_input_index`, `temporal_bin`, `spatial_row`, `spatial_col`, and represented sampled frame indices/timestamps.
- Added spatial heatmap overlay plotting for Experiment 1 artifacts.
- Updated architecture notes with the exact memory blocker for scalable attention extraction.
- Ran the first post-fix visual-access cutoff mini-pilot on Colab A100:
  - `none`, `early`, `middle`, and `late`
  - low resolution
  - 4 sampled frames
  - 12 single-real-video examples
  - `reduced_sdpa` attention extraction

### Experiments / Results

- Current post-fix single-real-video mini pilot:
  - size: 12 examples
  - categories: `fine_grained` 3, `gaze` 3, `ingredient` 3, `object_motion` 3
  - real video inputs per example: 1
  - videos required: 3 MP4s, all already present
  - visual tokens: 512 and 768 depending on whether the example includes an image/reference input
  - stored summary: `outputs/experiment1_colab_low_f4_cutoff_summary.json`
- Full-vs-reduced post-fix validation:
  - `fine_grained_action_localization_1309`: max normalized-frame-score diff 0.0236, aggregate-frame diff 0.0057, absolute-visual-mass diff 0.0207
  - `ingredient_ingredient_retrieval_42`: max normalized-frame-score diff 0.0400, aggregate-frame diff 0.0015, absolute-visual-mass diff 0.0283
  - top-k logits are not identical, so `reduced_sdpa` is acceptable for exploratory relevance sweeps but not yet final-grade accuracy claims
- Low resolution, 4 sampled frames, post-fix visual-access cutoff sweep:
  - no cutoff: 1/12 accuracy
  - early cutoff: 0/12 accuracy
  - middle cutoff: 1/12 accuracy
  - late cutoff: 1/12 accuracy
  - category accuracy for no/middle/late: `fine_grained` 0/3, `gaze` 0/3, `ingredient` 1/3, `object_motion` 0/3
  - category accuracy for early: all categories 0/3
- No-cutoff layer-fusion summary:
  - layer 0 mean top-1 frame mass 0.547, entropy 0.972, absolute visual mass 0.108
  - layer 18 mean top-1 frame mass 0.598, entropy 0.917, absolute visual mass 0.131
  - layer 27 mean top-1 frame mass 0.627, entropy 0.919, absolute visual mass 0.155
  - peak top-1 frame mass layer: 27
  - lowest entropy layer: 18
- Cutoff masking sanity:
  - early cutoff zeros visual relevance after layer 6
  - middle cutoff zeros visual relevance after layer 13
  - late cutoff zeros visual relevance after layer 20
  - these zeros are imposed by the intervention and should not be interpreted as natural model behavior
- Superseded pre-fix / exploratory runs:
  - Low resolution, 4 sampled frames before causal-mask and mixed-input fixes:
  - completed: 12/12
  - accuracy: 2/12
  - visual tokens: 512 for single-input examples, 1024 for two-input examples
  - Low resolution, 8 sampled frames before causal-mask and mixed-input fixes:
  - completed: 12/12
  - accuracy: 3/12
  - category accuracy: `fine_grained` 0/3, `gaze` 1/3, `ingredient` 1/3, `object_motion` 1/3
  - visual tokens: 1024 for single-input examples, 2048 for two-input examples
  - Medium resolution, 8 sampled frames before scalable attention validation:
  - completed: 9/12
  - accuracy on completed examples: 1/9
  - single-input visual tokens: 2704
  - two-input object-motion examples failed with CUDA OOM on A100 40GB under naive full-attention extraction
- Invalidated vision-access validation on `fine_grained_action_localization_1309` before causal-mask fix:
  - no cutoff with `reduced_sdpa`: predicted index 3
  - early cutoff with `reduced_sdpa`: predicted index -1
  - relevance comparison showed large changes, including max normalized-frame-score difference 0.757 and absolute visual-mass difference 0.488
  - status: invalid because custom attention names had not yet been registered with Transformers' causal-mask interface

### Initial Insights

- Query-to-visual relevance in Qwen2.5-VL is not uniform across decoder depth.
- In the post-fix no-cutoff low-resolution 4-frame mini pilot, query-to-visual relevance becomes more temporally concentrated through decoder depth: mean top-1 frame mass rises from 0.547 at layer 0 to 0.627 at layer 27.
- The lowest no-cutoff entropy occurs around layer 18, suggesting the strongest temporal concentration appears in later middle/late decoder layers on this tiny pilot.
- The first visual-access cutoff result suggests direct visual-token access remains useful beyond early decoder layers: early cutoff drops accuracy from 1/12 to 0/12, while middle and late cutoffs recover the no-cutoff 1/12.
- The current pilot does not support a precise cutoff-depth or category-specific claim because baseline accuracy is too low and there are only three examples per category.
- Fine-grained and gaze remain 0/3 at low resolution and 4 frames; this supports increasing visual evidence/resolution before drawing task-level conclusions.
- Absolute visual-attention mass is now logged because within-visual normalization alone cannot answer whether the model's total reliance on visual tokens decreases by layer.
- Earlier reduced/cutoff findings before causal-mask registration should not be used as scientific evidence.

### Code / Infrastructure

- Current full-attention debug extractor remains the correctness baseline for small runs.
- New `--attention-extraction reduced_sdpa` mode avoids returning/storing full attention tensors and captures reduced question-to-visual relevance per layer while preserving causal masks via registered mask functions.
- New visual-access cutoff path masks text/query attention to visual-token keys after `none/early/middle/late` or an explicit layer index. Local tests prove attention outputs change after the cutoff and remain unchanged before the cutoff.
- The reduced extractor has local synthetic correctness tests, including a causal-mask test, and Colab full-vs-reduced validation on two HD-EPIC examples. It still needs broader validation before final accuracy claims.
- Mixed image/video inputs are represented in token layout with original input indices, separate image/video grids, and per-input relevance fields.

### Blockers / Dependencies

- Full balanced 48-example pilot currently requires 72 unique MP4s, about 70.1 GiB, which is too slow for tight Colab iteration.
- Medium/high resolution and multi-real-video examples require validated memory-efficient attention extraction before scaling.
- Faithful layer-wise visual-access intervention is implemented at the Qwen attention-interface level and has post-fix real-Qwen Colab validation for `none/early/middle/late` on the 12-example mini pilot. Hidden-state zeroing remains intentionally avoided.
- Vision-encoder and merger tracing is not implemented yet; Experiment 1 relevance still begins at decoder self-attention over inserted visual-token positions.

### Next Steps

- Produce a compact table of per-question predictions for `none/early/middle/late` to identify which examples change under cutoff.
- Run the same single-real-video mini pilot at 8 frames or medium resolution to test whether the very low baseline accuracy is caused by insufficient visual evidence.
- Validate `reduced_sdpa` against full attention on a few more examples if final accuracy claims will use reduced extraction.
- Add or export plots for the final post-fix `none/early/middle/late` layer-depth result.
- Expand from the mini pilot to a larger video-aware subset before attempting the full balanced pilot.

## Week of 2026-08-03

### Completed

- Inspected the official HD-EPIC VQA evaluation and annotation repositories.
- Confirmed the public VQA annotation schema:
  - examples are keyed by question ID
  - each example includes `inputs`, `question`, five `choices`, and `correct_idx`
  - visual inputs include video IDs plus optional temporal metadata
- Built standalone research scaffold:
  - dataset parsing
  - modular frame sampling
  - Qwen2.5-VL wrapper boundary
  - structured JSONL outputs
  - query-relevance stub
  - future temporal propagation interface
- Added local tests that do not require HD-EPIC videos or GPU inference.
- Added a synthetic MP4 frame-sampling test using `ffmpeg`.
- Verified Qwen wrapper import/initialization without checkpoint loading.
- Generated deterministic 50-example pilot manifest from official annotations.
- Produced full dataset counts by all 30 question types and broader categories.

### Pilot Subset

- Manifest: `outputs/pilot_manifest.jsonl`
- Summary: `outputs/pilot_summary.json`
- Seed: `20260806`
- Size: 50 examples
- Pilot distribution:
  - `3d_perception`: 6
  - `fine_grained`: 12
  - `gaze`: 10
  - `ingredient`: 12
  - `object_motion`: 10

### Verification

Commands run successfully:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/inspect_dataset.py --questions-dir /private/tmp/hd-epic-annotations/vqa-benchmark --limit 2
.venv/bin/python scripts/run_baseline.py --questions-dir /private/tmp/hd-epic-annotations/vqa-benchmark --limit 5 --dry-run --output-dir outputs/qwen_smoke
.venv/bin/python scripts/create_pilot_manifest.py --questions-dir /private/tmp/hd-epic-annotations/vqa-benchmark --pilot-size 50 --seed 20260806 --output-jsonl outputs/pilot_manifest.jsonl --summary-json outputs/pilot_summary.json
```

Test result:

```text
11 passed
```

### Blockers / Dependencies

- HD-EPIC videos are still needed for real frame loading over dataset examples.
- CUDA and Qwen2.5-VL model weights are needed for actual 7B baseline inference.
- Query-relevance implementation is intentionally paused until the updated experiment definition is received.

### Next Steps

- Review pilot subset with Shivani / Oxford collaborators.
- Confirm which VQA families should be prioritized for the first relevance experiment.
- Once experiment definition is finalized, implement the first `compute_relevance(question, frames, model_outputs)` method behind the existing interface.
- After videos/checkpoints are available, run a small non-dry-run smoke baseline on the pilot subset.

## Weekly Template

Copy this section for future updates.

## Week of YYYY-MM-DD

### Completed

- 

### Experiments / Results

- 

### Code / Infrastructure

- 

### Blockers / Dependencies

- 

### Next Steps

- 
