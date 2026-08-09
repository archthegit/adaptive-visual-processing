# Weekly Progress Tracker

Use this file to keep a concise record of research progress and make weekly updates easier to assemble.

## Current Status

Project: query-relevant adaptive visual processing for HD-EPIC VQA  
Baseline VLM: Qwen2.5-VL-7B-Instruct  
Dataset: HD-EPIC VQA  
Current phase: Experiment 1 mini-pilot execution, query-to-visual relevance analysis, and memory-scaling work

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

### Experiments / Results

- Video-aware mini pilot:
  - size: 12 examples
  - categories: `fine_grained` 3, `gaze` 3, `ingredient` 3, `object_motion` 3
  - videos required: 3 MP4s, all already present
- Low resolution, 4 sampled frames:
  - completed: 12/12
  - accuracy: 2/12
  - visual tokens: 512 for single-input examples, 1024 for two-input examples
- Low resolution, 8 sampled frames:
  - completed: 12/12
  - accuracy: 3/12
  - category accuracy: `fine_grained` 0/3, `gaze` 1/3, `ingredient` 1/3, `object_motion` 1/3
  - visual tokens: 1024 for single-input examples, 2048 for two-input examples
- Medium resolution, 8 sampled frames:
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
- In the low-resolution 8-frame mini pilot, temporal concentration is diffuse in the earliest layers, peaks around layer 7, drops in the middle-late layers, and rises again near the final decoder layer.
- Mean top-1 temporal-bin mass changed from 0.585 at 4 sampled frames to 0.341 at 8 sampled frames, indicating that relevance spreads across more Qwen temporal bins when more frames are available.
- The layer x temporal-bin heatmap shows a strong average bias toward the earliest Qwen temporal bin; this could reflect dataset evidence location, model positional bias, or mini-pilot selection bias.
- Fine-grained localization stayed at 0/3 across low 4-frame, low 8-frame, and medium 8-frame settings, so this tiny pilot does not support the idea that global frame count or medium global resolution alone solves fine-grained failures.
- Absolute visual-attention mass is now logged because within-visual normalization alone cannot answer whether the model's total reliance on visual tokens decreases by layer.
- Earlier reduced/cutoff findings should not be used as scientific evidence until rerun after the causal-mask registration fix.

### Code / Infrastructure

- Current full-attention debug extractor remains the correctness baseline for small runs.
- New `--attention-extraction reduced_sdpa` mode avoids returning/storing full attention tensors and captures reduced question-to-visual relevance per layer while preserving causal masks via registered mask functions.
- New visual-access cutoff path masks text/query attention to visual-token keys after `none/early/middle/late` or an explicit layer index. Local tests prove attention outputs change after the cutoff and remain unchanged before the cutoff.
- The reduced extractor has local synthetic correctness tests, including a causal-mask test, but still needs Colab validation against `full` on one 4-8 frame HD-EPIC example before it is used as the main medium/high-resolution path.
- Mixed image/video inputs are represented in token layout with original input indices, separate image/video grids, and per-input relevance fields.

### Blockers / Dependencies

- Full balanced 48-example pilot currently requires 72 unique MP4s, about 70.1 GiB, which is too slow for tight Colab iteration.
- Medium/high resolution and multi-input examples require validated memory-efficient attention extraction before scaling.
- Faithful layer-wise visual-access intervention is implemented at the Qwen attention-interface level, but requires real-Qwen revalidation after the causal-mask fix. Hidden-state zeroing remains intentionally avoided.
- Vision-encoder and merger tracing is not implemented yet; Experiment 1 relevance still begins at decoder self-attention over inserted visual-token positions.

### Next Steps

- Validate the new summarizer on Colab outputs.
- Validate `reduced_sdpa` against the existing full-attention extractor on 4-8 frame examples.
- Validate the layer-wise visual-access intervention on real Qwen by showing logits or predictions differ from the no-intervention run.
- Re-run the video-aware mini pilot with `--attention-extraction reduced_sdpa`, then retry medium-resolution two-input examples.
- Run the low/medium resolution x `none/early/middle/late` visual-access sweep on the completed mini-pilot examples.
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
