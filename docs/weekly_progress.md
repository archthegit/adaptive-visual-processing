# Weekly Progress Tracker

Use this file to keep a concise record of research progress and make weekly updates to Shivani easier to assemble.

## Current Status

Project: query-relevant adaptive visual processing for HD-EPIC VQA  
Baseline VLM: Qwen2.5-VL-7B-Instruct  
Dataset: HD-EPIC VQA  
Current phase: repository setup, annotation inspection, baseline scaffolding, and pilot subset preparation

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
