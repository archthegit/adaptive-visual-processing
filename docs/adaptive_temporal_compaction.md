# Adaptive Temporal Compaction Protocol

## Scientific Question

Can a video-language decoder use intermediate decoder states to choose when to compact, how much temporal context to retain, and which native temporal cells to retain, while physically reducing later decoder computation and preserving answer quality?

This experiment extends the fixed Qwen temporal compaction pilot. It reuses the existing physical compaction implementation in `src/experiment1/temporal_handoff.py`. It does not use attention masking as a substitute for compaction and does not directly copy K/V tensors across decoder layers.

## Splits and Leakage Policy

The adaptive experiment uses deterministic source-video-disjoint splits:

| Split | Per category | Total |
|---|---:|---:|
| train | 50 | 200 |
| development | 20 | 80 |
| test | 50 | 200 |

The four categories are `gaze`, `ingredient`, `fine_grained`, and `object_motion`.

Rules:

- No source video may appear in more than one split.
- Question IDs are disjoint across splits.
- A source video may contribute at most four questions.
- The manifests preserve category, question type, participant, duration interval, choices, and answer metadata.
- `summary.json` reports pairwise source-video overlap counts and must assert that all are zero.
- The test split must not be passed to training, router fitting, threshold selection, or development scripts until the router and decision thresholds are frozen.

Create manifests:

```bash
python scripts/create_adaptive_compaction_splits.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --output-dir outputs/experiment1_v3_adaptive_compaction/manifests \
  --seed 20260928
```

## Intervention Grid

The frozen grid is:

- Frame counts: `8`, `16`, `32`
- Compaction boundaries after decoder layers: `4`, `8`, `12`, `16`, `20`
- Retention fractions: `0.25`, `0.50`, `0.75`
- Primary condition: physical hard deletion

The runner uses `adaptive_fixed_count` sampling, not `cross_model_8`. For each requested frame count it partitions the annotated interval into exactly `8`, `16`, or `32` equal-time bins and takes one deterministic center frame per bin. It asserts that the decoded batch contains exactly that many distinct valid frames before writing any artifact. The runner derives the native temporal-cell count from the actual processed token layout after sampling. It does not assume that `frame_count / 2` is correct without validation.

## Candidate Route Construction

For each question, frame count, boundary layer, and retention budget:

1. Compute the retained-cell count from the native temporal-cell count and retention fraction.
2. If the action space is small, enumerate all retained-cell subsets.
3. If exhaustive enumeration is too large, construct a deterministic candidate set containing:
   - uniform temporal coverage,
   - prefix retention,
   - suffix retention,
   - contiguous windows,
   - deterministic random subsets.
4. Cap candidates at 24.
5. Assign every route a stable SHA-derived `action_id`.

The same question and seed produce the same candidates regardless of shard count or execution order.

Approximate action counts, assuming Qwen maps 8, 16, and 32 sampled frames to 4, 8, and 16 native temporal cells respectively:

| Frame count | Native cells estimate | Actions/question |
|---|---:|---:|
| 8 | 4 | 70 |
| 16 | 8 | 360 |
| 32 | 16 | 360 |

Estimated training total: `200 * (70 + 360 + 360) = 158,000` action labels, plus dense passes and router features.

## Prefix Caching Semantics

For each question and frame count:

1. Run the dense custom Qwen decoder once.
2. Verify dense custom logits against stock Qwen using the existing BF16-aware equivalence gate.
3. Cache dense hidden states and position IDs after each candidate boundary.
4. For an action at boundary `L`, compact the cached output of layer `L`, recompute rotary embeddings for the compacted sequence, and execute only layers `L+1` through the final decoder layer.
5. Validate cached-prefix outputs against full-prefix intervention execution on synthetic and smoke-test cases.

Cached hidden states are in memory only during execution and are not serialized to JSON.

## Router Features

At every candidate boundary, the dense pass saves compact feature tensors:

- mean residual vector per native temporal cell,
- mean residual vector for question tokens,
- residual norm per temporal cell,
- within-cell residual dispersion,
- temporal-cell IDs,
- token count per cell,
- decoder layer,
- frame count.

Feature files are written as `.pt` tensors. JSON artifacts store only metadata and feature paths.

## Action Artifact Schema

Every completed action artifact records:

- question ID and source video ID,
- split,
- frame count,
- compaction layer,
- native temporal-cell count,
- retention fraction,
- retained cell IDs,
- route family and stable action ID,
- correct-choice full-vocabulary log probability,
- paired change from dense,
- answer margin and paired margin change,
- predicted answer and correctness,
- prediction-changed flag,
- original and compacted sequence lengths,
- active sequence length at each decoder layer,
- estimated QK plus AV attention FLOPs,
- paired FLOP reduction from dense,
- memory-token count,
- execution status,
- git commit,
- complete immutable run configuration.

## Smoke Command

Run one training question, eight frames, all five depths, all three budgets, and frozen candidate routes:

```bash
python scripts/run_qwen_adaptive_compaction.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3_adaptive_compaction/manifests/train.jsonl \
  --split train \
  --output-dir outputs/experiment1_v3_adaptive_compaction/smoke \
  --frame-counts 8 \
  --smoke
```

The smoke run writes `smoke_validation.json` and exits nonzero if any required check fails. It verifies sampled frame count, contiguous native temporal-cell IDs, dense equivalence, cached-prefix versus full-prefix physical intervention equivalence at every candidate boundary, physical sequence shortening, finite answer metrics, finite router features after save/reload, artifact resumption, immutable-configuration mismatch detection, and zero source-video/question leakage in the manifest summary.

## Sharded Train Commands

Eight-frame train labels:

```bash
python scripts/run_qwen_adaptive_compaction.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3_adaptive_compaction/manifests/train.jsonl \
  --split train \
  --frame-counts 8 \
  --num-shards 4 \
  --shard-index 0
```

Sixteen-frame train labels:

```bash
python scripts/run_qwen_adaptive_compaction.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3_adaptive_compaction/manifests/train.jsonl \
  --split train \
  --frame-counts 16 \
  --num-shards 4 \
  --shard-index 0
```

Thirty-two-frame train labels:

```bash
python scripts/run_qwen_adaptive_compaction.py \
  --questions-dir /workspace/data/hd-epic-annotations/vqa-benchmark \
  --mp4-dir /workspace/data/hd_epic_mp4 \
  --manifest outputs/experiment1_v3_adaptive_compaction/manifests/train.jsonl \
  --split train \
  --frame-counts 32 \
  --num-shards 4 \
  --shard-index 0
```

Repeat with `--shard-index 1`, `2`, and `3` for the remaining shards.

## Stop/Go Criteria

Before opening the test split:

1. Splits must have zero source-video overlap.
2. Dense custom equivalence must pass on smoke examples.
3. Cached-prefix intervention outputs must match full-prefix intervention outputs on validation cases.
4. Router feature tensors must be finite and have expected shapes.
5. Sharded resume must reject immutable configuration drift.
6. Development router and thresholds must be frozen.

No test split command should be run before these conditions are met.

## Current Validation Status

The implementation includes CPU/unit tests for split generation, exact adaptive frame-count sampling, action generation, cached-prefix equivalence, complete per-layer compacted instrumentation, router feature shape, resume configuration mismatch, dense resume artifact validation, and compact artifact schema. Real-checkpoint smoke testing requires an A100 environment with Qwen and local HD-EPIC videos.

At the time of this corrective commit, the local development machine does not provide the A100, model cache, and HD-EPIC video paths required to run the real smoke. The smoke command above is the required A100 validation before launching the training corpus. Do not treat this branch as ready for the full train sweep until `smoke_validation.json` passes and the observed runtime/storage estimates are recorded from that A100 run.
