# State-Preserving Temporal Handoff Design

## Scope

This document designs the central Experiment 1 method contribution: decoder-side state-preserving temporal handoff for Qwen2.5-VL. It is a physical sequence-compaction method. It is not hard attention masking, not zeroing, not pre-encoder frame removal, and not direct cross-layer KV copying.

The design is based on the current `experiment1-temporal` branch, especially:

- `src/experiment1/qwen_execution.py`
- `src/experiment1/qwen_reduced_attention.py`
- `src/experiment1/token_layout.py`
- `src/experiment1/temporal.py`
- `src/experiment1/route_reuse.py`
- `src/models/qwen.py`
- `docs/experiment1.md`
- `docs/experiment1_architecture.md`
- `docs/experiment1_v3.md`
- `docs/experiment1_weekly_report.md`

Local inspection note: the local environment used for this design does not have `transformers` installed, so live source introspection of the installed Qwen classes could not be executed here. The design therefore includes a required one-example real-checkpoint smoke test to validate every Qwen class signature and tensor shape before any multi-example run.

## Existing Qwen Data Path

The current runner builds Qwen inputs in `run_qwen_relevance_example()`:

1. HD-EPIC frames are sampled into one or more `FrameBatch` objects.
2. Qwen messages are constructed with `{"type": "video", "video": PIL frames, "fps": effective_sample_fps}` and a final text prompt.
3. `AutoProcessor.apply_chat_template(..., add_generation_prompt=True)` renders text.
4. `qwen_vl_utils.process_vision_info()` returns image/video inputs and processor video kwargs.
5. `model._processor(..., return_tensors="pt")` produces tensors including:
   - `input_ids`
   - `attention_mask`
   - `pixel_values_videos`
   - `video_grid_thw`
   - `second_per_grid_ts`
   - optionally `mm_token_type_ids`
6. `apply_corrected_second_per_grid_ts()` overwrites `second_per_grid_ts` with `temporal_patch_size / effective_sample_fps`.
7. `build_token_layout()` maps decoder-sequence visual placeholders to temporal/spatial Qwen visual cells.
8. Qwen inserts merged video features at positions where `input_ids == video_token_id`.
9. The language decoder consumes one sequence containing visual placeholder embeddings and text tokens.

Current route-reuse interventions operate only by additive attention masks inside `qwen_reduced_attention.py`; they do not change `hidden_states.shape[1]`. Temporal handoff must be implemented below this level, after visual features have been inserted into decoder hidden states and after some decoder layers have already run.

## Tensor Shapes and Token Ordering

Use these symbols:

| Symbol | Meaning |
|---|---|
| `B` | batch size; Experiment 1 uses `B = 1` |
| `S0` | original expanded decoder sequence length after visual placeholder expansion |
| `S_l` | physical decoder sequence length entering decoder layer `l` |
| `V` | number of video visual tokens in the original decoder sequence |
| `Q` | number of natural-language question tokens |
| `D` | decoder hidden size |
| `L` | Qwen text decoder layers; current artifacts use `L = 28` |
| `Hq` | query attention heads |
| `Hkv` | key/value heads before grouped-query repeat |
| `Dh` | head dimension |
| `Tq` | Qwen native temporal cells from `video_grid_thw[0][0]` |
| `Ha`, `Wa` | analysis spatial grid after `spatial_merge_size` |

### Processor outputs

For one example:

| Tensor / object | Expected shape |
|---|---|
| `input_ids` | `[1, S0]` |
| `attention_mask` | `[1, S0]` before model-internal causal expansion |
| `video_grid_thw` | `[num_video_inputs, 3]`; each row `[T_raw, H_raw, W_raw]` |
| `second_per_grid_ts` | `[num_video_inputs]` |
| `pixel_values_videos` | model/processor-specific flattened video patches |

The current code corrects `second_per_grid_ts` before forward. The temporal mRoPE compatibility patch changes the Qwen computation from `tokens_per_second * int(second_per_grid_ts)` to `int(tokens_per_second * float(second_per_grid_ts))`.

### Expanded decoder sequence

`TokenLayout` treats the final decoder input as one ordered sequence:

```text
[system / chat / prompt prefix tokens]
+[video visual placeholder token span]
+[question and answer-choice prompt text]
+[assistant generation prefix]
```

The exact placement depends on Qwen's chat template, so the implementation must derive token positions from `input_ids`, `video_token_id`, `image_token_id`, and `mm_token_type_ids`, not from string offsets alone.

Current layout fields:

| Field | Meaning |
|---|---|
| `question_token_indices` | absolute decoder positions for natural-language question rows |
| `prompt_token_indices` | `tuple(range(S0))` |
| `visual_token_indices` | absolute decoder positions for video/image visual placeholder columns |
| `visual_cells` | one `VisualTokenCell` per visual token |

### Visual token cell order

`cells_from_grid_specs()` consumes visual placeholders in Qwen temporal-major order:

```python
for temporal_index in range(Tq):
    for spatial_y in range(H_raw // spatial_merge_size):
        for spatial_x in range(W_raw // spatial_merge_size):
            visual token
```

Each `VisualTokenCell` records:

| Field | Meaning |
|---|---|
| `token_index` | absolute decoder sequence index |
| `visual_index` | index within the visual-token block, `0..V-1` |
| `temporal_index` | Qwen native temporal cell |
| `spatial_y`, `spatial_x` | merged spatial grid coordinates |
| `grid_t`, `grid_h`, `grid_w` | native temporal and merged spatial grid shape |
| `seconds_per_grid`, `timestamp` | Qwen temporal metadata |

For an 8-frame cross-model Qwen run, Qwen may merge eight sampled frames into four native temporal cells. The handoff method must route by native temporal regions or analysis bins explicitly, and must record both.

### Decoder layer tensors

At layer `l` before compaction:

| Tensor | Shape |
|---|---|
| `hidden_states_l` | `[B, S_l, D]` |
| decoder layer output | `[B, S_l, D]` |
| `query_states` | `[B, Hq, q_len, Dh]` |
| `key_states` | `[B, Hkv, k_len, Dh]` before GQA repeat |
| repeated `key_states` | `[B, Hq, k_len, Dh]` |
| `value_states` | `[B, Hkv, k_len, Dh]` before GQA repeat |
| repeated `value_states` | `[B, Hq, k_len, Dh]` |
| causal/additive attention mask | `[B, 1, q_len, k_len]` |
| prefill `position_ids` | Qwen multimodal RoPE IDs, expected `[3, B, S_l]` or model-equivalent |
| generation `cache_position` | model-dependent, normally `[S_l]` for prefill and `[1]` for one-token decode |

Current reduced attention uses query rows and key columns with `q_len`/`k_len` inferred from actual tensors, not from mRoPE position IDs. Handoff must preserve that discipline: position IDs are not sequence indices.

## Temporal and Spatial Mapping

Handoff operates on temporal regions while preserving spatial structure until compression. The mapping source is `TokenLayout.visual_cells`, plus frame/bin metadata in `FrameBatch.metadata["frame_bin_mapping"]`.

For each native temporal region `r`:

```text
native_region_id = temporal_index
tokens(r) = {
  cell.token_index
  for cell in layout.visual_cells
  if cell.modality == "video"
  and cell.input_index == selected_video_input
  and cell.temporal_index in region.temporal_indices
}
spatial positions = (cell.spatial_y, cell.spatial_x)
analysis bins = qwen_temporal_to_analysis_weights(batch, cell.temporal_index, cell.grid_t)
sampled frames = represented_sampled_frames(batch, cell.temporal_index, cell.grid_t)
```

The artifact must record:

- original visual token index
- original decoder token index
- native temporal cell
- analysis bin(s)
- spatial row/column
- sampled frame indices/timestamps represented
- source and replacement sequence positions after every compaction layer

## Handoff Selection

This design reuses the frozen 15-example development manifest:

```text
outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl
```

Held-out causal results must not be inspected while choosing handoff hyperparameters.

Candidate handoff layers:

```text
4, 8, 12, 16, 20
```

Retention ratios:

```text
0.25, 0.50, 0.75
```

Memory budgets per handed-off region:

```text
1, 2, 4 memory tokens
```

For the first development smoke test, select a single handoff layer and a single region from a dense baseline artifact. Only after shape/correctness tests pass should the grid above be scheduled.

## Compression Into Memory Tokens

### Region definition

A handoff region is a contiguous set of temporal cells. For budget matching, regions should be built from the same native temporal units used by prior temporal routing analysis, but the method may merge adjacent selected units into one contiguous handoff region.

Let a region contain original visual token positions:

```text
R = [i_0, i_1, ..., i_{n-1}]
```

and hidden states after handoff layer `h`:

```text
X_R = hidden_states_after_layer_h[:, R, :]  # [B, n, D]
```

### Memory-token initializer

The design should start with a deterministic non-learned initializer so that engineering smoke tests are not confounded by training:

| Memory budget | Initializer |
|---:|---|
| 1 | mean pool over all region tokens: `mean(X_R, dim=1)` |
| 2 | mean pool first and second temporal halves of the region |
| 4 | mean pool four ordered temporal/spatial-balanced shards |

The production implementation can later compare this with a learned attention-pooling compressor, but the first causal pilot should avoid training.

For memory budget `M`, the compressor returns:

```text
memory_states = compress(X_R, M)  # [B, M, D]
```

Each memory token must be marked as a visual-memory token in metadata, not as an original visual token.

### State-preserving meaning

The memory tokens are initialized from same-example, same-forward hidden states after the selected handoff layer. They are not static embeddings and are not copied from another example. After insertion, they participate in all subsequent decoder layers like ordinary sequence tokens:

```text
for layer in h+1 ... L-1:
    memory_states = layer(memory_states, context=compacted_sequence)
```

Thus the compact memory continues evolving through the remaining decoder depth.

## Replacing Original Visual Tokens

At the compaction boundary immediately after layer `h`:

1. Build a sequence rewrite plan over the current sequence positions.
2. For every selected temporal region, remove its original visual token positions.
3. Insert `M` memory tokens at the position of the first removed token.
4. Preserve the relative order of all retained original tokens.
5. Recompute index maps:
   - `old_position -> new_position | removed`
   - `new_position -> original_position | memory_token_id`
   - `original_visual_token -> retained/new/memory-region`

If two adjacent temporal regions are both handed off, they should be merged into one contiguous handoff region for opportunity accounting and to avoid inserting redundant neighboring memory summaries.

Example:

```text
before:
  [prefix, v_t0_s0, v_t0_s1, v_t1_s0, v_t1_s1, question, assistant]

handoff region:
  v_t0_s0, v_t0_s1, v_t1_s0, v_t1_s1

M = 2

after:
  [prefix, mem_region0_0, mem_region0_1, question, assistant]
```

This is physical compaction: `S_{h+1} < S_h`.

## Updating Masks, Positions, and Indices

### Attention masks

After compaction, build fresh causal masks from the new sequence length:

```text
attention_mask_{h+1}  # [B, 1, S_{h+1}, S_{h+1}] for prefill
```

The mask must:

- preserve causal text-token semantics;
- allow visual/memory prefix tokens to be attended by later text tokens;
- allow memory tokens to attend only to allowed earlier/equal-prefix tokens according to Qwen's normal decoder behavior;
- not preserve stale columns for removed tokens.

No additive `-inf` mask to omitted visual tokens should remain in handoff conditions. Removed tokens must be absent from `hidden_states`, keys, values, and attention masks.

### RoPE / mRoPE position IDs

Qwen uses multimodal RoPE position IDs rather than scalar absolute sequence IDs. Compaction requires a new `position_ids` tensor matching the compacted sequence:

```text
position_ids_compacted  # expected [3, B, S_{h+1}]
```

For retained original tokens, copy their original three-axis position IDs.

For memory tokens, the first implementation should use deterministic representative positions from the removed region:

| Memory budget | Position-ID policy |
|---:|---|
| 1 | copy the medoid original visual token position ID from the region |
| 2 | copy medoid position IDs from first and second ordered halves |
| 4 | copy medoid position IDs from four ordered shards |

This avoids inventing fractional or out-of-distribution mRoPE coordinates. The artifact must record the source original token used for each memory token's position ID.

Stop/go condition: if real-checkpoint tests show that copying representative mRoPE positions causes pathological output drift even under very mild compression, a learned or explicit memory-position convention must be designed before scaling.

### Question-token indices

Question token positions shift after visual tokens are removed. Maintain both:

```text
original_question_token_indices
question_token_indices_by_layer[str(layer)]
```

For layers before compaction, use original indices. For layers after compaction, map original question indices through `old_position -> new_position`.

Any reduced-attention or answer-analysis capture after compaction must use the layer-specific question positions. Never reuse pre-compaction absolute indices after sequence shortening.

### Visual-token indices

Maintain:

```text
original_visual_token_indices
retained_original_visual_token_indices_by_layer
memory_visual_token_indices_by_layer
effective_visual_token_indices_by_layer
```

For layers after compaction, `effective_visual_token_indices` includes retained original visual tokens and memory tokens. Removed original visual tokens are not valid key columns.

### Generation caches

This is the main architectural risk.

During prefill with handoff at layer `h`:

- layers `0..h` process dense length `S0`;
- layers `h+1..L-1` process compact length `S'`.

Therefore the natural cache lengths differ by layer:

```text
past_key_values[layer <= h].seq_len == S0
past_key_values[layer > h].seq_len == S'
```

Stock Hugging Face generation usually assumes one shared `attention_mask`, one shared `cache_position`, and a common notion of past sequence length across layers. True mid-decoder compaction therefore requires one of:

1. a custom Qwen text-decoder forward loop that supports layer-specific cache lengths and layer-specific attention masks;
2. a custom cache object accepted by the installed Qwen implementation that can report per-layer sequence lengths and accept per-layer masks;
3. disabling cache during generation and recomputing the full compacting prefill for every generated token, which is correctness-only and not a valid efficiency implementation.

The first implementation should support prefill scoring first. Generation with cache is a stop/go gate before any claim about end-to-end latency.

## Why Earlier-Layer K/V Tensors Are Not Reused Directly

Direct cross-layer KV copying is mathematically invalid for this method:

- each decoder layer has its own `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- the same hidden state projected by layer `h` is not in the key/value space expected by layer `h+k`;
- residual streams and MLP outputs change hidden states between layers;
- RoPE is applied inside each layer's attention projection path;
- attention weights are functions of both current queries and current keys, so earlier weights are not portable to later layers.

State-preserving handoff therefore compresses hidden states in the residual stream, then lets later layers compute their own K/V tensors from the evolved compact sequence.

## Physical Sequence-Length Recording

Every handoff artifact must record:

```json
"temporal_handoff": {
  "schema_version": "temporal_handoff_v1",
  "condition": "...",
  "handoff_layer": 8,
  "retention_ratio": 0.5,
  "memory_tokens_per_region": 2,
  "sequence_length_by_layer": {
    "0": 1234,
    "1": 1234,
    "8": 1234,
    "9": 842,
    "27": 842
  },
  "num_original_visual_tokens_by_layer": {},
  "num_memory_tokens_by_layer": {},
  "num_effective_visual_tokens_by_layer": {},
  "regions": []
}
```

Also record per-region:

- original temporal cells;
- analysis bins;
- original visual token indices;
- removed original decoder positions;
- inserted memory-token positions;
- memory-token position-ID source tokens;
- retained-token fraction;
- compression ratio;
- old/new question-token index maps.

## FLOPs, Latency, and Memory Measurement

### Theoretical attention FLOPs

For each layer, estimate decoder self-attention score/value work as proportional to:

```text
prefill_attention_edges_l = S_l * S_l
prefill_qk_flops_l ~= 2 * B * Hq * S_l * S_l * Dh
prefill_av_flops_l ~= 2 * B * Hq * S_l * S_l * Dh
```

The ratio relative to dense is:

```text
sum_l S_l^2 / (L * S0^2)
```

For generated token step `g`, with `q_len = 1`:

```text
decode_edges_l(g) = 1 * (past_len_l + 1)
```

Report theoretical ratios separately for prefill and generation.

### Wall-clock latency

Measure with CUDA synchronization:

```python
torch.cuda.synchronize()
start = time.perf_counter()
...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Record at least:

- vision preprocessing;
- vision encoder;
- dense decoder prefill;
- handoff compression operation;
- compact decoder suffix;
- answer scoring;
- generation prefill;
- per-token generation loop.

Do not report latency improvement from masking. A latency claim requires observed lower wall-clock time from physical sequence compaction.

### Peak CUDA memory

Reset and record stage-specific peaks:

```python
torch.cuda.reset_peak_memory_stats()
...
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Record CPU RSS through the existing profiler as well. Memory improvement requires lower peak allocated/reserved CUDA memory in the physically compacted path, not just lower theoretical edges.

## Budget-Matched Conditions

The development experiment must compare exactly budget-matched conditions:

| Condition | Meaning |
|---|---|
| dense | no handoff; original sequence all layers |
| hard_eviction | physically remove selected temporal regions after handoff layer with no memory tokens |
| temporal_handoff | replace selected temporal regions by 1/2/4 memory tokens per contiguous region |
| random_handoff | same number of temporal regions and memory tokens, random regions |
| uniform_handoff | same number of temporal regions and memory tokens, deterministic spread |

Budget matching rules:

1. All conditions use the same source frames, prompt, model, resolution, answer scoring, and generation settings.
2. Handoff, random, and uniform select the same number of native temporal regions.
3. For a fixed memory budget, all handoff conditions insert the same number of memory tokens per handed-off contiguous region.
4. Hard eviction removes the same original tokens but inserts zero memory tokens; it is a lower-budget ablation, not the primary equal-budget control.
5. If contiguous selected regions differ by condition and would produce different memory-token counts, split/merge rules must be applied so the total memory-token count is equal, or the candidate comparison is invalid.
6. Record exact `S_l` per layer and reject condition sets whose effective token budgets differ beyond a predeclared tolerance.

## Minimal 15-Example Development Experiment

Use only:

```text
outputs/experiment1_v3_cross_model/manifests/dev_eligible_8frame.jsonl
```

Do not inspect held-out causal outcomes while choosing hyperparameters.

Grid:

| Factor | Values |
|---|---|
| Handoff layer | 4, 8, 12, 16, 20 |
| Retention ratio | 0.25, 0.50, 0.75 |
| Memory tokens per handed-off region | 1, 2, 4 |
| Conditions | dense, hard_eviction, temporal_handoff, random_handoff, uniform_handoff |

Primary development metrics:

- correct-choice log-probability delta from dense;
- answer-margin delta from dense;
- routed/generated predicted choice;
- prediction flip rate;
- exact-match accuracy;
- physical sequence-length reduction;
- theoretical prefill attention-FLOP ratio;
- measured decoder-prefill latency;
- measured peak CUDA memory.

Development selection should prioritize preserving answer quality at real sequence reduction. A setting is not promising unless it shows physical `S_l` reduction and non-worse quality than random/uniform at the same budget.

## Unit Tests

Add synthetic unit tests before any real checkpoint run:

1. `TokenLayout` visual cells map temporal-major token order correctly.
2. Qwen native temporal cells map to planned analysis bins and sampled frames.
3. A compaction plan replaces only selected visual tokens and preserves all nonselected token order.
4. Question-token indices are remapped exactly after compaction.
5. Retained visual-token indices and memory-token indices are disjoint and complete.
6. Causal masks are regenerated with compacted sequence length and contain no removed-token columns.
7. Position IDs for retained tokens are copied unchanged.
8. Memory-token position IDs are copied from recorded representative original tokens.
9. Compaction with an empty selected set is numerically equivalent to dense forward.
10. Memory budget zero with hard eviction physically shortens the sequence and has no memory-token entries.
11. Cache metadata records layer-specific sequence lengths.
12. Generation cache test rejects unsupported stock-cache paths rather than silently falling back to masking.
13. Dense, temporal_handoff, random_handoff, and uniform_handoff budget checks fail if `S_l` differs unexpectedly.
14. No spatial-routing code path is invoked.

Real-checkpoint focused tests:

1. One forward pass through Qwen with no compression matches the existing dense logits within tolerance.
2. Handoff after one layer reduces `hidden_states.shape[1]` before the next layer.
3. The model produces finite answer-choice scores after compaction.
4. Stage-specific CUDA memory and latency are recorded.

## Staged Implementation Plan

### Stage 0: Source validation

On the RunPod environment with the real Qwen checkpoint:

- print the class names and signatures for Qwen model, text model, decoder layer, and attention modules;
- verify where `inputs_embeds`, `position_ids`, `cache_position`, and `past_key_values` enter the text decoder;
- verify whether the cache class permits layer-specific sequence lengths.

Stop if Qwen internals cannot be wrapped without editing site-packages or vendoring a minimal text-decoder loop.

### Stage 1: One-example prefill smoke test

Run one development example:

- baseline dense prefill;
- no-op handoff path with no selected regions;
- compare next-token logits and answer-choice scores;
- assert exact token mappings and unchanged sequence length.

### Stage 2: Physical compaction smoke test

On the same example:

- choose one handoff layer, one temporal region, and `M = 1`;
- compact after the handoff layer;
- assert `S_{h+1} < S_h`;
- assert finite logits;
- record `sequence_length_by_layer`;
- run without generation cache first.

### Stage 3: Cache compatibility smoke test

Implement or reject generation cache support:

- if stock cache supports layer-specific lengths, test one generated token;
- otherwise implement a custom generation loop or mark generation-latency claims blocked;
- do not silently use dense generation for a handoff artifact.

### Stage 4: 15-example development grid

Run the full development grid only after Stages 1-3 pass.

### Stage 5: Development analysis and hyperparameter freeze

Freeze one handoff setting using development examples only. The selected setting must satisfy the stop/go criteria below.

### Stage 6: Held-out run

Only after freeze, run the 56 held-out examples once for the frozen setting and matched controls.

## Stop/Go Criteria Before 56 Held-Out Examples

Go only if all are true:

1. The no-op handoff path matches dense logits within a predeclared tolerance.
2. A real handoff physically reduces `hidden_states.shape[1]` after the selected layer.
3. The artifact records `sequence_length_by_layer` and shows shorter later layers.
4. Question-token and visual-token mappings are valid after compaction.
5. Position IDs are regenerated and validated on the real checkpoint.
6. Answer-choice scoring uses the compacted forward, not an unmodified dense forward.
7. Generation either uses a validated compacted cache path or is explicitly excluded from latency claims.
8. Development temporal_handoff preserves log probability and margin better than budget-matched random and uniform controls.
9. Measured decoder-prefill latency or peak CUDA memory improves on at least a one-example smoke benchmark.
10. No held-out outcomes were inspected during selection.

Stop if any are true:

1. Stock Qwen generation requires one uniform cache length across all layers and cannot be adapted safely.
2. Position-ID handling for memory tokens produces unstable or invalid outputs.
3. The implementation falls back to masking, zeroing, or dense generation while labeling the result as handoff.
4. Budget matching cannot be made exact across adaptive/random/uniform conditions.

## Architectural Obstacles

The principal obstacle is generation with mid-decoder compaction. Prefill can be implemented with a custom layer loop because each layer can receive its own `hidden_states`, `attention_mask`, and `position_ids`. Generation with cache is harder because layers before the handoff have dense-prefix caches while layers after the handoff have compact-prefix caches. If the installed Qwen/Hugging Face cache stack assumes one shared `cache_position` and one shared attention mask length for all layers, genuine mid-decoder sequence compaction requires a custom decoder/generation loop.

This obstacle does not prevent a prefill-only proof of physical compaction. It does prevent any honest end-to-end generation speedup claim until layer-specific cache handling is implemented and validated.

The second obstacle is mRoPE for memory tokens. Memory tokens are not original image patches. The safest initial policy is to copy representative original visual-token position IDs and record those representatives. If that policy fails real-checkpoint smoke tests, the method needs a better memory-position convention before scaling.

The third obstacle is outcome interpretation. A compacted temporal memory token is a new residual-stream state, not a guarantee that all information from removed tokens is preserved. Dense, hard-eviction, temporal_handoff, random_handoff, and uniform_handoff controls are therefore mandatory.
