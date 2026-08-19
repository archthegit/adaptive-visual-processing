# Experiment 1 Architecture Notes

Local inspection date: 2026-08-08  
Installed source package for inspection: `transformers==5.14.1`  
Qwen utility package installed: `qwen-vl-utils==0.0.14`, but importing it requires `torch`.

No Qwen checkpoint was downloaded. `torch` is not installed in the local venv, so model instantiation and CUDA behavior were not executed locally.

## Relevant Installed Source Files

- `.venv/lib/python3.13/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py`
- `.venv/lib/python3.13/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `.venv/lib/python3.13/site-packages/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py`

## Data Path

The inspected Qwen2.5-VL path is:

```text
single-video pixels
  -> Qwen2_5_VLProcessor / image and video processors
  -> pixel_values_videos + video_grid_thw + second_per_grid_ts
  -> Qwen2_5_VisionPatchEmbed
  -> Qwen2_5_VisionTransformerPretrainedModel.blocks
  -> Qwen2_5_VLPatchMerger
  -> Qwen2_5_VLModel.get_video_features
  -> placeholder positions with input_ids == config.video_token_id
  -> inputs_embeds.masked_scatter(video_mask, video_embeds)
  -> Qwen2_5_VLTextModel decoder layers
  -> lm_head / answer tokens
```

## Temporal Query Relevance Definition

Qwen2.5-VL does not expose a separate text/vision cross-attention block in this implementation. Visual features are projected/merged and inserted into the decoder sequence at visual placeholder token positions.

For Experiment 1, query-to-visual relevance is:

```text
decoder self-attention rows for natural-language question tokens
  x
decoder self-attention columns for visual token positions
```

The result is immediately aggregated to temporal bins by summing all spatial
visual tokens that share the same Qwen `video_grid_thw` temporal index. The
vision encoder attention is not question-conditioned and should not be labeled
query relevance.

Experiment 1 also records non-query-conditioned temporal representations from
the Qwen vision path when the expected modules are present:

```text
visual.patch_embed output -> pool by video_grid_thw temporal bin
visual.merger output      -> pool by merged video_grid_thw temporal bin
```

For each captured stage, artifacts include temporal embedding vectors and
adjacent-bin cosine similarities. These encoder/merger summaries answer a
different question from decoder relevance: whether neighboring temporal bins are
represented similarly before the language decoder consumes them.

## Visual Token Counts and Layout

The processor expands visual placeholders according to:

```text
num_video_tokens = prod(video_grid_thw) / merge_size**2
num_image_tokens = prod(image_grid_thw) / merge_size**2
```

The model-side visual merger uses `vision_config.spatial_merge_size`, defaulting to `2` in the inspected config source. Qwen visual tokens are dynamic-resolution tokens; one LLM visual token should be interpreted as a merged spatiotemporal grid cell, not as one raw frame or one original patch.

Milestone 1 temporal manifests include exactly one non-image video input per
question. Image-only, video-plus-image, and multi-video examples are excluded
from the temporal engineering, pilot, and confirmatory splits.

The pinned local Qwen processor validates video FPS as a scalar. Restricting
Milestone 1 to one real video input also avoids ambiguous per-video temporal
metadata and keeps compute comparable across categories.

Experiment 1 reconstructs the reduced grid as:

```text
T_llm = video_grid_thw[0]
H_llm = video_grid_thw[1] / spatial_merge_size
W_llm = video_grid_thw[2] / spatial_merge_size
```

and maps flattened visual-token order as temporal-major:

```text
for t in T_llm:
  for h in H_llm:
    for w in W_llm:
      visual token
```

This must be manually validated on a small GPU run by comparing produced
`video_grid_thw`, visual placeholder count, and token positions from the real
processor/model inputs.

## Temporal Bin Metadata

Qwen temporal bins may represent multiple sampled frames. Artifact metadata
therefore records the sampled frame indices and timestamps represented by each
visual-token temporal bin.

Artifacts store temporal-only relevance under `temporal_relevance`:

```text
raw_temporal_bin_scores[layer, temporal_bin]
normalized_temporal_bin_scores[layer, temporal_bin]
absolute_question_to_visual_attention_mass[layer]
layer_metrics[layer]
temporal_bins[temporal_bin]
```

Layer metrics include normalized temporal entropy, top-1 temporal-bin mass, the
number and fraction of temporal bins needed for 80% attention mass, temporal-bin
rank order, Spearman correlation with final-layer ordering, and top-K overlap
with final-layer important bins. Full attention tensors are not retained after
aggregation.

Artifacts also store `encoder_temporal` with pooled temporal representations and
adjacent-bin similarity for captured Qwen vision stages. Early/middle/late
vision block hooks are canonicalized back from Qwen's window order before
temporal pooling, and the final stage uses the canonical post-merger visual
output. Raw merger-hook output is treated as pre-reverse-index and is not
interpreted as chronological without canonicalization.

## Attention Extraction

The installed decoder attention class is `Qwen2_5_VLAttention`. Its forward path calls the Transformers attention interface selected by `config._attn_implementation`.

Correctness/debug extraction should use an attention implementation that returns weights. FlashAttention-style paths may not return full attention probabilities, so they should not be enabled until relevance extraction is validated.

The local implementation currently supports aggregation from returned attention tensors:

```text
attentions[layer][head, query_position, key_position]
  -> select question-token query rows
  -> select visual-token key columns
  -> average heads
  -> average question tokens
  -> keep layer dimension
```

As of 2026-08-09, two extraction modes are available:

```text
full
  Uses output_attentions=True.
  This is the correctness/debug baseline.
  It still materializes full [heads, seq, seq] tensors.

reduced_sdpa
  Registers a custom Transformers attention implementation.
  Registers the same custom implementation name with Transformers'
  causal-mask interface.
  Normal attention output is computed with torch scaled_dot_product_attention.
  Separately computes only question-token rows over all keys, selects visual
  columns, averages batch/heads/question tokens immediately, and stores one
  reduced vector per decoder layer.
```

The reduced implementation follows this path:

```text
Qwen2_5_VLAttention.forward
  -> q_proj/k_proj/v_proj
  -> qwen_relevance_reduced_sdpa_forward
  -> scaled_dot_product_attention for normal attn_output
  -> exact softmax over all keys for question-token rows only
  -> select visual-token columns
  -> immediately aggregate batch, heads, and question tokens
  -> return normal attn_output and no full attn_weights
```

This should lower peak attention memory relative to `full` because full
attention weights are no longer returned or stored. It still computes exact
question-row probabilities over all keys, so it is not a windowed or sampled
approximation. Before using it for medium/high resolution or long-frame runs,
validate it against `full` on a small example and compare temporal scores and
answer-choice logits.

Answer-choice scores in `answer_choice_scores` are computed from a separate
prefill forward pass. For baseline and decoder direct-access masking runs this
is the unmodified input. For pre-encoder masking runs this is already the masked
input, so compare it against the matching baseline artifact's
`answer_choice_scores`. Do not compare pre-encoder `answer_choice_scores`
against same-artifact `intervention_answer_choice_scores`.

Important correctness note: custom attention implementation names must also be
registered with `transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS`. If
they are not, `create_causal_mask` treats the backend as externally-managed and
returns `None`; using SDPA with `is_causal=False` would then leak future tokens
during prefill. The current implementation registers both custom attention
names with `eager_mask`, which materializes an additive causal mask and disables
causal-mask skipping.

## Temporal Interventions

`--decoder-mask-temporal-bin` performs decoder direct-access masking: it adds an
attention mask from non-visual query rows to all visual key columns belonging to
the requested Qwen temporal bin. Query rows are determined from `key_len` and
`q_len`; Qwen's three-axis multimodal RoPE `position_ids` are not treated as
absolute sequence indices. The same mask path is used for prefill and
generation.

`--pre-encoder-mask-temporal-bin` is a separate intervention. It masks the
sampled frames represented by the requested Qwen temporal bin before the Qwen
vision encoder runs. This tests whether the temporal evidence is needed before
vision encoding, while decoder direct-access masking tests whether later text
tokens can directly attend to those temporal-bin columns.

The recommended analysis is to remove final-layer high-ranked bins and compare
the intervention choice logits against removals of low-ranked and random bins.
For decoder direct-access masking, the normal `answer_choice_scores` field
remains unmodified and `intervention_answer_choice_scores` stores the masked
decoder scoring forward. For pre-encoder masking, `answer_choice_scores` is
already scored on masked frames and there is no separate same-artifact
intervention score; compare baseline and masked artifacts with
`scripts/compare_relevance_artifacts.py`.

## Dataset Balance Caveat

Temporal splits record bounded duration, duration bucket, participant and source
video IDs because duration/category confounds are expected in HD-EPIC. Category
comparisons should be treated as descriptive unless duration/subtype controls
are applied. Within-category temporal claims are more defensible.

## Scope Exclusions

Milestone 1 does not implement spatial selection, spatial heatmaps,
high-resolution crops, APT, or Experiment 2 behavior. Existing legacy utilities
may remain in the repository for compatibility, but the Experiment 1 output and
plots are temporal-only.
