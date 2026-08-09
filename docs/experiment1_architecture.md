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
video pixels
  -> Qwen2_5_VLProcessor / video processor
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

## Query Relevance Definition

Qwen2.5-VL does not expose a separate text/vision cross-attention block in this implementation. Visual features are projected/merged and inserted into the decoder sequence at visual placeholder token positions.

For Experiment 1, query-to-visual relevance is therefore:

```text
decoder self-attention rows for natural-language question tokens
  x
decoder self-attention columns for visual token positions
```

The vision encoder attention is not question-conditioned and should not be labeled query relevance.

## Visual Token Counts and Layout

The processor expands visual placeholders according to:

```text
num_video_tokens = prod(video_grid_thw) / merge_size**2
```

The model-side visual merger uses `vision_config.spatial_merge_size`, defaulting to `2` in the inspected config source. Qwen visual tokens are dynamic-resolution tokens; one LLM visual token should be interpreted as a merged spatiotemporal grid cell, not as one raw frame or one original patch.

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

This must be manually validated on a small GPU run by comparing produced `video_grid_thw`, visual placeholder count, and token positions from the real processor/model inputs.

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
validate it against `full` on a 4-8 frame example and compare per-layer token or
frame scores within numerical tolerance.

## Vision-Access Intervention Blocker

The desired intervention is layer-specific masking:

```text
layers <= L: normal attention
layers > L: text/question query rows cannot attend to visual key columns
```

In the inspected implementation, `Qwen2_5_VLTextModel.forward` constructs causal masks once, then passes one mask per layer type into each decoder layer. There is no public API to provide a different text-to-visual mask per decoder layer while preserving all other causal behavior.

The local code implements and tests construction of the desired layer-specific masks, but does not patch Qwen internals yet. The smallest faithful implementation on GPU is likely one of:

1. subclass/patch `Qwen2_5_VLTextModel.forward` to add a per-layer additive mask before each `decoder_layer` call, or
2. wrap each `Qwen2_5_VLDecoderLayer` / `Qwen2_5_VLAttention` to add the layer-specific text-to-visual block to `attention_mask`.

No hidden-state zeroing substitute should be used; it is a different intervention.
