# Experiment 1 v3: Preliminary Temporal Baseline

## Objective

This experiment tests whether Qwen2.5-VL distributes attention unevenly across temporal regions and how temporal allocation changes through the vision encoder and language decoder.

Encoder attention is query-agnostic. Decoder attention from question-token rows to visual-token columns is query-conditioned. Attention alone is not treated as evidence of causal importance.

## Configuration

- Model: Qwen2.5-VL-7B
- Videos/questions: 77
- Resolution: medium
- Sampling: full-video adaptive temporal sampling
- Temporal bins: 8–64
- Frames per bin: 2
- Frame budget: 16–128 frames
- Vision layers: 32
- Decoder layers: 28
- Attention extraction: reduced SDPA
- Query scope: question tokens
- Independent experimental unit: source video

All 77 examples completed and passed structural validation. Decoder distributions had the expected layer-by-bin shape, encoder distributions had the expected layer-by-head-by-bin shape, values were finite, and normalized distributions summed to one.

## Baseline VQA Results

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Fine-grained | 24 | 39 | 61.54% |
| Gaze | 5 | 13 | 38.46% |
| Ingredient | 6 | 13 | 46.15% |
| Object motion | 2 | 12 | 16.67% |
| Overall | 37 | 77 | 48.05% |

## Preliminary Temporal Findings

Decoder temporal attention is broad overall but changes non-monotonically with depth.

- Peak mean top-bin mass occurs at decoder layer 27: 0.1908.
- Lowest mean normalized temporal entropy occurs at decoder layer 11: 0.9170.
- Mean absolute visual-attention mass at layer 27 is 0.3394.

These measurements indicate layer-wise changes in temporal allocation. They do not yet establish query relevance, causal importance, or safe temporal pruning.

## Aggregate Figures

### Encoder temporal attention

![Encoder temporal heatmap](assets/experiment1_v3/encoder_temporal_heatmap.png)

### Decoder temporal attention

![Decoder temporal heatmap](assets/experiment1_v3/decoder_temporal_heatmap.png)

### Decoder entropy by category

![Decoder entropy](assets/experiment1_v3/decoder_entropy_by_category.png)

### Encoder temporal similarity

![Encoder pairwise cosine similarity](assets/experiment1_v3/encoder_pairwise_cosine.png)

### Encoder local temporal advantage

![Encoder local temporal advantage](assets/experiment1_v3/encoder_local_temporal_advantage.png)

## Duration Confound

The global duration groups contain 26 short, 25 medium, and 26 long videos. However, duration is strongly confounded with question type:

- Short videos are almost entirely fine-grained action-recognition questions.
- Gaze questions occur only in the medium group.
- Long videos contain action localization, ingredient localization, and object-motion itinerary questions.

Consequently, pooled accuracy differences between duration groups cannot be interpreted as causal duration effects. Duration-conditioned results must be reported within question type or through models adjusting for question type and log duration.

The paired controls and interventions compare each example against itself and therefore hold video duration and question type fixed.

## Pending Experiments

1. Repeated-frame positional-bias control.
2. Reversed-video content-versus-position control.
3. Mismatched-query specificity control.
4. Top, bottom, and position-matched random bin masking.
5. Top versus random keep-only interventions.
6. Decoder fusion-depth blocking.
7. Fixed-budget 128-frame robustness condition.
8. Participant/video-clustered statistical analysis.

## Current Interpretation

The baseline provides descriptive evidence that temporal allocation changes across model depth. Claims about query relevance and temporal pruning remain provisional until the paired controls and causal interventions are complete.