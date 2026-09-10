# Experiment 1 v3: Preliminary Temporal Baseline

## Research question

This experiment asks whether Qwen2.5-VL processes different temporal regions of a video differently across its vision encoder and language decoder, and whether the resulting temporal allocation could eventually support adaptive temporal computation.

The two stages measure different phenomena:

- Vision-encoder attention and representations are query-agnostic because the encoder has not seen the question.
- Decoder attention from question-token rows to visual-token columns is query-conditioned.

Attention is treated as a descriptive signal, not as proof of importance. Query specificity, positional bias, and causal utility require additional controls and interventions.

## Baseline configuration

- Model: Qwen2.5-VL-7B
- Independent experimental units: 77 source videos
- Primary questions: one per source video
- Resolution: medium
- Sampling: adaptive full-video temporal sampling
- Temporal bins: 8â€“64 per video
- Frames per bin: 2 chronological frames
- Frame budget: 16â€“128 frames
- Vision blocks: 32
- Decoder layers: 28
- Decoder query scope: question tokens only
- Attention extraction: reduced SDPA
- Aggregation: equal weight per source video

All 77 examples completed successfully. Structural validation confirmed:

- 77 artifacts from 77 unique source videos;
- 8â€“64 temporal bins per example;
- 32 encoder-attention layers and 28 decoder-attention layers;
- finite values and correctly normalized temporal distributions;
- valid monotonic timestamps and full-video temporal coverage.

## Baseline VQA performance

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Fine-grained | 24 | 39 | 61.54% |
| Gaze | 5 | 13 | 38.46% |
| Ingredient | 6 | 13 | 46.15% |
| Object motion | 2 | 12 | 16.67% |
| **Overall** | **37** | **77** | **48.05%** |

The accuracy table establishes that the sampled inputs retain meaningful task signal. Category differences are descriptive because category, question type, and duration are not independently balanced.

## Duration-normalized attention aggregation

Videos contain different numbers of temporal bins, so raw bin probabilities cannot be averaged directly. For each example with \(B\) bins and normalized temporal mass \(p_b\), the corrected analysis computes

\[
\operatorname{lift}_b = Bp_b.
\]

Uniform temporal allocation therefore has lift 1 for every example, regardless of whether it contains 8 or 64 bins. Each lift distribution is interpolated at normalized relative-time bin centers and only then averaged across videos. Heatmaps display \(Bp_b-1\), for which zero denotes uniform attention.

## Encoder attention is uniform in aggregate

![Encoder temporal attention relative to uniform](assets/experiment1_v3_corrected/encoder_attention_lift_heatmap.png)

Across all 77 videos, 32 encoder blocks, heads, and normalized temporal positions, the maximum absolute deviation of mean incoming temporal-attention lift from uniform is

\[
\max |Bp_b-1| = 9.97\times 10^{-7}.
\]

This is numerical noise. The earlier autoscaled encoder heatmap visually exaggerated these microscopic differences.

The result establishes that **mean incoming encoder-attention mass does not provide a useful aggregate temporal ranking signal**. It does not establish that encoder representations lack temporal information, nor does it rule out per-example or per-head selectivity that cancels under aggregation.

## Decoder attention becomes strongly nonuniform

![Decoder temporal attention relative to uniform](assets/experiment1_v3_corrected/decoder_attention_lift_heatmap.png)

The decoder exhibits substantial depth-dependent temporal allocation. Its maximum mean deviation from uniform is 1.3266 lift units, corresponding to a maximum mean lift of approximately

\[
1 + 1.3266 = 2.3266.
\]

Thus, the most emphasized normalized temporal position receives approximately **2.33 times its uniform expected attention mass** after averaging across videos.

The heatmap shows two broad regimes:

- early decoder layers preferentially allocate attention to the beginning of the video;
- middle and later layers increasingly emphasize the end of the video;
- layers 11â€“15 form a conspicuous redistribution region;
- late layers again exhibit strong boundary-focused allocation.

This is evidence of nonuniform decoder temporal allocation. The boundary-shaped pattern may reflect positional bias rather than content relevance, so it cannot yet justify temporal pruning.

## Temporal concentration changes non-monotonically with depth

![Decoder temporal entropy by question type](assets/experiment1_v3_corrected/decoder_entropy_by_question_type.png)

Normalized entropy remains high overall, meaning decoder attention is generally broad rather than concentrated in a single bin. Nevertheless, its concentration changes substantially and non-monotonically across layers.

- Peak mean top-bin mass occurs at decoder layer 27: 0.1908.
- Lowest overall mean normalized entropy occurs at decoder layer 11: 0.9170.
- Localization and object-itinerary questions show stronger middle- and late-layer concentration than gaze and action-recognition questions.
- Question-type curves remain descriptive because question type is entangled with absolute video duration and temporal-bin count.

### Layer-14 redistribution

Mean and median statistics show that the layer-14 behavior is systematic rather than an artifact of a few outliers.

| Question type | Layer 13 entropy | Layer 14 entropy | Layer 15 entropy |
|---|---:|---:|---:|
| Action localization | 0.858 | 0.947 | 0.891 |
| Ingredient localization | 0.861 | 0.950 | 0.905 |
| Object itinerary | 0.851 | 0.952 | 0.871 |
| Action recognition | 0.981 | 0.948 | 0.998 |
| Gaze anticipation | 0.982 | 0.981 | 0.979 |

Localization and itinerary tasks become markedly more diffuse at layer 14 before concentrating again. Action recognition instead becomes slightly more concentrated at layer 14 and almost uniform at layer 15. This indicates a systematic layer-dependent redistribution rather than monotonic temporal focusing.

## Absolute visual access is distinct from temporal allocation

![Decoder absolute visual-attention mass](assets/experiment1_v3_corrected/decoder_absolute_visual_mass.png)

Absolute question-to-visual attention mass varies strongly with both layer and question type.

- Action and ingredient localization exhibit the strongest early visual access.
- Action recognition and gaze anticipation allocate substantially less absolute mass to vision.
- Visual access falls sharply in early layers, partially recovers through the middle, and rises again in the final layers.
- Layers can change their conditional temporal distribution even while total visual access decreases.

Consequently, absolute visual-attention mass and the temporal distribution conditioned on visual access must remain separate measurements.

## Encoder representations retain local temporal structure

![Encoder local temporal advantage](assets/experiment1_v3_corrected/encoder_local_temporal_advantage_ci.png)

Although aggregate encoder-attention mass is uniform, encoder representations are locally structured. Adjacent-bin representations are more similar than far-bin representations at every captured stage.

| Encoder stage | Raw adjacent-minus-far | Mean-centered adjacent-minus-far |
|---|---:|---:|
| Early block | 0.0021 [0.0012, 0.0031] | 0.327 [0.248, 0.429] |
| Middle block | 0.0148 [0.0114, 0.0188] | 0.478 [0.420, 0.552] |
| Late block | 0.000014 [0.000011, 0.000018] | 0.369 [0.235, 0.512] |
| Merger | 0.0263 [0.0202, 0.0322] | 0.443 [0.399, 0.497] |
| Final | 0.0263 [0.0202, 0.0324] | 0.443 [0.397, 0.491] |

Values in brackets are hierarchical participant-then-video bootstrap 95% confidence intervals.

Raw representations share an overwhelmingly strong common direction, making all temporal bins appear extremely similar. Removing the per-video mean direction exposes a large and consistently positive local temporal advantage. This supports the presence of high temporal redundancy together with residual local temporal organization.

The merger and final values are identical because canonical token reordering does not change pairwise cosine relationships; they are not independent effects. Representation similarity was captured at five encoder stages, whereas attention mass was captured at all 32 blocks.

## Duration and question-type confounding

The global duration groups contain 26 short, 25 medium, and 26 long videos, but their task composition is highly imbalanced:

- 25 of 26 short examples are fine-grained action recognition;
- all gaze-anticipation examples occur in the medium group;
- long examples consist of action localization, ingredient localization, and object-motion itinerary tasks.

Accordingly, pooled short/medium/long accuracy or attention differences cannot be interpreted as causal duration effects. Duration must be treated as a control variable through within-question-type analysis, continuous log-duration adjustment, andâ€”most importantlyâ€”paired within-example interventions.

## Current interpretation

The corrected baseline supports three descriptive conclusions:

1. Mean incoming temporal-attention mass in the query-agnostic vision encoder is effectively uniform.
2. Encoder representations nevertheless contain substantial local temporal structure after removing their dominant shared component.
3. Strong, non-monotonic temporal allocation emerges in the query-conditioned decoder, reaching approximately 2.33 times uniform expectation at its most emphasized averaged position.

The baseline does **not** yet show that decoder-selected bins are query-relevant, content-driven, causally important, or safely prunable.

## Required follow-up experiments

1. **Repeated-frame control:** preserve temporal positions while removing content differences to measure pure positional bias.
2. **Reversed-video control:** determine whether allocation follows content or absolute early/late positions.
3. **Mismatched-query control:** measure whether temporal rankings change with the question.
4. **Necessity interventions:** mask top, bottom, and position-matched random temporal bins.
5. **Sufficiency interventions:** retain only query-ranked top bins versus matched random or uniform selections.
6. **Fusion-depth blocking:** determine when direct access to selected visual evidence stops affecting predictions.
7. **Fixed-budget robustness:** repeat the principal findings using 128 uniform frames and 16 relative-time bins.
8. **Clustered statistical analysis:** treat source video as the independent unit and cluster resampling by participant and video.

Only these paired controls and interventions can establish whether the observed decoder allocation is useful for temporal optimization.

## Cross-Architecture Replication: VILA-Llama3

Experiment 1 now includes an additive cross-model replication path for
`Efficient-Large-Model/Llama-3-VILA1.5-8B`. This path is intentionally separate
from the completed Qwen v3 baseline and does not change the Qwen protocol,
prompts, sampling policy, token mappings, attention definitions, or artifact
schema.

The matched replication condition uses the same 77 primary examples from
`outputs/experiment1_v3/primary_manifest.jsonl` and the same adaptive
full-coverage temporal bins. For cross-model comparability, it samples exactly
one deterministic frame per temporal bin by running the realtime policy with
`--frames-per-bin 1`. This yields 8-64 decoded RGB frames per example. Qwen and
VILA can therefore be run on identical frame indices and timestamps under the
matched condition.

The VILA path supports three descriptive conditions:

- `baseline`
- `repeated_frame`
- `reversed_video`

For VILA, the runner uses the official NVLabs/VILA implementation pinned at
commit `0f1426e8da9181e6e6653e10bc15f62d515fa2f6`. The isolated setup script
clones that repository, installs its dependencies, and loads
`Efficient-Large-Model/Llama-3-VILA1.5-8B` through VILA's official model
builder, tokenizer, image processor, and conversation template.

The runner bypasses VILA's MP4 sampler. It passes Experiment 1's already-decoded
ordered RGB frames directly as image inputs. It constructs the multimodal prompt
with one VILA image placeholder per sampled frame, measures the actual
encoded/projected image-feature length for every frame, and maps the inserted
visual token columns back to input frames and temporal bins. It does not infer
equal visual-token counts across frames. The run fails if prepared frame order,
token mapping, or truncation checks fail.

Decoder prefill extraction wraps VILA/Llama SDPA during the measurement forward.
It preserves the original SDPA output path, uses the same attention mask and
causal semantics, and computes only question-token rows by visual-token columns
before reducing to temporal bins. It does not request full sequence-by-sequence
attention tensors from the real backend.

The VILA decoder analysis records the same primary temporal measurements used
for Qwen decoder prefill analysis:

- raw temporal-bin mass;
- normalized temporal-bin distribution;
- absolute question-to-visual attention mass;
- normalized entropy;
- top-bin mass;
- first-bin and last-bin mass;
- bins required for 80% mass.

Cross-model analysis is performed with
`scripts/analyze_cross_model_temporal.py`. It aligns layers by normalized depth
\(l/(L-1)\), compares temporal lift relative to uniform, entropy, first/last-bin
mass, absolute visual mass, and per-example Spearman correlation, and reports
paired bootstrap confidence intervals clustered by source video. Baseline,
repeated-frame, and reversed-video conditions are analyzed separately.

The current repository implementation includes CPU regression tests for VILA-style
image placeholder insertion, variable image-feature lengths, frame order,
question-row isolation, and temporal reduction. Real-checkpoint validation is
still required before interpreting cross-model results: the backend should not
be treated as empirically runnable until `scripts/smoke_test_vila_temporal.py`
successfully reaches the checkpoint, preprocesses one 8-frame and one 64-frame
example, and validates token mapping, output equivalence, memory and runtime.

## Preliminary repeated-frame positional control

To distinguish content-dependent temporal allocation from architectural position bias, we constructed a repeated-frame control that preserves the original number of frames, temporal bins, visual tokens, and temporal positions while replacing every sampled frame with the same frame. Because visual content is identical across time, any nonuniform temporal attention observed under this condition cannot be attributed to changing video content.

The following results are preliminary and include 12 of the 77 videos. The subset contains five action-recognition, three action-localization, three gaze-anticipation, and one ingredient-localization example; it contains no object-motion examples. Consequently, these results establish an engineering and qualitative signal but are not used for final effect-size or significance claims.

### Decoder attention reveals a strong positional prior

![Repeated-frame decoder temporal attention](../outputs/experiment1_v3/preliminary/repeated_frame_partial/decoder_attention_lift_heatmap.png)

The decoder remains strongly nonuniform even though every temporal position contains identical visual content. The maximum absolute deviation from uniform is \(2.739\) in lift-minus-one units, corresponding to a peak temporal-bin allocation of approximately \(3.74\times\) the uniform expectation.

The heatmap reveals that this concentration is highly structured: the decoder consistently overweights the beginning of the visual sequence, with particularly strong first-bin preference around layers 9, 14, and 21–27. Intermediate temporal positions are correspondingly underweighted. This demonstrates that temporal attention concentration alone is insufficient evidence of content relevance; Qwen possesses a substantial intrinsic temporal-position prior.

Qualitatively, this pattern differs from the real-video baseline. The baseline initially favors early positions but transitions toward a strong end-of-video preference after approximately layers 11–15. In contrast, the repeated-frame control remains dominated by the beginning of the sequence. This suggests that the baseline’s late-video preference is not explained by a static positional prior alone: changing visual content, temporal ordering, or their interaction with the query modifies the decoder’s temporal allocation. This comparison remains descriptive until the repeated-frame run is completed and evaluated through paired per-video statistics.

### Aggregate encoder attention remains uniform

![Repeated-frame encoder temporal attention](../outputs/experiment1_v3/preliminary/repeated_frame_partial/encoder_attention_lift_heatmap.png)

Aggregate incoming encoder attention is effectively uniform across temporal positions. Its maximum absolute deviation from uniform is only \(2.68\times10^{-8}\), consistent with floating-point noise. This agrees with the real-video baseline and indicates that aggregate encoder attention mass does not expose meaningful temporal selectivity in this model.

This result does not imply that the encoder discards temporal information. Attention mass and residual representation structure measure different properties: the former records where tokens attend, whereas the latter tests whether representations of neighboring moments remain more similar than representations of distant moments.

### Encoder representation structure disappears when content is repeated

![Repeated-frame encoder local temporal advantage](../outputs/experiment1_v3/preliminary/repeated_frame_partial/encoder_local_temporal_advantage_ci.png)

Under the repeated-frame control, the adjacent-minus-far representation-similarity advantage collapses to approximately \(10^{-17}\) at every captured encoder stage, for both raw and mean-centered representations. All confidence intervals lie at or overlap numerical zero.

This contrasts with the real-video baseline, where adjacent temporal bins exhibit a positive similarity advantage at the captured encoder stages. The disappearance of this effect when visual content is held constant provides evidence that the baseline encoder structure is driven by local changes and continuity in video content, rather than merely by temporal-position encoding. Thus, while aggregate encoder attention is uniform, encoder representations still appear to preserve content-dependent local temporal organization.

### Absolute visual access remains substantial despite redundancy

![Repeated-frame decoder absolute visual mass](../outputs/experiment1_v3/preliminary/repeated_frame_partial/decoder_absolute_visual_mass.png)

The decoder continues to allocate substantial absolute attention mass to visual tokens even though those tokens contain heavily duplicated information. Visual access is highest in the earliest layer, decreases sharply, and then varies non-monotonically across subsequent layers.

This separates two notions that must not be conflated:

* **Absolute visual mass** measures how much the question tokens access visual tokens.
* **Conditional temporal allocation** measures how that visual attention is distributed across time.

The repeated-frame result shows that high visual access does not necessarily imply access to additional or useful evidence. Likewise, a highly ranked temporal bin may reflect positional preference rather than unique information.

### Temporal concentration remains depth-dependent

![Repeated-frame decoder temporal entropy](../outputs/experiment1_v3/preliminary/repeated_frame_partial/decoder_entropy_by_question_type.png)

Normalized temporal entropy remains relatively high overall, but it decreases at specific decoder depths even with identical frames. The strongest concentration appears around layers 10, 14, 22, and 27. These layer-specific changes indicate that positional concentration is not a constant offset that can be removed globally; it evolves through the decoder.

### Preliminary conclusion

The repeated-frame control supports three preliminary conclusions:

1. Aggregate encoder attention is temporally uniform and therefore is not, by itself, a useful temporal-pruning signal.
2. Local temporal structure in encoder representations disappears when content variation is removed, suggesting that the corresponding baseline structure is genuinely content-dependent.
3. Decoder temporal attention contains a strong, depth-dependent positional component, particularly a first-bin bias, even when every temporal position presents identical visual information.

These results motivate bias-corrected, layer-specific temporal selection rather than direct pruning based on raw decoder attention. The completed 77-video paired analysis will quantify how much of the real-video decoder distribution is explained by this positional prior. Reversed-video and mismatched-query controls are additionally required to separate content following, temporal-order sensitivity, and query-specific relevance.
