# Experiment 1: Current Status and Findings

**Model:** Qwen2.5-VL-7B-Instruct  
**Dataset:** HD-EPIC VQA  
**Focus:** Temporal query relevance and redundancy  
**Status:** Eight-example engineering pilot complete; scaled evaluation pending

## 1. Work completed

We moved the experiment from Colab to a persistent A100 RunPod environment and established a reproducible HD-EPIC → frame sampling → Qwen inference → temporal relevance → causal intervention pipeline.

The current engineering pilot uses eight balanced examples: two each from fine-grained action, gaze, ingredient, and object-motion questions. For every example, we uniformly sample 32 frames. Qwen groups these into 16 temporal bins because each temporal visual token spans two sampled frames.

The implementation now supports:

- Tracing temporal representations through early, middle, and late vision-encoder stages.
- Extracting question-token attention received by visual tokens across decoder layers.
- Aggregating visual-token relevance into ordered temporal-bin scores.
- Plotting layer-by-bin relevance heatmaps and bin-rank trajectories.
- Masking selected temporal bins before the vision encoder.
- Comparing top-relevance, bottom-relevance, random, and position-matched interventions.
- Recording frame timing, temporal-position metadata, answers, choice probabilities, and relevance artifacts.
- Resuming interrupted runs and storing reproducible JSON/JSONL outputs.

The repository test suite currently passes **103 tests**.

## 2. Important correctness work

During validation, we discovered that Qwen was receiving incorrect temporal information for our externally sampled video frames. The effective sampling rate was approximately 3.11 FPS, but Qwen initially treated the input as though it were 24 FPS.

We then found a Transformers 5.14.1 issue in Qwen's temporal rotary-position calculation. A sub-second interval was converted to an integer before multiplication, causing the temporal-position interval to become zero. This meant distinct temporal steps were not represented correctly inside the decoder.

We implemented a guarded repository-level fix that:

- Passes the actual effective frame rate.
- Computes the correct seconds per temporal grid step.
- Corrects the temporal mRoPE interval calculation.
- Records whether the patch is active in every artifact.
- Adds regression tests.

For the current configuration, the corrected temporal interval is **1**, rather than 0.

This correction materially changed both predictions and temporal relevance. Across the eight examples:

- Accuracy changed from **25% to 50%**.
- Two previously incorrect answers became correct.
- Relevance values changed by as much as **0.322**.

The correction did not improve every example's confidence, so this is primarily a correctness finding—not evidence that the patch universally improves VQA performance.

## 3. Runs completed

### Corrected baseline

- Eight examples.
- 32 frames / 16 temporal bins per example.
- Qwen2.5-VL-7B.
- Correct temporal mRoPE handling.
- Accuracy: **4/8 = 50%**.

### Encoder temporal analysis

We measured adjacent, nonadjacent, and far-bin representation similarity through the vision encoder.

At the final visual representation:

- Adjacent-bin cosine similarity: **0.933**.
- Nonadjacent similarity: **0.906**.
- Far-bin similarity: **0.899**.
- Centered local advantage: **0.312**.
- Effective rank: **9.28 of 16**.
- PC1 variance fraction: **0.312**.

### Pre-encoder causal masking

For each example, we masked four of 16 temporal bins before the vision encoder:

| Intervention | Mean correct-answer logP change | Median change | Accuracy after masking |
|---|---:|---:|---:|
| Top 25% relevance | **−0.157** | +0.045 | **25%** |
| Bottom 25% relevance | −0.025 | −0.064 | **25%** |
| Random 25% | +0.097 | +0.139 | **37.5%** |

Relative to random masking, top masking was more damaging by:

- **0.254 mean correct-answer log-probability**.
- **0.151 median correct-answer log-probability**.
- Top masking was more harmful for **5/8 examples**.

### Position-matched control

Top relevance included bin 0 in 7/8 examples, compared with 2/8 for random and 0/8 for bottom. This exposed an early-position confound.

To control for it, we swapped top-bin masks between the two examples within each question category. This preserved the overall temporal-position distribution while breaking the association between each example and its own relevance map.

Results:

- True top-mask accuracy: **25%**.
- Position-matched-mask accuracy: **50%**.
- Matched-minus-top correct-answer logP: **+0.118 mean**, **+0.097 median**.
- True top masking was more harmful for **5/8 examples**.

## 4. Insights from Experiment 1

### Temporal information is genuinely important to Qwen

Correcting temporal positions changed answers and relevance maps substantially. Temporal preprocessing and positional encoding must therefore be validated before interpreting attention plots.

### The encoder contains temporal redundancy

Adjacent bins are more similar than distant bins, even after accounting for a large shared visual component. This supports compressing or merging some temporally adjacent information.

However, the final temporal representation has an effective rank of approximately nine across 16 bins. The entire video cannot be collapsed into only one or two temporal tokens without likely losing useful information.

### Decoder temporal relevance is not extremely sparse

The earlier layer summary required approximately 11 of 16 bins to capture 80% of temporal attention mass. Relevance has identifiable peaks but remains relatively distributed. This suggests selective reduction rather than retaining only one or two keyframes.

### The model has a strong early-bin bias

Bin 0 appeared in 7/8 top-four relevance sets. Some apparent benefit of attention-ranked selection is therefore caused by temporal position or early context, not necessarily query-specific evidence.

### Query-ranked bins still contain example-specific signal

The position-matched control preserved the aggregate position bias but did not reproduce the damage caused by masking each example's own top bins. The positive mean and median matched-versus-top differences provide preliminary causal evidence that the relevance maps capture example-specific task information beyond position alone.

### The evidence remains preliminary

The pilot contains only eight examples. It identifies a promising effect and validates the methodology, but it cannot establish statistical significance or category-level generalization.

## 5. What remains

### Required to complete the temporal Experiment 1 study

1. Run the corrected baseline on the balanced **48-example** set.
2. Generate corrected relevance rankings for those 48 examples.
3. Run top-25% masking and within-category position-matched masking at scale.
4. Run several random or matched permutations instead of a single assignment.
5. Report paired confidence changes, accuracy, bootstrap confidence intervals, and paired statistical tests.
6. Break results down by gaze, ingredient, fine-grained action, and object motion.
7. Repeat a smaller subset with 64 and 128 frames to verify that the 32-frame findings persist.
8. Run the planned layer-wise decoder intervention: after selected decoder layers, block direct query access to original visual tokens and measure whether performance remains stable.
9. Add a query-specific control, such as asking mismatched questions about the same video, to separate query relevance from generic visual saliency.
10. Measure visual-token count, FLOPs, latency, memory, and throughput before making an efficiency claim.

## 6. Current conclusion

The engineering pilot is successful. We have a temporally correct and tested Qwen pipeline, evidence of adjacent-frame redundancy, and preliminary causal evidence that decoder-derived query relevance identifies useful temporal regions beyond aggregate position bias.

The next step is no longer additional debugging. It is to freeze the implementation and scale the corrected baseline, true top masking, and position-matched controls to the balanced 48-example set.
