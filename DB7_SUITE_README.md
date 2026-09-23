# Seven controlled BRB experiments

## DB7-022: matched-checkpoints
Does BRB help when calibration and test use exactly the same neural model?

Reuse the 66 seed-42 held-repetition-6 expert checkpoints. Infer calibration 6 and test 2/5 from those same checkpoints and the exact saved scaler. Compare learned global probability fusion, confidence weighting and the eight-rule correctness BRB structure. The BRB structure is refitted as a shared gate; this is not a numerical replay of the previous subject-specific BRB.

## DB7-023: directional-inputs
Do candidate-specific margins and class distances outperform generic reliability indicators?

Compare entropy/disagreement/global-deviation against baseline margin/candidate margin/prototype contrast. Keep three antecedents, two references each, correctness consequents and ER unchanged. Prototypes use neural training repetitions 1/3/4 only.

## DB7-024: recovery-consequents
Does learning recovery versus harm make better switching decisions?

Hold the selected input representation fixed. Compare per-expert incorrect/correct beliefs used for probability weighting against recovery/harm/neither beliefs for a baseline-versus-challenger decision. The challenger is selected from expert probabilities without the true label. Switch only when predicted recovery exceeds predicted harm. A hard switch has no defined 17-class probability distribution: gesture NLL/Brier/ECE are not reported for it.

## DB7-025: reference-capacity
Does finer reference resolution improve development accuracy?

Compare 2x2x2=8, 3x2x2=12, 3x3x2=18 and 3x3x3=27 rules. Three-reference axes use 0, training-fold median clipped to [0.1,0.9], and 1. Keep the chosen inputs/consequents unchanged. Equal accuracy favours fewer rules.

## DB7-026: causal-history
Can past predictions identify when intervention is useful?

Compare no history against one added antecedent at a time: past prediction stability, margin change or EMG activation change. Use at most ten past 10-ms outputs and current output; reset using timestamps only. Never use true labels or phase at inference. Each added attribute has two references. This is causal at the gate level only: inherited zero-phase filtering and rest removal remain annotation-assisted offline preprocessing.

## DB7-027: mechanism-controls
Does learned ER weighting beat simpler matched controls?

Keep selected antecedents, reference counts and target fixed. Compare ER with fixed weights, ER with regularized learned positive rule/attribute weights, zero-order Sugeno aggregation and multinomial logistic regression. Use identical folds and labels. The learned attribute exponent is bounded by exp(-1)..exp(1); rule weights by exp(-2)..exp(2). Consequent logits and weights have L2 penalty 0.01. These bounded designs are experiments, not established solutions.

## DB7-028: seed-robustness
Does the locked design behave consistently across neural seeds?

Reuse seed 42 and train seeds 43/44 on repetitions 1/3/4, 13 fixed epochs, W/S/I separately, 132 new fits in total. Use both T4 GPUs with independent subject workers. Lock stage-6 structure and the baseline/no-intervention decision; refit numerical gate parameters using development repetition 6 for each seed. Report all seeds without choosing the best. Existing DB7 has already informed research decisions, so this is robustness, not independent confirmation.

## Fixed protocol and assumptions

- Exercise B, E1 labels 1–17, all 22 subjects; rest excluded using restimulus/rerepetition.
- Within-subject neural experts: train repetitions 1/3/4, development repetition 6, test repetitions 2/5. This deliberately differs from the earlier four-training-repetition final model.
- 200 ms (400 samples at 2 kHz), 10 ms (20 samples) stride; windows stay within an active repetition. No new boundary trim is added.
- EMG 12 channels, fourth-order 20–450 Hz Butterworth plus 50 Hz notch (Q=30), inherited zero-phase filtering per repetition. ACC, gyroscope and magnetometer each retain 36 channels and original offsets; align/resample per repetition. Input z-scaling and spectral scaling use neural training only.
- The unchanged W/S/I experts use the existing temporal CNN architecture. Each branch is an independent 17-class expert, not a newly trained joint fusion network. Fixed 13 epochs, dropout .65, batch 512, Adam, learning rates 1e-3/1e-4/1e-5 in epochs 1–3/4–9/10–13; augmentation off.
- A compact shared gate is fitted across subjects. Five outer subject-disjoint development folds evaluate each candidate; three inner subject folds produce baseline outputs for gate training. Neural experts remain within-subject. This is not LOSO neural evaluation.
- Fit meta models on up to 32 evenly spaced windows per subject/gesture/trial, with equal trial weights; evaluate all windows. These are correlated samples, not 32 independent trials.
- Select mean development subject accuracy, then fewer rules, then prespecified order. Preserve the global baseline whenever no candidate beats it. No test-based tuning. Later notebooks read only the previous selected configuration.
- Test inference is cached before gate selection for efficiency, but the meta fitting/selection code does not load test labels until writing the selection lock. This is reused DB7 data, not an untouched confirmatory dataset.

## BRB inputs, rules and outputs

All antecedents lie in [0,1]. With three two-reference inputs, the Cartesian product yields eight rules. Matching is piecewise linear; rule activation is the normalized product of matching degrees, optionally raised to learned positive attribute weights and multiplied by rule weights. Consequent belief degrees are softmax-normalized and sum to one. ER combines complete beliefs; Sugeno uses the same rule activations and weighted consequent average. Logistic is a matched non-rule control.

Correctness heads output P(incorrect), P(correct) for each expert. Fusion multiplies the development-fitted global expert weights by predicted correctness and renormalizes. Recovery heads output P(recovery), P(harm), P(neither); they compare an observable challenger against the global baseline. If recovery > harm, choose the challenger, otherwise retain the baseline. Initial beliefs are smoothed training outcome frequencies. SciPy L-BFGS-B fits consequent logits with analytic gradients. Test labels never fit rules.

## Required outputs and interpretation

Export every development candidate and optimization diagnostic; selection lock; exact checkpoint provenance; initial and trained rules; reference values, rule/attribute weights; fitted models; per-window predictions; per-subject/gesture/repetition/phase errors; recovery/harm counts; confusion matrices; accuracy, macro F1 and balanced accuracy; genuine probability NLL/Brier/ECE where defined. Export paired subject bootstrap intervals, subject sign-flip and Wilcoxon tests. Window-level exact McNemar is descriptive only because windows overlap 95%. Multiple arms and sequential research decisions make inferential results exploratory. No method is promised a 90% chance of improvement.

The selected-policy and structural-candidate results are both retained. A structural candidate can lose while the selected policy correctly remains the baseline. Compare with this study's matched-checkpoint baseline, not directly with the older four-repetition accuracy.
