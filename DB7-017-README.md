# DB7-017 BRB reliability fusion pilot

S1 and S15, seed42, 34 research neural fits. The workflow `db7-017-brb.yml` submits one private Kaggle run through the existing secondary-account credentials, monitors it and retrieves a verified result archive. One subject is assigned to each T4 GPU. Existing experiments are not rerun or modified.

## Question

Can observable uncertainty, disagreement and training-signal deviation identify useful W/S/I experts well enough to outperform SI and WSI? Does BRB improve on simpler fixed, confidence, logistic or Sugeno weighting? These are hypotheses, not measured BRB gains.

## Protocol

Exercise B E1 labels1–17, 200ms window/10ms stride, no trim. Twelve EMG channels and36 channels each of ACC/gyro/mag. Existing segment-wise EMG20–450Hz bandpass plus50Hz notch and inertial resampling. Training-only channel/spectral normalization. Preserve inertial offsets. The offline annotations and zero-phase filters do not establish real-time performance.

Training repetitions1/3/4/6; test2/5;13 fixed epochs; Adam, batch512, dropout0.65, no augmentation. W/S/I each train in four leave-one-training-repetition-out folds (12fits). Five final models W/S/I/SI/WSI train on all four training repetitions (5fits). Total17fits per subject,34 for the pilot. Smoke runs are small, separate implementation checks.

The W/S/I experts use the unchanged DB7-016 architecture and separate temporal/classifier heads. This is a decision ensemble, not three classifier heads extracted from a trained WSI model.

## Meta fitting

OOF repetitions1/3/4 fit temperatures, global importance and reliability models. Cross-fitted temperature predictions construct the reliability-training indicators. OOF repetition6 only calibrates scalar reliability. Every fitted object is frozen before outer-test predictions are evaluated.

OOF base models share underlying training recordings across these development groups. Repetition6 is therefore a development calibration holdout, not a fully nested validation estimate. Hyperparameters and comparisons are fixed in advance. Test2/5 were examined in previous studies; the two selected subjects and one seed support exploratory feasibility conclusions, not new confirmatory or population-level claims.

Each expert receives three bounded indicators: normalized entropy, mean pairwise total-variation disagreement, and robust training deviation. Physical descriptors are EMG log-RMS12, EMG mean-frequency12 and inertial channel-means108. Fold-specific references use training-only medians/IQRs with equally capped windows per gesture/repetition.

Each expert has eight low/high rules with fixed premise weights and learned binary correct/incorrect beliefs. Analytical RIMER combines rules inside each BRB. Calibrated reliability multiplied by frozen global importance determines final probability-average weights. No additional cross-expert ER is included.

## Outputs

The archive includes subject/method accuracy, balanced accuracy, macroF1, NLL/Brier/ECE; per-window predictions; gesture/repetition/phase error tables; confusion matrices; corrected and newly wrong cases versus SI/WSI; rule support and beliefs; calibrations; fitted models/references; complete training histories; data/code hashes and two-GPU telemetry.

`completion.json` must explicitly verify34 full neural fits and2 meta completions. A result folder or successful upload alone does not prove completion. Failure saves partial outputs with success=false.

SI/WSI controls use raw softmax; ensemble experts use temperatures fitted on development data. Their accuracy and paired recovery comparisons remain valid, but NLL/ECE changes versus SI/WSI include calibration differences. Comparisons between the different ensemble gates share the same calibrated expert probabilities.

## Launch and rerun

Run **DB7-017 BRB pilot on Kaggle** manually from GitHub Actions with the default empty monitor reference. To resume monitoring an already-submitted notebook, supply its `owner/slug` in the monitor field; that path never submits training. Each new full dispatch otherwise creates a uniquely named Kaggle experiment, so avoid duplicate dispatches.

The workflow automatically downloads results and keeps its artifact for30 days. No local Python environment needs to stay running. Use the actual Kaggle URL printed by the submission step and saved in launch provenance.
