# DB7-030: Hudgins TD4 complementarity

This notebook asks whether an interpretable EMG time-domain classifier supplies useful evidence where the frozen DB7-016 SI and I networks fail. It **does not** train a neural network or BRB.

## Implemented protocol

- E1 gestures 1–17, S1–S20, seeds 42/43/44 for the **saved** SI/I outputs.
- Neural training repetitions 1/3/4/6; reserved test repetitions 2/5. Existing DB7-016 outputs have already been inspected, so this is exploratory.
- 200 ms windows. TD4 classifiers use 50 ms training stride to reduce repeated evidence; evaluation matches the saved 10 ms prediction grid exactly.
- 20–450 Hz fourth-order zero-phase Butterworth bandpass, then 50 Hz Q30 zero-phase notch, separately within each gesture repetition. Rest is excluded.
- MAV, WL, thresholded ZC and thresholded SSC on each of 12 EMG channels (48 features).
- ZC/SSC threshold candidates: 0.5%, 1%, 2% of each subject/channel's 95th percentile absolute EMG. Each CV fold estimates this scale using only its three fitting repetitions; the final fit uses all four training repetitions. Four leave-one-training-repetition-out folds select the threshold ratio on full TD4.
- All 15 nonempty feature-family subsets use that selected threshold and are scored by the same four training folds. Per-subject selection is fixed before evaluation. Each subset then fits a regularized LDA with training-only scaling, pooled covariance 10% shrinkage toward its diagonal, and a small diagonal floor.
- The compact SI/I trace is a hash-recorded subset of the previously verified DB7-016 window audit. The GitHub Action publishes it as a **private Kaggle dataset** for the notebook.

## Outputs

The notebook exports each subject's aligned window predictions; subset CV selection and test accuracy; recovery and harm for all 15 combinations; class, subject, seed and phase breakdowns; diagnostic SI/I/TD4 oracle bounds; and a fixed zero-threshold candidate preference check. A correct TD4 prediction on a shared SI/I error is **an opportunity, not achieved fusion accuracy**. LDA probabilities are not calibrated. Reused CV scores are selection scores, not unbiased validation estimates. The GitHub Action checks all 20 subjects, exact seed/window coverage, 695,163 window–seed evaluations and the result ZIP CRC before reporting completion.

## Historical local validation (before the fold-scale correction)

The initial implementation ran on all 20 local DB7 files and all 695,163 saved prediction rows. Full TD4 pooled accuracy was 71.6823%; it corrected 3,584 of the 10,043 SI/I shared errors as a standalone classifier. A fixed candidate-margin switch recovered 2,764 SI errors and harmed 2,230 SI-correct cases, for pooled accuracy 97.8572% versus SI's 97.7804%. These historical numbers are superseded by the corrected rerun: the initial implementation included the held-out training repetition when estimating a CV fold's threshold scale. Test labels were still excluded. This is not a BRB result or independent test confirmation.

## Rerun repairs

The notebook accepts compressed or Kaggle-decompressed trace files and checks their canonical content hash before fitting. The workflow downloads and verifies the ready trace dataset before submission. The corrected CV scale isolation and all-subset complementarity outputs were checked on S1; completion requires their verification flags. Prior Kaggle attempts failed before any classifier fitting.

The Kaggle notebook is [`db7-030-hudgins-td4.ipynb`](db7-030-hudgins-td4.ipynb); the implementation is [`scripts/db7_030_hudgins.py`](scripts/db7_030_hudgins.py). The workflow is [`.github/workflows/db7-030-hudgins.yml`](.github/workflows/db7-030-hudgins.yml).
