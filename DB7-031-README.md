# DB7-031: trainable hierarchical Hudgins BRB

Direct 17-gesture classification using all 48 EMG features. Twelve channel BRBs each have four antecedents (MAV/WL/ZC/SSC), three references per antecedent, 81 rules and 17 consequent beliefs. A second ER aggregation combines channel beliefs. This is the user-selected hierarchical design, not the infeasible complete 3^48 rule grid and not an SI-to-I gate.

Jointly optimized parameters: consequent beliefs, rule weights, attribute weights and channel fusion weights. Three raw reference values (fitting-data 5th/50th/95th percentiles) are fixed during optimization. Tiny tie expansion keeps references strictly ordered. Full initial/trained rule tables and weights are exported.

S1–S20, E1 labels1–17; train repetitions1/3/4/6, test2/5. Windows200ms, training stride50ms, test stride10ms; same filtered48 features as DB7-030. Four training-repetition validation folds select the epoch; nested threshold screening excludes each outer validation repetition. Final ZC/SSC thresholds match DB7-030. The LDA baseline must reproduce the parent result before BRB results are accepted.

Three BRB seeds42/43/44. Up to60 epochs per validation fit, Adam0.02, batch512, cosine T_max60, regularization0.001, gradient clipping5. Final refits stop at their training-validation-selected epoch. This adds240 validation plus60 final BRB fits; it does not reuse the earlier CNN's13-epoch training duration. Two Kaggle GPUs split subjects between processes; the run requires both.

Launch via `.github/workflows/db7-031-brb.yml`. It runs CPU ER/gradient/learning checks before submitting `db7-031-hudgins-hierarchical-brb.ipynb` to Kaggle. To resume monitoring an existing submission, supply `monitor_ref` instead of creating a duplicate.

No accuracy is claimed before completion. The result archive includes mean and pooled accuracy, macro-F1, per-subject/gesture errors, confusion matrices, paired subject uncertainty versus matched LDA, reference values, trained weights, rule beliefs and model states. These previously inspected recordings provide exploratory evidence, not independent confirmation. SI includes inertial inputs, so it is not a modality-matched baseline for this EMG-only experiment.
