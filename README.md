# Automate DB7 experiments with GitHub Actions and Kaggle

GitHub stores the notebook and starts the job. Kaggle trains on its GPU. GitHub downloads the diagnostic ZIP and saves it as an Actions artifact. Your computer can be turned off after launch.

This starter uses the amplitude–velocity notebook already prepared for your experiment. It runs the original baseline and derivative candidate with validation-only evaluation. It does not tune hyperparameters automatically, inspect results with an AI service, or modify the model based on scores.

## 1. Create a private GitHub repository

Choose a name such as `ninapro-db7-experiments` and create a README so that the default branch exists. Upload the files from this starter to the repository root; do not upload the starter ZIP as one file.

Expected structure:

```text
.github/workflows/run-experiment.yml
.gitignore
experiment.json
requirements.txt
scripts/run_experiment.py
notebooks/mka-db7-amplitude-velocity.ipynb
README.md
```

You can use GitHub Desktop: clone the new repository, copy these files into its folder, commit, and push. Ensure `.github/workflows/run-experiment.yml` exists at that exact path on the default branch. Windows can hide dot-prefixed folders in some file pickers. If necessary, create that file using GitHub's Add file → Create new file and paste its contents.

Do not commit Kaggle credentials, raw DB7 recordings, result ZIPs or checkpoint files.

## 2. Create a Kaggle API token

Open https://www.kaggle.com/settings/api and generate a new API token. Copy the token value for the GitHub secret in the next step. Use the modern token, not the entire legacy `kaggle.json` contents.

Your existing successful Kaggle GPU runs indicate that the account can run these notebooks. Check its current GPU allowance before a large seed sweep.

Official authentication documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication

## 3. Add the token to GitHub Secrets

In your repository:

1. Settings → Secrets and variables → Actions.
2. On the Secrets tab, choose New repository secret.
3. Name: `KAGGLE_API_TOKEN`.
4. Value: paste the Kaggle token.
5. Save it.

Do not paste the token into this README, a notebook, workflow code, an issue, or chat.

Official instructions: https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets

## 4. Add your Kaggle username as a variable

On the same repository page, switch to Variables → New repository variable.

- Name: `KAGGLE_USERNAME`
- Value: the username in your Kaggle profile URL, e.g. `yourname` from `https://www.kaggle.com/yourname`.

It is your Kaggle username, not necessarily your GitHub username, email address or displayed full name. No GitHub personal access token is needed for this workflow.

## 5. Confirm the attached dataset

Open `experiment.json`. Its dataset source is already set to the one in your diagnostic manifest:

```json
"dataset_sources": ["rayaanraza1/ninapro-db7"]
```

Confirm that you can access that dataset on Kaggle. The dataset reference is `owner/dataset-slug`; do not put the `/kaggle/input/...` filesystem path here.

The accelerator is explicitly `NvidiaTeslaT4`, matching your previous runs. The notebook needs no internet during training. GitHub's runner does need internet to install the Kaggle client and call the API.

Metadata documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md

## 6. Run a smoke test

Go to repository Actions → Run DB7 on Kaggle → Run workflow.

- Branch: your default branch.
- `scope`: `smoke`.
- `seeds`: `42`.

Click Run workflow. This trains both variants for S01 only, including the notebook's synthetic preflight.

The Launch, monitor and collect step prints the URL of the newly created private Kaggle notebook. Open that link if you want to inspect execution there. The GitHub job checks status every minute.

Each seed uses a unique notebook name containing GitHub's run ID and retry attempt. Existing manual Kaggle notebooks are not overwritten. These private automated notebooks remain in your Kaggle account after completion; the workflow does not delete them.

The Kaggle push command both uploads and starts the notebook; you do not need to press Run All separately. See https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md

## 7. Download the result from GitHub

After the workflow finishes, open the Actions run and find Artifacts. Download `db7-<run-id>-<attempt>`.

Inside it, `results/seed-42/` contains the diagnostic ZIP, execution/status logs, the Kaggle link and extracted summary files. `submitted/seed-42/` contains the exact submitted notebook, metadata and commit/hash provenance. The inner diagnostic ZIP contains the full reports and checkpoints.

The artifact is retained for 14 days. Download important results before it expires. If you cannot find the folder under that exact nesting, search the extracted artifact for `seed-42`—the artifact action preserves paths relative to their common parent.

The script checks that the returned ZIP has the expected model seed, validation-only mode, and every requested subject/model pair. A partial ZIP or `FAILURE.txt` causes the workflow to fail, while available files are still uploaded.

Artifact documentation: https://github.com/actions/upload-artifact

## 8. Run all subjects and repeat seeds

After the smoke test succeeds, run the workflow again with:

```text
scope: all
seeds: 42
```

This trains 22 subjects × 2 variants = 44 fits.

For an initialization comparison, use:

```text
scope: all
seeds: 42,43,44
```

Seeds run sequentially, not simultaneously. Each uses the same repetition split seed of 42; only the model/shuffle seed changes. Three seeds require 132 model fits. You receive one Kaggle notebook and diagnostic ZIP per seed.

Up to three seeds are accepted per Actions run. The default monitoring limit is 100 minutes per seed and the Actions job limit is 350 minutes. If your runs need longer, run one seed at a time and increase the per-seed setting and job limit appropriately within platform limits.

Concurrency prevents two active jobs from this workflow from running simultaneously. It is not a durable unlimited queue: avoid repeatedly pressing Run workflow while another request is pending. Manual jobs or other repositories can still use your Kaggle quota.

## 9. Optional: launch automatically when code changes

Manual launch is the initial default. To start a smoke run whenever notebook code is pushed, add this beside `workflow_dispatch` under `on:` in the workflow file (change `main` if needed):

```yaml
  push:
    branches: [main]
    paths:
      - 'notebooks/**'
      - 'scripts/**'
      - 'experiment.json'
```

Push-triggered runs use the fallback `scope=smoke` and seed 42. Keep full runs manually triggered initially so ordinary commits do not launch 44 training fits.

For a scheduled daily smoke test, add this under `on:` instead or as well:

```yaml
  schedule:
    - cron: '17 18 * * *'
```

This requests 18:17 UTC (00:17 the next day in Bangladesh). The supplied schedule defaults to a smoke test, not an all-subject sweep. Scheduled events can be delayed and use the default branch. No schedule is enabled in the supplied starter.

Workflow triggers: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax

## Cost, quota and interruption behavior

Training uses Kaggle's GPU allowance. This simple implementation leaves GitHub's Ubuntu runner active while it polls, so waiting consumes Actions time in a private repository. Artifacts consume storage. Review your account allowances before enabling frequent runs. GitHub billing documentation: https://docs.github.com/en/billing/concepts/product-billing/github-actions

If you cancel GitHub Actions or it times out, Kaggle may continue training. Open the saved Kaggle link to inspect or stop that run; canceling the GitHub job does not reliably cancel the remote notebook. Before retrying a failed submission, check the link: Kaggle may have accepted the request even if GitHub lost its response.

The script does not automatically re-submit after an ambiguous push failure. It retries read-only status/output requests. No quota bypass or automatic billing change is included.

## Troubleshooting

| Symptom | Check |
|---|---|
| Workflow not visible | File is at `.github/workflows/run-experiment.yml` on the default branch; Actions is enabled. |
| Missing token | Secret is named exactly `KAGGLE_API_TOKEN` and has the modern token value. |
| 401/403 | Token validity, correct Kaggle username and access to the dataset/notebook. |
| GPU unavailable or job queued | Kaggle account's current accelerator quota and availability. |
| No matching diagnostic ZIP | Kaggle log or notebook failed before generating outputs; inspect saved logs. |
| GitHub monitor timeout | Kaggle may still be active; use the saved notebook URL. |
| All files exist but workflow fails | Check for `FAILURE.txt`, incomplete result rows, or mismatched manifest. |

The notebook and payload generation were checked locally without credentials or remote submission. Kaggle execution, GitHub Actions and authenticated output collection have not been run from your accounts. Your first smoke run validates that integration.
