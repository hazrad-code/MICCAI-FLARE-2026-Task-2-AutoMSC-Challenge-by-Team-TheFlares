# Evaluating Self-Configuring Multi-Task Learning for Joint Segmentation and Case-Level Classification of Heterogeneous 3D Medical Images

**Team theflares — MICCAI FLARE 2026, Task 2 (AutoMSC)**

This repository is the official implementation of
**Evaluating Self-Configuring Multi-Task Learning for Joint Segmentation and Case-Level Classification of Heterogeneous 3D Medical Images**,
our submission to Task 2 (AutoMSC: Automated Medical Image Segmentation and
Classification) of the MICCAI FLARE 2026 challenge.

We do not propose a new architecture. We take the organisers'
[AutoMSC baseline](https://github.com/medfm-flare/AutoMSC-Baseline) — an nnU-Net v2 fork
with a multi-task classification head — apply it unchanged to two previously unseen
datasets, and evaluate where it succeeds and where it does not:

- **Dataset009_PHLF** — contrast-enhanced hepatic MRI, four-structure segmentation, binary outcome (post-hepatectomy liver failure)
- **Dataset008_EGCT** — non-enhanced paediatric CT, tumour segmentation, three-class tumour subtype

**Main finding.** Segmentation transfers across both datasets without manual tuning
(DSC 85–96% for organs, 75.5% and 85.0% for tumours). Classification ranks well —
per-class AUC 0.84–0.89 on EGCT — but on the three-class task the argmax decisions
concentrate on the majority class (balanced accuracy 0.472). The gap is between
**ranking quality** and **decision quality**, not in the shared encoder.

---

## Repository structure

```
.
├── codebase/
│   ├── AutoMSC-Baseline/     upstream baseline, with the modifications listed under Contributing
│   ├── run_dataset.sh        end-to-end driver: preprocess → splits → train → infer → package
│   ├── make_splits.py        stratified 5-fold splits + normalised cls_data.csv
│   ├── build_submission.py   raw predictions → challenge submission format
│   └── verify_masks.py       geometry check of predicted masks
├── predictions/              submitted test-set predictions (40 PHLF, 129 EGCT)
└── results/                  evaluation logs, metrics, training logs and curves
```

---

## Environments and Requirements

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS (container image) |
| CPU | 192 cores |
| RAM | 1 TB |
| GPU | 1× NVIDIA H200 slice, 24.5 GB addressable |
| CUDA driver | 12.8 (`12080`) |
| CUDA runtime (PyTorch wheel) | 12.1 |
| Python | 3.11.10 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |

Runs in a container-based JupyterHub environment accessed through virtual desktop
infrastructure (VDI). A CUDA GPU and Python 3.11 are required.

### Installation

```bash
# 1. uv
pip install -U uv
export PATH="$HOME/.local/bin:$PATH"   # pip installs uv here; not on PATH by default

# 2. virtual environment  (the --python flag is required, see note below)
cd codebase/AutoMSC-Baseline
uv venv --python 3.11 .venv
source .venv/bin/activate        # bash/zsh. The JupyterLab default shell is dash,
                                 # where `source` does not exist — run `bash` first.

# 3. the framework
uv pip install -e .
uv pip install wandb torchmetrics transformers

# 4. PyTorch matching your driver (this pin matters, see below)
uv pip install --reinstall torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 5. evaluation only
uv pip install SimpleITK scikit-learn pandas
```

Verify before continuing:

```bash
python -V                                  # expected: Python 3.11.x
python -c "import torch, nnunetv2; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu121 True
```

### Installation notes

These issues were found by reinstalling from this repository into a clean environment.
Each one blocks the pipeline, and two fail silently.

**`uv` is installed outside `PATH`.** `pip install -U uv` writes to `~/.local/bin`,
which is not on `PATH` by default, so the next command fails with
`bash: uv: command not found`. Hence the `export PATH=...` line above.

**`uv venv` without `--python` selects the newest interpreter on the machine.**
On a host with Python 3.13 this creates an environment for which
`torch==2.5.1+cu121` has no wheels, so the pin in step 4 fails to resolve while the
CUDA 13 build pulled in by step 3 stays installed. Installation *appears* to succeed,
but `torch.cuda.is_available()` returns `False` and training silently runs on CPU.
Always check `python -V` first.

**Why PyTorch is pinned.** The current default PyPI wheel is built against CUDA 13 and
refuses to initialise on a CUDA 12.x driver:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
```

Install the `cu121` build as above. If your driver supports CUDA 13, the default wheel
is fine and step 4 can be skipped.

**`transformers` is required even though it is not used here.** nnU-Net resolves a
trainer class by importing every module beneath `nnunetv2/training/nnUNetTrainer/`, and
the experimental `primus` variants import `transformers`. Without it, no training can
start regardless of which trainer is requested.

**`primus` has been removed from this copy.** `primus/retension_primus.py` imports a
`retention` package that is not installable from PyPI, which blocks the same recursive
trainer lookup. The only remaining reference is a string comparison in
`nnunetv2/inference/predict_from_raw_data.py`.

### Environment variables

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export WANDB_MODE=disabled
```

**Do not export `nnUNet_n_proc_DA=0` globally.** `run_dataset.sh` sets it per step,
because the value that training needs breaks preprocessing. See *Training* below.

---

## Dataset

The PHLF and EGCT datasets are distributed by the FLARE 2026 organisers under CC BY 4.0
and are **not redistributed here**. Obtain them from the challenge:
[FLARE-MedFM/FLARE-AutoMSC on Hugging Face](https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC).

Expected layout — standard nnU-Net raw convention plus two classification files:

```
$nnUNet_raw/Dataset0XX_NAME/
├── dataset.json          channels, label schema, classification_labels, counts
├── imagesTr/             {case_id}_0000.nii.gz  (…_0001, … for extra channels)
├── labelsTr/             {case_id}.nii.gz
├── imagesTs/             {case_id}_0000.nii.gz
├── cls_data.csv          training classification labels
└── clinical_data.csv     optional clinical variables
```

`cls_data.csv` needs an identifier column and a label column. The identifier column is
renamed to `identifier` automatically (the data loader requires that name); if a column
called `label` is absent, the second column is used.

Datasets used:

| | Dataset009_PHLF | Dataset008_EGCT |
|---|---|---|
| Modality | Gd-EOB-DTPA MRI, single channel | Non-enhanced CT, single channel |
| Clinical context | Hepatectomy candidates, 3 centres | Paediatric extracranial germ cell tumours |
| Segmentation labels | 0 background, 1 liver, 2 tumour, 3 spleen, 4 muscle | 0 background, 1 tumour |
| Classification target | PHLF (0 absent, 1 present) | Tumour subtype (0 MT, 1 IT, 2 MGCT) |
| Training / test cases | 159 / 40 | 512 / 129 |
| Class distribution (training) | 147 / 12 | 331 / 84 / 97 |

MT: mature teratoma; IT: immature teratoma; MGCT: malignant germ cell tumour.

Clinical variables in `clinical_data.csv` are not used by this pipeline, so differences
in their column names and encodings between datasets (for example `age`/`gender` numeric
versus `age_months`/`gender` as `F`/`M`) do not affect it.

---

## Preprocessing

Preprocessing is nnU-Net's own, derived automatically from each dataset's fingerprint:
cropping to the nonzero region, resampling to a target spacing, and intensity
normalisation chosen by modality. No parameters were set by hand.

```bash
nnUNet_n_proc_DA=8 nnUNetv2_plan_and_preprocess -d <DATASET_ID> -c 3d_fullres --verify_dataset_integrity
```

Resulting configuration (from `$nnUNet_preprocessed/<Dataset>/nnUNetPlans.json`, copies in `results/`):

| Parameter | Dataset009_PHLF | Dataset008_EGCT |
|---|---|---|
| Target spacing (mm) | 3.0 × 1.1364 × 1.1364 | 5.0 × 0.46875 × 0.46875 |
| Median image size (voxels) | 64 × 250 × 351 | 45 × 512 × 512 |
| Patch size | 48 × 160 × 256 | 24 × 256 × 256 |
| Batch size | 2 | 2 |
| Normalisation | Z-score (per image) | CT (percentile clip + global z-score) |
| Architecture | PlainConvUNet, 6 stages | PlainConvUNet, 7 stages |
| Features per stage | 32, 64, 128, 256, 320, 320 | 32, 64, 128, 256, 320, 320, 320 |

The normalisation scheme differs between the two datasets: nnU-Net selects CT
normalisation for the CT dataset and z-score for the MRI dataset automatically.

Cross-validation splits and the normalised classification table:

```bash
python codebase/make_splits.py $nnUNet_raw/Dataset009_PHLF $nnUNet_preprocessed/Dataset009_PHLF
```

Splits are stratified by classification label using
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, so they are
deterministic and are not shipped (they list challenge case identifiers).

| | Dataset009_PHLF | Dataset008_EGCT |
|---|---|---|
| Fold 0 train / validation | 127 / 32 | 409 / 103 |
| Fold 0 validation class distribution | 30 / 2 | 67 / 16 / 20 |

---

## Training

Full pipeline for one dataset — preprocessing, splits, training, inference, packaging:

```bash
bash codebase/run_dataset.sh 9 Dataset009_PHLF
bash codebase/run_dataset.sh 8 Dataset008_EGCT
```

Training alone:

```bash
nnUNet_n_proc_DA=0 nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetCLSTrainerMTL_250epochs
```

Trained checkpoints (`checkpoint_best.pth`, ≈437 MB each, one per dataset) are available on request from the authors, subject to the FLARE 2026 data terms.

### Configuration

| Setting | Value |
|---|---|
| Trainer | `nnUNetCLSTrainerMTL_250epochs` |
| Configuration | 3D full resolution |
| Plans | default nnU-Net plan |
| Fold | 0 only, no ensembling |
| Epochs | 250 |
| Initial learning rate | 0.01 |
| LR scheduler | polynomial decay |
| Segmentation loss | Dice + cross-entropy, deep supervision |
| Classification loss | selected automatically (below) |
| Task weighting | uncertainty-weighted; realised 0.5 : 0.5 (below) |
| Checkpoint selection | mean pseudo-Dice + classification AUC on the validation fold |

### `nnUNet_n_proc_DA` — different values for different steps

nnU-Net's data-augmentation workers exchange tensors through `/dev/shm`. Containers
commonly provide only 64 MB, which the workers exhaust within the first epoch:

```
RuntimeError: unable to write to file </torch_..._0>: No space left on device (28)
RuntimeError: One or more background workers are no longer alive.
```

`nnUNet_n_proc_DA=0` selects `SingleThreadedAugmenter` and removes shared-memory use
entirely, at a cost of roughly 25–30% in epoch time. Changing PyTorch's sharing strategy
to `file_system` does not help — on Linux both strategies allocate in `/dev/shm`.

**That value must not be applied to preprocessing.** `nnUNetv2_plan_and_preprocess` calls
`torch.set_num_threads(get_allowed_n_proc_DA())`, and with a value of 0 this raises
`RuntimeError: set_num_threads expects a positive integer`, aborting the script at its
first step. `run_dataset.sh` therefore sets a positive default and overrides it to 0 only
on the training command.

A shell default of the form `${nnUNet_n_proc_DA:-8}` does **not** fix this: `:-` only
substitutes when the variable is unset or empty, so an exported `0` inherited from the
environment still wins. The value must be set unconditionally.

If your host provides a larger `/dev/shm` (e.g. `docker run --shm-size=8g`), raise the
training value to the number of workers you can afford.

### Automatic per-dataset adaptation

Besides the preprocessing parameters above, the classification loss is chosen from the
training class counts by `DynamicClassificationLoss`
(`nnunetv2/training/nnUNetTrainer/nnUNetCLSTrainer.py`), using the rule
`class_weights.max() // class_weights.min() >= 10`:

| | binary | multi-class |
|---|---|---|
| balanced | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| imbalanced | `FocalBCEWithLogitsLoss` (α=0.25, γ=2.0) | `FocalCEWithLogitsLoss` |

PHLF (147/12, ratio 12) crosses the threshold and trains with focal BCE; EGCT (331/84,
ratio 3) does not and trains with cross-entropy — despite being meaningfully imbalanced.
See *Limitations*.

### Task weighting did not adapt

`JointUncertaintyLoss` (`nnunetv2/training/loss/uncertainty_loss.py`) implements
homoscedastic uncertainty weighting (Kendall et al., CVPR 2018) with learnable per-task
log-variances:

```
L = ½·exp(−s_seg)·L_seg + ½·exp(−s_cls)·L_cls + ½·(s_seg + s_cls),   s = log σ²
```

In practice those parameters never update. The optimiser is constructed at
`nnUNetCLSTrainer.py:388`, before `self.uncertainty_loss` is created at line 397, and
`configure_optimizers()` collects parameters only from `self.network`. `log_vars`
therefore receives gradients but is never stepped, and stays at its initial value of
zero. Every logged checkpoint across 250 epochs on both datasets reports effective
weights of `0.5 / 0.5`. The realised objective is an equally weighted sum.

This is a property of the baseline, reported here so that anyone building on it is
aware of it.

### Training length

`nnUNetCLSTrainerMTL_250epochs` is defined in
`nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetCLSTrainerMTL_Xepochs.py`,
alongside 100- and 500-epoch variants. Pass a different trainer as the third argument to
`run_dataset.sh`.

The epoch count is set inside `initialize()` rather than `__init__`, because
`nnUNetTrainer.__init__` records its own parameters by inspecting the signature of
`self.__init__` and looking each name up in `locals()`. A subclass declaring
`__init__(*args, **kwargs)` therefore raises `KeyError: 'args'`.

Reducing the epoch count is preferable to interrupting a longer run: `PolyLRScheduler`
anneals the learning rate across exactly `num_epochs`, so a 1000-epoch schedule stopped
early leaves the learning rate un-decayed.

### Applying the method to a new dataset

Place the dataset under `$nnUNet_raw/` in the layout above and run
`bash codebase/run_dataset.sh <DATASET_ID> <Dataset0XX_NAME>`. Everything below is read
from the data:

| Property | Source |
|---|---|
| Number of input channels | `dataset.json → channel_names` |
| Segmentation label schema | `dataset.json → labels` |
| Number of classification classes | `dataset.json → classification_labels` |
| Number of test cases | file count in `imagesTs/` |
| Identifier column name | first column of `cls_data.csv`, renamed to `identifier` |
| Label column name | `label` if present, otherwise the second column |
| Fold count | `min(5, size of the rarest class)`, floor 2 |
| Output CSV schema | single `label` column for 2 classes, otherwise `label_0 … label_{k-1}` normalised to sum to 1 |

---

## Inference

```bash
python codebase/AutoMSC-Baseline/segcls_ensemble_infer.py \
  -i  $nnUNet_raw/Dataset009_PHLF/imagesTs \
  -o  work/predictions_raw/Dataset009_PHLF \
  --model_path $nnUNet_results/Dataset009_PHLF/nnUNetCLSTrainerMTL_250epochs__nnUNetPlans__3d_fullres \
  --fold 0 --checkpoint checkpoint_best.pth --device cuda --cls_mode mean
```

Then package and verify:

```bash
python codebase/build_submission.py $nnUNet_raw/Dataset009_PHLF \
                                    work/predictions_raw/Dataset009_PHLF work/submission
python codebase/verify_masks.py     work/submission/predictions/Dataset009_PHLF_prediction \
                                    $nnUNet_raw/Dataset009_PHLF/imagesTs
```

### Output format

```
work/submission/predictions/
├── Dataset008_EGCT_prediction/
│   ├── {case_id}.nii.gz          one per test case
│   └── predictions.csv           case_id, label_0, label_1, label_2
└── Dataset009_PHLF_prediction/
    ├── {case_id}.nii.gz
    └── predictions.csv           case_id, label
```

`predictions.csv` holds probabilities, not hard labels; for multi-class tasks the values
sum to 1. The submitted files are in `predictions/`.

The classification head emits per-class **sigmoid** outputs (`segcls_ensemble_infer.py`,
`class_probs = torch.sigmoid(class_logit)`), which are not normalised. For binary tasks
the script writes a two-element list of identical values and `probs[0]` is the
probability of the positive class; for multi-class tasks `build_submission.py` divides
the per-class values by their sum so the row totals 1.

Predicted masks are written by nnU-Net's own writer and inherit the source image's
shape, spacing, origin and direction. `verify_masks.py` re-checks this against
`imagesTs/` and reports the set of label values present.

### Runtime and memory

Measured on the hardware above.

| Stage | Dataset009_PHLF | Dataset008_EGCT |
|---|---|---|
| Preprocessing | 2 min 43 s | ~9 min |
| Training (250 epochs, fold 0) | 5 h 19 min (~76 s/epoch) | 2 h 06 min (~29 s/epoch) |
| Inference | ~7 s/case, 40 cases | ~13 s/case, 129 cases |

| Stage | Peak GPU memory | Budget |
|---|---|---|
| Training (3d_fullres, batch size 2) | not separately measured | ≤ 24 GB |
| Inference (PHLF plan, 48×160×256) | 2.53 GB allocated / 3.43 GB reserved, plus ≈0.5 GB CUDA context | ≤ 12 GB |

Measured with `torch.cuda.reset_peak_memory_stats()` and `max_memory_allocated()` /
`max_memory_reserved()` around an in-process invocation of the inference script. EGCT
inference memory was not measured. Epoch time is dominated by the planner's patch size
and by the single-threaded augmentation described above.

---

## Evaluation

Test labels are withheld, so all reported metrics are computed on the **fold-0
validation split** of the training data, by full-volume inference with the same
checkpoint and settings as the submission.

```bash
# 1. fold-0 validation set: the case identifiers are listed under fold 0, "val", in
#    $nnUNet_preprocessed/<Dataset>/splits_final.json (written by make_splits.py).
#    Copy or symlink imagesTr/{case}_0000.nii.gz -> valeval/<Dataset>/images/
#    and labelsTr/{case}.nii.gz -> valeval/<Dataset>/labels/

# 2. inference with the submitted checkpoint
python codebase/AutoMSC-Baseline/segcls_ensemble_infer.py \
  -i valeval/Dataset009_PHLF/images -o valeval/Dataset009_PHLF/pred \
  --model_path $nnUNet_results/Dataset009_PHLF/nnUNetCLSTrainerMTL_250epochs__nnUNetPlans__3d_fullres \
  --fold 0 --checkpoint checkpoint_best.pth --device cuda --cls_mode mean

# 3. DSC, NSD and classification metrics
python codebase/AutoMSC-Baseline/eval_metrics.py \
  --pred_seg_path valeval/Dataset009_PHLF/pred \
  --gt_seg_path   valeval/Dataset009_PHLF/labels \
  --pred_cls_csv  valeval/Dataset009_PHLF/pred/fold0_results.csv \
  --gt_cls_csv    $nnUNet_preprocessed/Dataset009_PHLF/cls_data.csv \
  --num_seg_classes 5 --num_cls_classes 2 \
  --output_csv    results/phlf_val_metrics.csv
```

For EGCT use `--num_seg_classes 2 --num_cls_classes 3`.

**NSD uses a fixed tolerance of τ = 1 mm** for every structure — this is hardcoded in
`eval_metrics.py` and is not the per-organ tolerance used by the official challenge
scoring, so these NSD values are not directly comparable to the leaderboard. For PHLF the
tolerance is also smaller than the 3.0 mm slice spacing, so a one-slice boundary error
fails.

The evaluator's binary branch does not report balanced accuracy; for PHLF it is
(Sensitivity + Specificity) / 2. For EGCT, macro AUC is reported as the mean of the
three per-class one-vs-rest AUCs, and macro F1 as the mean of the per-class F1 scores.

---

## Results

### Full-volume evaluation, fold-0 validation

`checkpoint_best.pth`, mean ± SD across cases. Full logs: `results/phlf_eval.log`,
`results/egct_eval.log`.

**Dataset009_PHLF** — 32 cases, 30 negative / 2 positive

| Structure | DSC (%) | NSD (%, τ = 1 mm) |
|---|---|---|
| Liver | 96.4 ± 2.6 | 77.6 ± 13.9 |
| Tumour | 75.5 ± 28.2 | 54.8 ± 26.6 |
| Spleen | 90.7 ± 15.9 | 69.3 ± 19.5 |
| Muscle | 85.2 ± 13.3 | 49.2 ± 13.9 |
| **Overall** | **86.9 ± 19.0** | **62.7 ± 22.1** |

| Classification | Value |
|---|---|
| AUC | 0.967 |
| Balanced accuracy | 0.983 |
| AUPRC | 0.583 |
| Sensitivity | 1.000 (2/2) |
| Specificity | 0.967 (29/30) |
| F1 | 0.800 |

**Dataset008_EGCT** — 103 cases, 67 MT / 16 IT / 20 MGCT

| Structure | DSC (%) | NSD (%, τ = 1 mm) |
|---|---|---|
| Tumour | 85.0 ± 16.8 | 68.8 ± 19.1 |

| Classification | Value |
|---|---|
| Accuracy | 0.689 (71/103; majority-class rate 0.650) |
| Balanced accuracy | 0.472 (chance 0.333) |
| Macro AUC | 0.860 |
| Macro F1 | 0.480 |
| Weighted F1 | 0.626 |

| Class | Recall | Precision | F1 | AUC |
|---|---|---|---|---|
| MT (0) | 0.940 (63/67) | 0.692 | 0.798 | 0.842 |
| IT (1) | 0.375 (6/16) | 0.600 | 0.462 | 0.848 |
| MGCT (2) | 0.100 (2/20) | 1.000 | 0.182 | 0.890 |

Confusion matrix (rows = truth, columns = prediction): `[[63, 4, 0], [10, 6, 0], [18, 0, 2]]`.

### What the results show

**Segmentation transfers.** The three PHLF organs reach DSC 85.2–96.4% and the EGCT
tumour 85.0%, with no dataset-specific tuning. PHLF tumour is the weakest structure
(75.5%, with large dispersion): two of the 32 tumours were missed entirely, while the
remaining predictions retained high precision.

**Classification ranks well but decides poorly on the three-class task.** Every EGCT
class has AUC between 0.84 and 0.89, so the predicted probabilities discriminate between
subtypes. The argmax decisions, however, assign 91 of 103 cases to the majority class,
and 18 of the 20 MGCT cases are classified as MT. The test set shows the same pattern:
the predicted argmax distribution is 111 / 17 / 1 across 129 cases.

**PHLF classification is coarse.** Both positives were detected with one false positive,
but with only two positive cases these estimates carry wide uncertainty (see
*Limitations*).

### Training-time metrics (pseudo-Dice), for reference

nnU-Net's pseudo-Dice is computed during training from randomly sampled validation
patches rather than full-volume predictions. It is a progress proxy, not a comparable
metric, and it overstates segmentation quality here — most for small structures: PHLF
tumour reads 0.882 as pseudo-Dice against 0.755 as full-volume DSC at the same
checkpoint.

| | Dataset009_PHLF | Dataset008_EGCT |
|---|---|---|
| Pseudo-Dice, final epoch | liver 0.964, tumour 0.890, spleen 0.943, muscle 0.905 | tumour 0.913 |
| Classification accuracy, final epoch | 0.813 | 0.657 |
| Classification AUC, final epoch | 0.949 | 0.785 |

Per-epoch series are in `results/phlf_curve.csv` and `results/egct_curve.csv`; the
original training logs are `results/*_training_log.txt`.

nnU-Net validates over `⌊n / batch_size⌋ × batch_size` samples, so EGCT's training-time
metrics cover 102 of its 103 validation cases.

### Post-hoc analysis: prior correction (not part of the submission or the paper)

Dividing the normalised EGCT probabilities by `prior**tau` and re-taking the argmax,
without retraining:

| τ | Balanced accuracy | Macro F1 | Predicted counts |
|---|---|---|---|
| 0.0 (as submitted) | 0.472 | 0.480 | 91 / 10 / 2 |
| 0.3 | 0.625 | 0.640 | 74 / 21 / 8 |
| 0.6 | 0.654 | 0.537 | 31 / 53 / 19 |
| 0.8 | 0.617 | 0.361 | 0 / 77 / 26 |

Consistent with the ranking-versus-decision reading above: adjusting only the decision
rule recovers much of the lost balanced accuracy. It was not applied to the submitted
predictions.

---

## Limitations

- Fold 0 only; no five-fold ensembling.
- 250 training epochs rather than the nnU-Net default of 1000, chosen to fit the
  submission window. Data augmentation ran single-process because of the 64 MB
  shared-memory limit.
- No hyperparameter search; the baseline configuration is used as published, with the
  deviations documented above.
- Test labels are withheld, so all metrics come from the fold-0 validation split, and
  `checkpoint_best.pth` was selected on that same split — the reported metrics, the
  classification metrics especially, may be biased upward.
- PHLF's validation fold contains 2 positive cases, so its classification metrics are
  very coarse: sensitivity can only take the values 0, 0.5 or 1.0, and a single rank
  change moves AUC by ≈0.017. The two positives score 0.746 and 0.508 against a nearest
  negative at 0.497 — a margin of 0.011, so thresholded metrics are fragile.
- Classification probabilities are per-class sigmoid outputs normalised to sum to 1, not
  a softmax over logits. This is a property of the baseline inference script; because it
  was not varied, its contribution to the majority-class bias cannot be separated from
  that of the reduced training budget.
- The focal-loss threshold of 10 is hardcoded; EGCT's class ratio of ≈3.9 falls below
  it, so its three-class task trains with plain cross-entropy despite meaningful
  imbalance.
- Task weighting did not adapt (see *Training*), so all multi-task results were obtained
  under fixed, equal weighting.
- NSD is computed at a fixed 1 mm tolerance, not the official per-organ tolerances.

---

## Contributing

This repository is released under the Apache License 2.0, inherited from the
[AutoMSC baseline](https://github.com/medfm-flare/AutoMSC-Baseline). See `LICENSE`.

Modifications to the upstream code:

- added `nnUNetCLSTrainerMTL_250epochs` (and 100/500-epoch variants) in
  `variants/training_length/nnUNetCLSTrainerMTL_Xepochs.py`
- removed `primus/` (uninstallable `retention` dependency)
- `run_dataset.sh`, `make_splits.py`, `build_submission.py` and `verify_masks.py` are new

Issues and pull requests are welcome.

---

## Acknowledgement

We thank the FLARE 2026 organisers for the datasets and the AutoMSC baseline, and the
nnU-Net authors for the framework this work builds on.

During the preparation of this work the author(s) used Claude in order to improve language and readability, with caution. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.
