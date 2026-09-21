#!/usr/bin/env bash
#
# End-to-end AutoMSC pipeline for a single dataset in nnU-Net raw format.
#
#   bash run_dataset.sh <DATASET_ID> <Dataset0XX_NAME> [TRAINER]
#
# Example:
#   bash run_dataset.sh 9 Dataset009_PHLF
#   bash run_dataset.sh 8 Dataset008_EGCT
#
# Nothing in this script is specific to a dataset: the number of input
# channels, the segmentation label schema and the number of classification
# classes are all read from dataset.json at run time.

set -euo pipefail

usage() {
  echo "Usage: bash run_dataset.sh <DATASET_ID> <Dataset0XX_NAME> [TRAINER]" >&2
  exit 1
}

[ $# -ge 2 ] || usage
DATASET_ID="$1"
DATASET_NAME="$2"
TRAINER="${3:-nnUNetCLSTrainerMTL_250epochs}"

: "${nnUNet_raw:?nnUNet_raw is not set}"
: "${nnUNet_preprocessed:?nnUNet_preprocessed is not set}"
: "${nnUNet_results:?nnUNet_results is not set}"

# 0 selects nnU-Net's SingleThreadedAugmenter. Required on hosts with a small
# /dev/shm (64 MB is the Docker/Kubernetes default); raise it if your host has
# more shared memory available.
export nnUNet_n_proc_DA=8      # positive for preprocessing/inference;training overrides to 0 below

export WANDB_MODE="${WANDB_MODE:-disabled}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/../work}"
mkdir -p "$WORK"

echo "=== [1/5] preprocessing $DATASET_NAME ==="
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 3d_fullres --verify_dataset_integrity

echo "=== [2/5] classification data and cross-validation splits ==="
python "$HERE/make_splits.py" \
  "$nnUNet_raw/$DATASET_NAME" \
  "$nnUNet_preprocessed/$DATASET_NAME"

echo "=== [3/5] training fold 0 with $TRAINER ==="
#nnUNetv2_train "$DATASET_ID" 3d_fullres 0 -tr "$TRAINER"
nnUNet_n_proc_DA=0 nnUNetv2_train "$DATASET_ID" 3d_fullres 0 -tr "$TRAINER"

MODEL="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__3d_fullres"
[ -d "$MODEL" ] || { echo "model directory not found: $MODEL" >&2; exit 1; }

echo "=== [4/5] inference on imagesTs ==="
python "$HERE/AutoMSC-Baseline/segcls_ensemble_infer.py" \
  -i "$nnUNet_raw/$DATASET_NAME/imagesTs" \
  -o "$WORK/predictions_raw/$DATASET_NAME" \
  --model_path "$MODEL" \
  --fold 0 \
  --checkpoint checkpoint_best.pth \
  --device cuda \
  --cls_mode mean

echo "=== [5/5] packaging and verification ==="
python "$HERE/build_submission.py" \
  "$nnUNet_raw/$DATASET_NAME" \
  "$WORK/predictions_raw/$DATASET_NAME" \
  "$WORK/submission"

python "$HERE/verify_masks.py" \
  "$WORK/submission/predictions/${DATASET_NAME}_prediction" \
  "$nnUNet_raw/$DATASET_NAME/imagesTs"

echo "=== done: $DATASET_NAME ==="
echo "submission tree: $WORK/submission/predictions/${DATASET_NAME}_prediction"
