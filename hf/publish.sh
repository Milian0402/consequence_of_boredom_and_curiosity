#!/bin/sh
set -eu

NAMESPACE=${HF_NAMESPACE:-Milian1}
DATASET_REPO=${HF_DATASET_REPO:-"$NAMESPACE/apple-silicon-sgemm-benchmarks"}
SPACE_REPO=${HF_SPACE_REPO:-"$NAMESPACE/cob-sgemm-dashboard"}
VISIBILITY=${HF_VISIBILITY:-private}

if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI not found. Install it with: curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    exit 1
fi

PRIVATE_FLAG=
if [ "$VISIBILITY" = "private" ]; then
    PRIVATE_FLAG=--private
fi

python3 tools/export_hf_dataset.py
hf auth whoami >/dev/null

hf repos create "$DATASET_REPO" --type dataset --exist-ok $PRIVATE_FLAG
hf upload "$DATASET_REPO" hf/sgemm-benchmarks-dataset . \
    --type dataset \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --delete "__pycache__/*" \
    --delete "*.pyc" \
    --commit-message "Publish Apple Silicon SGEMM benchmark dataset"

hf repos create "$SPACE_REPO" --type space --space-sdk static --exist-ok $PRIVATE_FLAG
hf upload "$SPACE_REPO" hf/sgemm-dashboard-space . \
    --type space \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --delete "__pycache__/*" \
    --delete "*.pyc" \
    --delete "app.py" \
    --delete "requirements.txt" \
    --commit-message "Publish SGEMM benchmark dashboard"

echo "Dataset: https://huggingface.co/datasets/$DATASET_REPO"
echo "Space:   https://huggingface.co/spaces/$SPACE_REPO"
