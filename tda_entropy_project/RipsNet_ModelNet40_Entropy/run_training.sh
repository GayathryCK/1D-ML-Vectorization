#!/usr/bin/env bash
set -e
source /opt/pytorch/bin/activate
PROJ=~/tda_entropy_project
LOG="$PROJ/ripsnet_train.log"

echo "====== RipsNet training started: $(date) ======" | tee "$LOG"

python "$PROJ/train_ripsnet_entropy.py" \
  --data_root  "$PROJ/ModelNet40" \
  --train_csv  "$PROJ/ground_truths/mn40_train_entropy.csv" \
  --test_csv   "$PROJ/ground_truths/mn40_test_entropy.csv" \
  --out_dir    "$PROJ/ripsnet_entropy_out" \
  --epochs     100 \
  --batch_size 32 \
  --num_points 1024 \
  --lr         5e-4 \
  --seed       42 \
  2>&1 | tee -a "$LOG"

echo "====== Syncing results to S3: $(date) ======" | tee -a "$LOG"
/usr/local/bin/aws s3 sync "$PROJ/ripsnet_entropy_out" \
  s3://gre-tda-entropy/ripsnet_entropy_out \
  --exclude "._*" 2>&1 | tee -a "$LOG"

echo "====== Done: $(date) ======" | tee -a "$LOG"
