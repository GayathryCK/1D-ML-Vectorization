#!/usr/bin/env bash
set -e
source /opt/pytorch/bin/activate
PROJ=~/tda_entropy_project
LOG="$PROJ/ripsnet_sonn_entropy.log"

echo "====== RipsNet SONN entropy training started: $(date) ======" | tee "$LOG"

python "$PROJ/train_ripsnet_entropy.py" \
  --data_root  "$PROJ/SONNDataSet/SONN" \
  --train_csv  "$PROJ/SONN_entropy/ground_truths/sonn_train_entropy.csv" \
  --test_csv   "$PROJ/SONN_entropy/ground_truths/sonn_test_entropy.csv" \
  --out_dir    "$PROJ/SONN_entropy/ripsnet_out" \
  --epochs     100 \
  --batch_size 32 \
  --num_points 1024 \
  --lr         5e-4 \
  --seed       42 \
  2>&1 | tee -a "$LOG"

echo "====== Syncing results to S3: $(date) ======" | tee -a "$LOG"
/usr/local/bin/aws s3 sync "$PROJ/SONN_entropy/ripsnet_out" \
  s3://gre-tda-entropy/SONN_entropy/ripsnet_out \
  --exclude "._*" 2>&1 | tee -a "$LOG"

echo "====== Done: $(date) ======" | tee -a "$LOG"
