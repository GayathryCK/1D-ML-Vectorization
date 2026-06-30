#!/usr/bin/env bash
set -e
source /opt/pytorch/bin/activate
PROJ=~/tda_entropy_project
LOG="$PROJ/ripsnet_all_amplitudes.log"

AMPLITUDES=(betti heat landscape persistence_image silhouette wasserstein)

echo "====== RipsNet multi-amplitude training started: $(date) ======" | tee "$LOG"

for AMP in "${AMPLITUDES[@]}"; do
  echo "" | tee -a "$LOG"
  echo "############################################################" | tee -a "$LOG"
  echo "  Starting: amplitude_${AMP}  [$(date)]" | tee -a "$LOG"
  echo "############################################################" | tee -a "$LOG"

  TRAIN_CSV="$PROJ/amplitudes/amplitude_${AMP}/ground_truths/mn40_train_amplitude_${AMP}.csv"
  TEST_CSV="$PROJ/amplitudes/amplitude_${AMP}/ground_truths/mn40_test_amplitude_${AMP}.csv"
  OUT_DIR="$PROJ/amplitudes/amplitude_${AMP}/ripsnet_out"

  python "$PROJ/train_ripsnet_entropy.py" \
    --data_root  "$PROJ/ModelNet40" \
    --train_csv  "$TRAIN_CSV" \
    --test_csv   "$TEST_CSV" \
    --out_dir    "$OUT_DIR" \
    --epochs     100 \
    --batch_size 32 \
    --num_points 1024 \
    --lr         5e-4 \
    --seed       42 \
    2>&1 | tee -a "$LOG"

  echo "  Syncing amplitude_${AMP} results to S3 [$(date)]" | tee -a "$LOG"
  /usr/local/bin/aws s3 sync "$OUT_DIR" \
    "s3://gre-tda-entropy/amplitudes/amplitude_${AMP}/ripsnet_out" \
    --exclude "._*" 2>&1 | tee -a "$LOG"

  echo "  Done: amplitude_${AMP} [$(date)]" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "====== ALL AMPLITUDES COMPLETE: $(date) ======" | tee -a "$LOG"
