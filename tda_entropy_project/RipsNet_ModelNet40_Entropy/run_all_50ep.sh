#!/usr/bin/env bash
set -e
source /opt/pytorch/bin/activate
PROJ=~/tda_entropy_project
LOG="$PROJ/ripsnet_all_50ep.log"
EPOCHS=50
BATCH=16

echo "====== RipsNet 50ep/bs16 full run started: $(date) ======" | tee "$LOG"

# ── MN40 amplitudes (all 7) ───────────────────────────────────────────────────
MN40_AMPS=(betti bottleneck heat landscape persistence_image silhouette wasserstein)

for AMP in "${MN40_AMPS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "#### MN40 amplitude_${AMP} [$(date)] ####" | tee -a "$LOG"
  python "$PROJ/train_ripsnet_entropy.py" \
    --data_root  "$PROJ/ModelNet40" \
    --train_csv  "$PROJ/amplitudes/amplitude_${AMP}/ground_truths/mn40_train_amplitude_${AMP}.csv" \
    --test_csv   "$PROJ/amplitudes/amplitude_${AMP}/ground_truths/mn40_test_amplitude_${AMP}.csv" \
    --out_dir    "$PROJ/amplitudes/amplitude_${AMP}/ripsnet_out_50ep" \
    --epochs     $EPOCHS \
    --batch_size $BATCH \
    --num_points 1024 \
    --lr         5e-4 \
    --seed       42 \
    2>&1 | tee -a "$LOG"
  echo "  Done: MN40 amplitude_${AMP} [$(date)]" | tee -a "$LOG"
done

# ── SONN amplitudes (all 7) ───────────────────────────────────────────────────
SONN_AMPS=(betti bottleneck heat landscape persistence_image silhouette wasserstein)

for AMP in "${SONN_AMPS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "#### SONN amplitude_${AMP} [$(date)] ####" | tee -a "$LOG"
  python "$PROJ/train_ripsnet_entropy.py" \
    --data_root  "$PROJ/SONNDataSet/SONN" \
    --train_csv  "$PROJ/SONN_amplitudes/amplitude_${AMP}/ground_truths/sonn_train_amplitude_${AMP}.csv" \
    --test_csv   "$PROJ/SONN_amplitudes/amplitude_${AMP}/ground_truths/sonn_test_amplitude_${AMP}.csv" \
    --out_dir    "$PROJ/SONN_amplitudes/amplitude_${AMP}/ripsnet_out_50ep" \
    --epochs     $EPOCHS \
    --batch_size $BATCH \
    --num_points 1024 \
    --lr         5e-4 \
    --seed       42 \
    2>&1 | tee -a "$LOG"
  echo "  Done: SONN amplitude_${AMP} [$(date)]" | tee -a "$LOG"
done

# ── SONN entropy ──────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "#### SONN entropy [$(date)] ####" | tee -a "$LOG"
python "$PROJ/train_ripsnet_entropy.py" \
  --data_root  "$PROJ/SONNDataSet/SONN" \
  --train_csv  "$PROJ/SONN_entropy/ground_truths/sonn_train_entropy.csv" \
  --test_csv   "$PROJ/SONN_entropy/ground_truths/sonn_test_entropy.csv" \
  --out_dir    "$PROJ/SONN_entropy/ripsnet_out_50ep" \
  --epochs     $EPOCHS \
  --batch_size $BATCH \
  --num_points 1024 \
  --lr         5e-4 \
  --seed       42 \
  2>&1 | tee -a "$LOG"
echo "  Done: SONN entropy [$(date)]" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "====== ALL 15 RUNS COMPLETE: $(date) ======" | tee -a "$LOG"
