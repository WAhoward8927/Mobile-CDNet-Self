#!/usr/bin/env bash
set -euo pipefail
cd /content/Mobile-CDNet
export PYTHONPATH=/content/Mobile-CDNet
export MOBILE_CDNET_DATA_ROOT=/content/BCDD
INIT=/content/drive/MyDrive/Mobile-CDNet/outputs/BCDD/MatchedEpoch10_HalfFeatureFusion_20260926T043342Z/initial_author_state.pth
OUT=/content/drive/MyDrive/Mobile-CDNet/outputs/BCDD
python -u tools/train.py --arch half_zero --initial_author_state "$INIT" --file_root BCDD --savedir "$OUT/BCDD_Improve_V1_HalfZero_Full" --batch_size 16 --lr 5e-4 --max_steps 76000 --lr_mode step --step_loss 100 --num_workers 2
python -u tools/train.py --arch author --initial_author_state "$INIT" --file_root BCDD --savedir "$OUT/BCDD_Improve_V1_Author_Matched_Full" --batch_size 16 --lr 5e-4 --max_steps 76000 --lr_mode step --step_loss 100 --num_workers 2
