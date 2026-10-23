#!/bin/bash
# Fastest usable demo on S100: scene-only (no face), HW MJPEG decode, W2/INF2.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Defaults for max speed (override by exporting before start)
export RAC_HWDEC="${RAC_HWDEC:-1}"
export RAC_WORKERS="${RAC_WORKERS:-2}"
export RAC_INFLIGHT="${RAC_INFLIGHT:-2}"
export RAC_PRESCALE="${RAC_PRESCALE:-1}"
export RAC_FACE="${RAC_FACE:-0}"   # 0=scene path ~14fps; 1=face hybrid (slow without structure HBM)
export DISPLAY="${DISPLAY:-:0}"
echo "start RAC_HWDEC=$RAC_HWDEC RAC_WORKERS=$RAC_WORKERS RAC_INFLIGHT=$RAC_INFLIGHT RAC_FACE=$RAC_FACE"
exec python3 local_display.py
