#!/usr/bin/env bash
set -euo pipefail

# Defaults use this checkout and the activated Python environment.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${RFWORLD_DATA_ROOT:-$ROOT/dataset}"
EXP_ROOT="${RFWORLD_EXP_ROOT:-$ROOT/exp}"
export RFWORLD_DATA_ROOT="$DATA_ROOT"
RAY_WORKERS="${RAY_WORKERS:-4}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export NAVSIM_DEVKIT_ROOT="$ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" "$ROOT/scripts/planning/gen_training_pdms.py" \
  experiment_name=planning/anchor_pdms \
  metric_cache_path="$EXP_ROOT/metric_cache" \
  split=trainval \
  scene_filter=navtrain \
  +anchor_path="$DATA_ROOT/planning/trajectory_anchors_1024.npy" \
  +output_path="$DATA_ROOT/planning/formatted_pdm_score_1024.npy" \
  +shard_dir="$EXP_ROOT/planning/anchor_pdms_shards" \
  +shard_size=16 \
  worker.threads_per_node="$RAY_WORKERS" \
  "$@"
