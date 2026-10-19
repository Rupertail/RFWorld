#!/usr/bin/env bash
set -euo pipefail

# Defaults use this checkout and the activated Python environment.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${RFWORLD_DATA_ROOT:-$ROOT/dataset}"
EXP_ROOT="${RFWORLD_EXP_ROOT:-$ROOT/exp}"
export RFWORLD_DATA_ROOT="$DATA_ROOT"
RAY_WORKERS="${RAY_WORKERS:-4}"

CHECKPOINT="${CHECKPOINT:-$DATA_ROOT/checkpoints/rfworld_tf_resnet34_pdms-905.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$EXP_ROOT/metric_cache_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP_ROOT/eval/RFWorld/rfworld_tf_resnet34_pdms-905}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$METRIC_CACHE_PATH/metadata" ]]; then
  echo "Test metric cache does not exist: $METRIC_CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export NAVSIM_DEVKIT_ROOT="$ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" "$ROOT/navsim/planning/script/run_pdm_score.py" \
  agent=RFWorld_agent \
  "agent.checkpoint_path=\"$CHECKPOINT\"" \
  agent.config._target_=navsim.agents.RFWorld.configs.default.RFWorldConfig \
  experiment_name=eval/RFWorld \
  split=test \
  scene_filter=navtest \
  metric_cache_path="$METRIC_CACHE_PATH" \
  worker=ray_distributed_no_torch \
  worker.threads_per_node="$RAY_WORKERS" \
  "output_dir=\"$OUTPUT_DIR\"" \
  "$@"
