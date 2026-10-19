#!/usr/bin/env bash
set -euo pipefail

# Defaults use this checkout and the activated Python environment.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${RFWORLD_DATA_ROOT:-$ROOT/dataset}"
EXP_ROOT="${RFWORLD_EXP_ROOT:-$ROOT/exp}"
export RFWORLD_DATA_ROOT="$DATA_ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export NAVSIM_DEVKIT_ROOT="$ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
unset PYTORCH_CUDA_ALLOC_CONF

"$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=29800 \
  "$ROOT/navsim/planning/script/run_rfworld_training.py" \
  agent=RFWorld_agent \
  agent.config._target_=navsim.agents.RFWorld.configs.default.RFWorldConfig \
  cache_path="$EXP_ROOT/training_cache" \
  experiment_name=RFWorld/default \
  scene_filter=navtrain \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=50 \
  split=trainval \
  "$@"
