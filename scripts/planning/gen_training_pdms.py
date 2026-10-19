import ctypes
import gc
import logging
import lzma
import os
import pickle
import uuid

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import numpy as np
import pandas as pd

from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task
from nuplan.planning.utils.multithreading.worker_utils import chunk_list

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.evaluate.pdm_score import pdm_score_multi_trajs
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


logger = logging.getLogger(__name__)

NUM_CLUSTERS = 1024
NUM_HORIZONS = 4
SCORE_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
)

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "navsim/planning/script/config/pdm_scoring")
CONFIG_NAME = "default_run_pdm_score"


def _atomic_save_numpy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as file:
            np.save(file, value, allow_pickle=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_score_value(token: str, value: Any, source: Path) -> None:
    if not isinstance(value, dict) or set(value) != {"valid", "trajectory_scores"}:
        raise RuntimeError(f"Invalid PDMS entry for token {token} in {source}")
    trajectory_scores = value["trajectory_scores"]
    if not isinstance(trajectory_scores, list) or len(trajectory_scores) != 1:
        raise RuntimeError(f"Invalid trajectory_scores for token {token} in {source}")
    score_dict = trajectory_scores[0]
    if not isinstance(score_dict, dict) or set(score_dict) != set(SCORE_KEYS):
        raise RuntimeError(f"Invalid PDMS score fields for token {token} in {source}")
    for key in SCORE_KEYS:
        values = np.asarray(score_dict[key])
        if values.shape != (NUM_CLUSTERS,):
            raise RuntimeError(
                f"Invalid {key} shape for token {token} in {source}: {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {key} values for token {token} in {source}")


def _load_shard(path: Path, expected_tokens: Sequence[str]) -> Dict[str, Any]:
    try:
        shard = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDMS shard {path}") from exc
    if not isinstance(shard, dict):
        raise RuntimeError(f"PDMS shard is not a dictionary: {path}")
    if set(shard) != set(expected_tokens) or len(shard) != len(expected_tokens):
        raise RuntimeError(
            f"PDMS shard token mismatch in {path}: "
            f"expected {len(expected_tokens)}, found {len(shard)}"
        )
    for token in expected_tokens:
        _validate_score_value(token, shard[token], path)
    return shard


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    metric_cache_path = Path(cfg.metric_cache_path)
    if not metric_cache_path.is_dir():
        raise FileNotFoundError(f"Metric cache directory does not exist: {metric_cache_path}")
    metric_cache_loader = MetricCacheLoader(metric_cache_path)

    scene_tokens = list(scene_loader.tokens)
    scene_token_set = set(scene_tokens)
    cache_tokens = set(metric_cache_loader.tokens)
    missing_tokens = scene_token_set - cache_tokens
    if missing_tokens:
        first_missing = sorted(missing_tokens)[0]
        raise RuntimeError(
            f"Metric cache is missing {len(missing_tokens)} of {len(scene_tokens)} required "
            f"scene tokens. First missing token: {first_missing}"
        )
    if not scene_tokens:
        raise RuntimeError("Scene filter selected no tokens; refusing to write an empty PDMS file")

    unused_tokens = cache_tokens - scene_token_set
    if unused_tokens:
        logger.info("Ignoring %d metric-cache tokens outside the selected scene filter", len(unused_tokens))

    anchor_path = Path(cfg.anchor_path)
    load_predefined_trajectories(anchor_path)

    shard_size = int(cfg.shard_size)
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    shard_dir = Path(cfg.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_specs: List[Dict[str, Any]] = []
    pending_specs: List[Dict[str, Any]] = []
    reused_tokens = 0
    for shard_index, start in enumerate(range(0, len(scene_tokens), shard_size)):
        tokens = scene_tokens[start : start + shard_size]
        shard_path = shard_dir / f"shard_{shard_index:06d}.npy"
        spec = {"tokens": tokens, "shard_path": str(shard_path)}
        shard_specs.append(spec)
        if shard_path.exists():
            _load_shard(shard_path, tokens)
            reused_tokens += len(tokens)
        else:
            pending_specs.append(spec)

    logger.info(
        "Anchor PDMS coverage: %d reused, %d missing across %d pending shards",
        reused_tokens,
        len(scene_tokens) - reused_tokens,
        len(pending_specs),
    )

    del scene_loader
    del metric_cache_loader
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass

    try:
        if pending_specs:
            object_chunks = chunk_list(
                pending_specs, min(worker.number_of_threads, len(pending_specs))
            )
            scattered_paths = worker.map(
                Task(fn=run_pdm_score, num_cpus=1),
                object_chunks,
                cfg,
            )
            completed_paths = [
                path for worker_paths in scattered_paths for path in worker_paths
            ]
            if len(completed_paths) != len(pending_specs):
                raise RuntimeError(
                    f"PDMS shard completion mismatch: expected {len(pending_specs)}, "
                    f"got {len(completed_paths)}"
                )
    finally:
        shutdown = getattr(worker, "shutdown", None)
        if shutdown is not None:
            shutdown()

    for spec in shard_specs:
        _load_shard(Path(spec["shard_path"]), spec["tokens"])

    format_and_save_scores(
        shard_specs=shard_specs,
        output_path=Path(cfg.output_path),
        output_dir=Path(cfg.output_dir),
    )


def format_and_save_scores(
    shard_specs: Sequence[Dict[str, Any]],
    output_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    score_dict: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    for spec in tqdm(shard_specs, desc="Merging PDMS shards"):
        shard_path = Path(spec["shard_path"])
        shard = _load_shard(shard_path, spec["tokens"])
        for token in spec["tokens"]:
            value = shard[token]
            score_dict[token] = value
            anchor_scores = np.asarray(value["trajectory_scores"][0]["score"])
            summary_rows.append(
                {
                    "token": token,
                    "valid": bool(value["valid"]),
                    "mean_anchor_score": float(anchor_scores.mean()),
                    "max_anchor_score": float(anchor_scores.max()),
                    "min_anchor_score": float(anchor_scores.min()),
                }
            )

    _atomic_save_numpy(output_path, score_dict)
    logger.info("Saved anchor PDM scores to %s", output_path)

    summary_df = pd.DataFrame(summary_rows)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    summary_path = output_dir / f"{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(
        "Finished anchor PDM scoring: %d scenarios, mean anchor score %.6f. Summary: %s",
        len(summary_df),
        summary_df["mean_anchor_score"].mean(),
        summary_path,
    )
    return summary_df


def run_pdm_score(
    shard_specs: List[Dict[str, Any]],
    cfg: DictConfig,
) -> List[str]:
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)

    proposal_sampling = simulator.proposal_sampling
    proposal_sampling.time_horizon = NUM_HORIZONS
    proposal_sampling.num_poses = int(NUM_HORIZONS * 10)
    scorer.proposal_sampling = proposal_sampling
    simulator.proposal_sampling = proposal_sampling

    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RuntimeError("Simulator and scorer proposal sampling must be identical")

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    predefined_trajectories = load_predefined_trajectories(Path(cfg.anchor_path))

    completed_paths: List[str] = []
    for spec in shard_specs:
        shard_path = Path(spec["shard_path"])
        tokens = spec["tokens"]
        if shard_path.exists():
            _load_shard(shard_path, tokens)
            completed_paths.append(str(shard_path))
            continue

        score_dict: Dict[str, Any] = {}
        for token in tqdm(tokens, desc=f"Scoring {shard_path.stem}"):
            metric_cache_file = metric_cache_loader.metric_cache_paths.get(token)
            if metric_cache_file is None:
                raise RuntimeError(f"Metric cache path missing for token {token}")
            with lzma.open(metric_cache_file, "rb") as file:
                metric_cache: MetricCache = pickle.load(file)

            pdm_result: PDMResults = pdm_score_multi_trajs(
                metric_cache=metric_cache,
                model_trajectory_list=predefined_trajectories,
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            value = {
                "valid": True,
                "trajectory_scores": [asdict(pdm_result)],
            }
            _validate_score_value(token, value, shard_path)
            score_dict[token] = value

        _atomic_save_numpy(shard_path, score_dict)
        _load_shard(shard_path, tokens)
        completed_paths.append(str(shard_path))

    return completed_paths


def load_predefined_trajectories(anchor_path: Path) -> np.ndarray:
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Trajectory anchor file does not exist: {anchor_path}")

    trajectories = np.load(anchor_path)
    expected_shape = (NUM_CLUSTERS, 8, 3)
    if trajectories.shape != expected_shape:
        raise ValueError(
            f"Expected trajectory anchors with shape {expected_shape}, got {trajectories.shape}: "
            f"{anchor_path}"
        )
    if not np.isfinite(trajectories).all():
        raise ValueError(f"Trajectory anchor file contains non-finite values: {anchor_path}")
    return trajectories


if __name__ == "__main__":
    main()
