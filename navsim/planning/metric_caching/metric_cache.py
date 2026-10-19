from __future__ import annotations

import lzma
import os
import pickle
import uuid
from dataclasses import dataclass

from typing import List
from pathlib import Path
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.ego_state import EgoState

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMDrivableMap,
)

from nuplan.common.utils.io_utils import save_buffer


@dataclass
class MetricCache:

    file_path: Path
    trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap

    def dump(self) -> None:
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path = self.file_path.with_name(
            f".{self.file_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            save_buffer(temporary_path, lzma.compress(pickle_object, preset=0))
            os.replace(temporary_path, self.file_path)
        finally:
            temporary_path.unlink(missing_ok=True)
