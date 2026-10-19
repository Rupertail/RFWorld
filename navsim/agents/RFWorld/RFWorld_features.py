from typing import Any, List, Dict, Union
import torch
import numpy as np
from torchvision import transforms

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import BoundingBoxIndex, LidarIndex

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import timm, cv2


class RFWorldFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
            self,
            slice_indices=[0, 1, 2, 3],
            config=None,
    ):
        self.slice_indices = slice_indices
        self._config = config

    def get_unique_name(self) -> str:
        return "RFWorld_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)

        features["lidar_feature"] = self._get_lidar_feature(agent_input, frame_idx=-1)

        lidar_seq_list = []
        for i in range(len(agent_input.lidars)):
            lidar_seq_list.append(self._get_lidar_feature(agent_input, frame_idx=i))
        features["lidar_feature_seq"] = torch.stack(lidar_seq_list, dim=0)

        pose_seq_list = []
        for i in range(len(agent_input.ego_statuses)):
            pose = agent_input.ego_statuses[i].ego_pose
            pose_seq_list.append(torch.tensor(pose, dtype=torch.float32))
        features["ego_poses_seq"] = torch.stack(pose_seq_list, dim=0)

        features["status_feature"] = self._get_ego_status_feature(
            agent_input.ego_statuses[-1]
        )

        features["ego_status_seq"] = self._get_ego_status_seq(agent_input)

        return features

    def _get_ego_status_feature(self, ego_status) -> torch.Tensor:
        """
        driving_command: [4]
        ego_velocity: [2]
        ego_acceleration: [2]
        return: [8]
        """
        return torch.concatenate(
            [
                torch.tensor(ego_status.driving_command, dtype=torch.float32),
                torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
                torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
            ],
            dim=0,
        )

    def _get_ego_status_seq(self, agent_input: AgentInput) -> torch.Tensor:
        """
        return: [T, 8]
        """
        ego_status_seq = []
        for i in range(len(agent_input.ego_statuses)):
            ego_status_seq.append(
                self._get_ego_status_feature(agent_input.ego_statuses[i])
            )

        return torch.stack(ego_status_seq, dim=0)

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput, frame_idx: int = -1) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :param frame_idx: index of the lidar frame to process (default is -1, the current frame)
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[frame_idx].lidar_pc[LidarIndex.POSITION].T

        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)