from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from navsim.common.enums import StateSE2Index
from navsim.agents.RFWorld.RFWorld_targets import BoundingBox2DIndex
from navsim.agents.RFWorld.transfuser_backbone import TransfuserBackbone

import os
import math
from datetime import datetime


class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Backbone, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x)


class RFWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # --- Constants & Dimensions ---
        STATUS_ENCODING_INPUT_DIM = 4 + 2 + 2
        hidden_dim = 256
        self.hidden_dim = hidden_dim

        SCORE_HEAD_HIDDEN_DIM = 128
        SCORE_HEAD_OUTPUT_DIM = 1
        NUM_SCORE_HEADS = 5  # NC, DAC, EP, TTC, Comfort

        self.x_max_meters = 32.0
        self.y_max_meters = 32.0
        self.x_min_meters = -32.0
        self.y_min_meters = -32.0

        # --- Backbone & Encoder ---
        self._backbone = TransfuserBackbone(config)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(STATUS_ENCODING_INPUT_DIM, config.tf_d_model)

        # Reward-side ego status encoders.
        # - current_status_reward_encoder: encodes the instantaneous command/velocity/acceleration.
        # - ego_status_history_encoder: encodes historical ego dynamics from ego_status_seq.
        # Both branches output hidden_dim so they can be added/concatenated with reward features safely.
        self.current_status_reward_encoder = nn.Sequential(
            nn.Linear(STATUS_ENCODING_INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ego_status_history_encoder = nn.GRU(
            input_size=STATUS_ENCODING_INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.ego_status_history_norm = nn.LayerNorm(hidden_dim)
        self.reward_status_context_fuser = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reward_embedding_status_fuser = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 64
        # self._keyval_embedding = nn.Embedding(num_keyval, config.tf_d_model)

        # Trajectory Anchors Setup
        cluster_file = config.cluster_file_path
        self.trajectory_anchors = torch.nn.Parameter(
            torch.tensor(np.load(cluster_file)), requires_grad=False
        )
        CLUSTER_CENTERS_FEATURE_DIM = 24
        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(CLUSTER_CENTERS_FEATURE_DIM, SCORE_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, hidden_dim),
        )

        # Cluster Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ================= Latent World Model (Spatio-Temporal Generator) =================
        self.spatial_position_embedding = nn.Embedding(num_keyval, hidden_dim) # 8x8 = 64 locations

        wm_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.latent_world_model = nn.TransformerEncoder(wm_layer, num_layers=2)

        # --- Spatio-Temporal Sampling & Attention ---
        self.num_fut_timestep = config.trajectory_sampling.num_poses
        self.total_readout_steps = 1 + self.num_fut_timestep  # current + future
        self.use_future_map_loss = getattr(config, "use_future_map_loss", True)

        future_steps = getattr(config, "future_supervision_steps", None)
        if future_steps is None or len(future_steps) == 0:
            self.future_supervision_indices = list(range(self.num_fut_timestep))  # future-only 0-base
        else:
            self.future_supervision_indices = self._validate_future_supervision_indices(list(future_steps))

        self.future_supervision_sampling = getattr(config, "future_supervision_sampling", "fixed")
        self.random_future_supervision = bool(getattr(config, "random_future_supervision", False)) or \
            self.future_supervision_sampling in ["random", "random_group", "random_groups"]
        self.future_supervision_sample_groups = getattr(
            config,
            "future_supervision_sample_groups",
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )
        self.future_supervision_num_per_group = getattr(
            config,
            "future_supervision_num_per_group",
            [2, 2],
        )

        self.readout_selected_indices = [0] + [idx + 1 for idx in self.future_supervision_indices]

        # --- Reward Heads (Re-designed) ---
        self.pooling_type = config.pooling_type if hasattr(config, 'pooling_type') else 'max'
        self.reward_encoder = TrajectoryDeformableRewardEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_points=4,
            pooling_type=self.pooling_type
        )

        self.im_reward_head = nn.Linear(hidden_dim, 1)

        self.sim_reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
            ) for _ in range(NUM_SCORE_HEADS)
        ])

        self.reward_weights = config.reward_weights if hasattr(config, 'reward_weights') else [0.1, 0.5, 0.5, 1.0]
        self.sim_weights = config.sim_weights if hasattr(config, 'sim_weights') else [5.0, 2.0, 5.0]

        self.inference_num_trajectory_candidates = getattr(
            config,
            "inference_num_trajectory_candidates",
            getattr(config, "inference_num_trajectory_anchors", None),
        )
        self.inference_anchor_selection = getattr(config, "inference_anchor_selection", "uniform")
        self.inference_candidate_augmentation = getattr(config, "inference_candidate_augmentation", "scale_xy")
        self.inference_anchor_longitudinal_scale_delta = float(
            getattr(config, "inference_anchor_longitudinal_scale_delta", 0.15)
        )
        self.inference_anchor_lateral_scale_delta = float(
            getattr(config, "inference_anchor_lateral_scale_delta", 0.05)
        )
        self.inference_anchor_recompute_heading = bool(
            getattr(config, "inference_anchor_recompute_heading", False)
        )
        self.inference_anchor_noise_std = float(getattr(config, "inference_anchor_noise_std", 0.0))

        # --- Auxiliary Modules (Decoders / Losses) ---
        # Map Decoder
        self.use_map_loss = config.use_map_loss if hasattr(config, 'use_map_loss') else False
        if self.use_map_loss:
            self.bev_upsample_head = BEVUpsampleHead(config)
            self.bev_semantic_head = nn.Sequential(
                nn.Conv2d(config.bev_features_channels, config.bev_features_channels, (3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.bev_features_channels, config.num_bev_classes, (1, 1)),
                nn.Upsample(size=(config.lidar_resolution_height // 2, config.lidar_resolution_width), mode="bilinear",
                            align_corners=False),
            )
        self._bev_upscale = nn.Conv2d(config.tf_d_model, 512, kernel_size=1)

        if self.use_future_map_loss or self.use_map_loss:
            self.future_bev_readout = FutureBEVSEReadout(
                hidden_dim=hidden_dim,
                total_future_steps=self.total_readout_steps,
                selected_indices=self.readout_selected_indices,
                reduction=4,
            )

        # Agent Decoder
        self.use_agent_loss = config.use_agent_loss if hasattr(config, 'use_agent_loss') else True
        if self.use_agent_loss:
            self.agent_query_embedding = nn.Embedding(config.num_bounding_boxes, hidden_dim)
            self.agent_head = AgentHead(num_agents=config.num_bounding_boxes, d_ffn=config.tf_d_ffn,
                                        d_model=config.tf_d_model)
            agent_dec_layer = nn.TransformerDecoderLayer(d_model=config.tf_d_model, nhead=config.tf_num_head,
                                                         dim_feedforward=config.tf_d_ffn, batch_first=True)
            self.agent_tf_decoder = nn.TransformerDecoder(agent_dec_layer, config.tf_num_layers)

        # Offset Decoder
        offset_dec_layer = nn.TransformerDecoderLayer(d_model=config.tf_d_model, nhead=config.tf_num_head,
                                                      dim_feedforward=config.tf_d_ffn, batch_first=True)
        self.offset_tf_decoder = nn.TransformerDecoder(offset_dec_layer, config.tf_num_layers)
        self.offset_head = TrajectoryOffsetHead(num_poses=config.trajectory_sampling.num_poses, d_ffn=config.tf_d_ffn,
                                                d_model=config.tf_d_model)
        self.offset_score_head = nn.Sequential(
            nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
        )

    # ================= World Model Logic =================
    def _validate_future_supervision_indices(self, indices):
        indices = [int(idx) for idx in indices]
        if len(indices) == 0:
            return indices
        if min(indices) < 0 or max(indices) >= self.num_fut_timestep:
            raise ValueError(
                f"future supervision indices must be in [0, {self.num_fut_timestep - 1}], got {indices}"
            )
        return indices

    def _sample_future_supervision_indices(self, device=None):
        """
        Return future-only 0-based indices used for future semantic-map supervision.
        - fixed mode: use self.future_supervision_indices, preserving the legacy behavior.
        - random_group mode: sample k frames per configured group every forward step.
        """
        if not self.random_future_supervision:
            return list(self.future_supervision_indices)

        groups = [list(group) for group in self.future_supervision_sample_groups]
        nums = list(self.future_supervision_num_per_group)
        if len(groups) != len(nums):
            raise ValueError(
                "future_supervision_sample_groups and future_supervision_num_per_group "
                f"must have the same length, got {len(groups)} and {len(nums)}"
            )

        selected = []
        for group, num in zip(groups, nums):
            valid_group = self._validate_future_supervision_indices(group)
            k = int(num)
            if k < 0:
                raise ValueError(f"future supervision sample count must be non-negative, got {k}")
            if k > len(valid_group):
                raise ValueError(
                    f"cannot sample {k} frames from group {valid_group}; group only has {len(valid_group)} frames"
                )
            if k == 0:
                continue
            perm = torch.randperm(len(valid_group), device=device)[:k].detach().cpu().tolist()
            sampled = sorted(valid_group[i] for i in perm)
            selected.extend(sampled)

        return selected

    def _get_runtime_trajectory_anchors(self, batch_size, device, dtype, use_inference_candidates=False):
        """
        Build the trajectory anchors used by the current forward pass.

        Training keeps the original cluster anchors so the loss target and tensor
        shapes remain unchanged. In evaluation, the candidate count can be
        changed with inference_num_trajectory_candidates.
        """
        base_anchors = self.trajectory_anchors.to(device=device, dtype=dtype)
        base_anchors = base_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

        if not use_inference_candidates:
            return base_anchors

        target_num = self.inference_num_trajectory_candidates
        if target_num is None:
            return base_anchors
        target_num = int(target_num)
        if target_num <= 0:
            return base_anchors

        base_num = base_anchors.shape[1]
        if target_num == base_num:
            return base_anchors
        if target_num < base_num:
            return self._select_inference_anchor_subset(base_anchors, target_num)
        return self._expand_inference_anchors(base_anchors, target_num)

    def _select_inference_anchor_subset(self, anchors, target_num):
        """
        Select a smaller inference candidate set from the trained anchor bank.

        Supported modes:
        - uniform: evenly spaced anchors, stable and deterministic.
        - first: first target_num anchors.
        - progress: largest final longitudinal displacement x.
        - random: random subset every inference call; mainly for oracle/debug.
        """
        base_num = anchors.shape[1]
        device = anchors.device
        mode = str(self.inference_anchor_selection).lower()

        if target_num > base_num:
            raise ValueError(f"target_num={target_num} exceeds base_num={base_num} for subset selection")

        if mode == "first":
            idx = torch.arange(target_num, device=device, dtype=torch.long)
        elif mode in ["progress", "top_progress", "ep", "longitudinal"]:
            progress = anchors[0, :, -1, 0]
            idx = torch.topk(progress, k=target_num, largest=True, sorted=True).indices
        elif mode in ["random", "rand"]:
            idx = torch.randperm(base_num, device=device)[:target_num]
        else:
            # uniform is the default. For target_num <= base_num, rounded linspace
            # gives a deterministic coverage over the original anchor bank.
            idx = torch.linspace(0, base_num - 1, steps=target_num, device=device).round().long()

        return anchors.index_select(dim=1, index=idx)

    def _expand_inference_anchors(self, anchors, target_num):
        """
        Expand inference candidates without changing any trained module shape.

        The extra anchors are deterministic scaled variants of the trained anchor
        bank by default. This gives the offset/reward heads more candidate modes
        at inference while preserving the original anchors as the first block.
        """
        batch_size, base_num, _, _ = anchors.shape
        repeat_factor = int(math.ceil(target_num / base_num))
        mode = str(self.inference_candidate_augmentation).lower()

        if mode in ["none", "repeat", "identity"]:
            expanded = anchors.repeat(1, repeat_factor, 1, 1)
            return expanded[:, :target_num].contiguous()

        long_deltas = self._make_symmetric_deltas(
            repeat_factor,
            max_abs=self.inference_anchor_longitudinal_scale_delta,
            device=anchors.device,
            dtype=anchors.dtype,
        )
        lat_deltas = self._make_symmetric_deltas(
            repeat_factor,
            max_abs=self.inference_anchor_lateral_scale_delta,
            device=anchors.device,
            dtype=anchors.dtype,
        )

        expanded_blocks = []
        for repeat_idx in range(repeat_factor):
            aug = anchors.clone()

            if mode in ["scale_xy", "scale", "progress", "longitudinal_scale"]:
                # x is treated as longitudinal progress and y as lateral offset.
                aug[..., 0] = aug[..., 0] * (1.0 + long_deltas[repeat_idx])
                if aug.shape[-1] > 1:
                    aug[..., 1] = aug[..., 1] * (1.0 + lat_deltas[repeat_idx])
            elif mode in ["scale_x", "progress_only"]:
                aug[..., 0] = aug[..., 0] * (1.0 + long_deltas[repeat_idx])
            elif mode in ["noise", "gaussian"]:
                # Optional stochastic mode. Keep repeat_idx==0 unchanged so the
                # original anchor bank is always included.
                if repeat_idx > 0:
                    noise = torch.randn_like(aug[..., :2]) * self.inference_anchor_noise_std
                    aug[..., :2] = aug[..., :2] + noise
            else:
                raise ValueError(
                    "Unsupported inference_candidate_augmentation="
                    f"{self.inference_candidate_augmentation}. Supported: scale_xy, scale_x, noise, none."
                )

            if self.inference_anchor_recompute_heading and aug.shape[-1] >= 3:
                aug[..., 2] = self._estimate_heading_from_xy(aug[..., :2])

            expanded_blocks.append(aug)

        expanded = torch.cat(expanded_blocks, dim=1)
        return expanded[:, :target_num].contiguous()

    @staticmethod
    def _make_symmetric_deltas(num_values, max_abs, device, dtype):
        """Return [0, +a1, -a1, +a2, -a2, ...] up to num_values."""
        if num_values <= 1 or max_abs <= 0:
            return torch.zeros(num_values, device=device, dtype=dtype)

        values = [0.0]
        num_magnitudes = int(math.ceil((num_values - 1) / 2))
        for k in range(1, num_magnitudes + 1):
            mag = float(max_abs) * k / max(1, num_magnitudes)
            values.append(mag)
            if len(values) < num_values:
                values.append(-mag)
        return torch.tensor(values[:num_values], device=device, dtype=dtype)

    @staticmethod
    def _estimate_heading_from_xy(xy):
        """
        Estimate per-step heading from consecutive xy points. The first segment
        starts at ego origin (0, 0), matching the local trajectory convention.
        """
        origin = torch.zeros_like(xy[..., :1, :])
        xy_with_origin = torch.cat([origin, xy], dim=-2)
        delta = xy_with_origin[..., 1:, :] - xy_with_origin[..., :-1, :]
        return torch.atan2(delta[..., 1], delta[..., 0])

    def _world_model_forward(self, flat_bev, batch_size):
        """
        Generate Unified Spatio-Temporal BEV Feature.
        """
        x = flat_bev

        bev_st_flat = self.latent_world_model(x)  # [B, HW, C]
        return bev_st_flat

    def extract_trajectory_feature(self, features, targets=None, use_inference_candidates=False, enable_aux_map_outputs=True):
        results = {}
        cam_feat = features["camera_feature"]  # [B,3,256,1024]
        # lidar_feat = features["lidar_feature"]      # [B,1,256,256]
        status_feat = features["status_feature"]  # [B, 8]
        batch_size = status_feat.shape[0]
        ego_status_seq = features.get("ego_status_seq", status_feat[:, None, :])  # [B, T_status, 8]

        lidar_seq = features["lidar_feature_seq"]
        ego_poses_seq = features["ego_poses_seq"]

        aligned_lidar_seq = self.align_history_features(lidar_seq, ego_poses_seq)

        results["aligned_lidar_seq"] = aligned_lidar_seq

        B, T, C, H, W = aligned_lidar_seq.shape
        lidar_concat = aligned_lidar_seq.view(B, T * C, H, W)

        # Backbone
        _, backbone_bev, _ = self._backbone(cam_feat, lidar_concat)
        flat_bev = self._bev_downscale(backbone_bev).flatten(-2, -1).permute(0, 2, 1)

        spatial_pos = self.spatial_position_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        flat_bev_with_pos = flat_bev + spatial_pos

        bev_st_flat = self._world_model_forward(flat_bev_with_pos, batch_size)

        status_enc = self._status_encoding(status_feat)[:, None, :]  # [B, 1, C]

        init_anchors = self._get_runtime_trajectory_anchors(
            batch_size=batch_size,
            device=status_feat.device,
            dtype=status_feat.dtype,
            use_inference_candidates=use_inference_candidates,
        )
        ego_feat, num_traj = self.encode_traj_into_ego_feat(status_enc, init_anchors, batch_size)

        # Offsets
        offset_dict = self._predict_offset(ego_feat, flat_bev_with_pos)
        results.update(offset_dict)

        traj_offset = offset_dict["trajectory_offset"]
        candidate_trajs = init_anchors + traj_offset

        # Auxiliary Losses
        if self.use_agent_loss:
            agent_q = self.agent_query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            agent_q_out = self.agent_tf_decoder(agent_q, flat_bev_with_pos)
            agents = self.agent_head(agent_q_out)
            results.update(agents)

        if enable_aux_map_outputs and (self.use_map_loss or self.use_future_map_loss):
            selected_future_indices = self._sample_future_supervision_indices(device=bev_st_flat.device)
            readout_selected_indices = [0] + [idx + 1 for idx in selected_future_indices]

            selected_bev_seq = self.future_bev_readout(
                bev_st_flat,
                h=8,
                w=8,
                selected_indices=readout_selected_indices,
            )  # [B, 1+S, C, 8, 8]
            selected_maps = self._process_future_map_seq(selected_bev_seq)  # [B, 1+S, K, H, W]

            if self.use_map_loss:
                results["bev_semantic_map"] = selected_maps[:, 0]

            if self.use_future_map_loss:
                results["pred_future_maps_seq"] = selected_maps[:, 1:]
                results["future_supervision_indices"] = torch.tensor(
                    selected_future_indices,
                    dtype=torch.long,
                    device=selected_maps.device,
                )

        return {
            "results": results,
            "batch_size": batch_size,
            "num_traj": num_traj,
            "flatten_bev_feature": flat_bev, 
            "bev_st_flat": bev_st_flat,      
            "ego_feat": ego_feat,
            "candidate_trajectories": candidate_trajs,
            "current_status": status_feat,
            "ego_status_seq": ego_status_seq,
        }

    def _encode_reward_status_context(self, current_status, ego_status_seq=None):
        """
        Encode current and historical ego status into a reward-side context vector.

        current_status: [B, 8] = command(4) + velocity(2) + acceleration(2)
        ego_status_seq: [B, T_status, 8], historical + current status sequence
        return: [B, hidden_dim]
        """
        if current_status.dim() != 2:
            current_status = current_status.view(current_status.shape[0], -1)

        if ego_status_seq is None:
            ego_status_seq = current_status[:, None, :]
        elif ego_status_seq.dim() == 2:
            ego_status_seq = ego_status_seq[:, None, :]

        ego_status_seq = ego_status_seq.to(device=current_status.device, dtype=current_status.dtype)

        current_context = self.current_status_reward_encoder(current_status)          # [B, C]
        _, history_hidden = self.ego_status_history_encoder(ego_status_seq)          # [1, B, C]
        history_context = self.ego_status_history_norm(history_hidden[-1])           # [B, C]

        status_context = self.reward_status_context_fuser(
            torch.cat([current_context, history_context], dim=-1)
        )                                                                            # [B, C]
        return status_context

    def extract_reward_feature(self, trajectory_outputs, targets=None) -> Dict[str, torch.Tensor]:
        results = trajectory_outputs["results"]
        batch_size = trajectory_outputs["batch_size"]
        num_traj = trajectory_outputs["num_traj"]

        bev_st_flat = trajectory_outputs["bev_st_flat"]  # [B, H*W, C]

        H, W = 8, 8
        bev_st_spatial = bev_st_flat.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, H, W)

        trajs = trajectory_outputs["candidate_trajectories"]
        trajs_coords = trajs[..., :2]

        reward_status_context = self._encode_reward_status_context(
            trajectory_outputs["current_status"],
            trajectory_outputs.get("ego_status_seq", None),
        )  # [B, C]

        reward_embedding = self.reward_encoder(
            trajs_coords,
            bev_st_spatial,
            status_context=reward_status_context,
        )  # [B*N, C]
        reward_embedding = reward_embedding.view(batch_size, num_traj, -1)
        reward_status_context = reward_status_context[:, None, :].expand(-1, num_traj, -1)
        reward_embedding = self.reward_embedding_status_fuser(
            torch.cat([reward_embedding, reward_status_context], dim=-1)
        )  # [B, N, C]

        im_score = self.im_reward_head(reward_embedding)    # [B, N, 1]

        results["reward_embedding"] = reward_embedding
        results["reward_scores"] = im_score.squeeze(-1)

        return trajectory_outputs

    def process_trajectory_and_reward(self, features, targets=None, use_inference_candidates=False, enable_aux_map_outputs=True):
        trajectory_outputs = self.extract_trajectory_feature(
            features,
            targets,
            use_inference_candidates=use_inference_candidates,
            enable_aux_map_outputs=enable_aux_map_outputs,
        )
        final_results = self.extract_reward_feature(trajectory_outputs, targets)
        return final_results

    def forward_train(self, features, targets=None) -> Dict[str, torch.Tensor]:
        self.is_eval = False
        result = {}

        encoder_results = self.process_trajectory_and_reward(
            features,
            targets,
            use_inference_candidates=False,
            enable_aux_map_outputs=True,
        )

        train_candidate_num = encoder_results["candidate_trajectories"].shape[1]
        base_candidate_num = self.trajectory_anchors.shape[0]
        if train_candidate_num != base_candidate_num:
            raise RuntimeError(
                "Training path unexpectedly used inference candidate anchors: "
                f"got {train_candidate_num}, expected base anchor count {base_candidate_num}. "
                "Call forward_train/process_trajectory_and_reward with use_inference_candidates=False."
            )

        im_rewards = encoder_results["results"].pop("reward_scores")
        reward_embedding = encoder_results["results"].pop("reward_embedding")

        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)
        result["im_rewards"] = im_rewards_softmax
        result["trajectory_anchors"] = self.trajectory_anchors
        result["candidate_trajectories"] = encoder_results["candidate_trajectories"]

        sim_rewards = []
        for sim_head in self.sim_reward_heads:
            s_out = sim_head(reward_embedding)
            sim_rewards.append(s_out)
        sim_rewards_logits = torch.cat(sim_rewards, dim=-1).permute(0, 2, 1)  # [B, 5, N]
        result["sim_rewards_logits"] = sim_rewards_logits
        result["sim_rewards"] = sim_rewards_logits.sigmoid()

        result.update(encoder_results["results"])

        return result

    def forward_test(self, features, targets=None) -> Dict[str, torch.Tensor]:
        self.is_eval = True

        encoder_results = self.process_trajectory_and_reward(
            features,
            targets=None,
            use_inference_candidates=True,
            enable_aux_map_outputs=False,
        )

        im_rewards = encoder_results["results"]["reward_scores"]
        reward_embedding = encoder_results["results"]["reward_embedding"]
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)

        sim_rewards_list = []
        for sim_head in self.sim_reward_heads:
            sim_rewards_list.append(sim_head(reward_embedding).sigmoid())

        final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards_list)

        trajectory_candidates = encoder_results["candidate_trajectories"]
        poses = self.select_best_trajectory_from_candidates(final_rewards, trajectory_candidates)

        results = {
            "trajectory": poses,
            "all_trajectory": trajectory_candidates,
            "final_rewards": final_rewards,
            "im_rewards": im_rewards_softmax,
            "num_inference_candidates": torch.tensor(
                trajectory_candidates.shape[1],
                device=trajectory_candidates.device,
                dtype=torch.long,
            ),
        }
        return results

    def encode_traj_into_ego_feat(self, ego_status_feat, init_trajectory_anchor, batch_size):
        num_traj = init_trajectory_anchor.shape[1]
        init_traj = init_trajectory_anchor.reshape(batch_size, num_traj, -1)
        traj_feat = self.mlp_planning_vb(init_traj)
        traj_feat = self.cluster_encoder(traj_feat)  # [B, N, C]

        ego_status_expanded = ego_status_feat.repeat(1, num_traj, 1)
        ego_feat = torch.cat([ego_status_expanded, traj_feat], dim=-1)
        ego_feat = self.encode_ego_feat_mlp(ego_feat).unsqueeze(-2)  # [B, N, 1, C]
        return ego_feat, num_traj

    def _predict_offset(self, ego_feat, flat_bev):
        ego_feat_sq = ego_feat.squeeze(2)
        ego_feat_out = self.offset_tf_decoder(ego_feat_sq, flat_bev)
        traj_offset = self.offset_head(ego_feat_out)
        scores = self.offset_score_head(ego_feat_out).squeeze(-1)
        return {
            "trajectory_offset": traj_offset,
            "trajectory_offset_rewards": torch.softmax(scores, dim=-1)
        }

    def _process_map(self, flat_bev, batch_size, h=8, w=8):
        flat_bev = flat_bev.permute(0, 2, 1).reshape(batch_size, -1, h, w)
        flat_bev = self._bev_upscale(flat_bev)
        up_bev = self.bev_upsample_head(flat_bev)
        sem_map = self.bev_semantic_head(up_bev)
        return sem_map, up_bev

    def _process_future_map_seq(self, future_bev_seq):
        """
        future_bev_seq: [B, T, C, H, W]
        return:
            pred_future_maps_seq: [B, T, num_classes, H_out, W_out]
        """
        B, T, C, H, W = future_bev_seq.shape
        x = future_bev_seq.reshape(B * T, C, H, W)
        x = self._bev_upscale(x)
        up_bev = self.bev_upsample_head(x)
        sem_map = self.bev_semantic_head(up_bev)

        _, num_cls, H_out, W_out = sem_map.shape
        sem_map = sem_map.view(B, T, num_cls, H_out, W_out)
        return sem_map

    def select_best_trajectory_from_candidates(self, final_rewards, candidate_trajs):
        best_idx = torch.argmax(final_rewards, dim=-1)
        B, N, T, D = candidate_trajs.shape
        batch_indices = torch.arange(B, device=best_idx.device)
        best_traj = candidate_trajs[batch_indices, best_idx]
        return best_traj

    def weighted_reward_calculation(self, im_rewards, sim_rewards_list):
        assert len(sim_rewards_list) == 5
        w = self.reward_weights
        sw = self.sim_weights
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = [s.squeeze(-1) for s in sim_rewards_list]
        cost = (
                w[0] * torch.log(im_rewards + 1e-6) +
                w[1] * torch.log(S_NC + 1e-6) +
                w[2] * torch.log(S_DAC + 1e-6) +
                w[3] * torch.log(sw[0] * S_TTC + sw[1] * S_COMFORT + sw[2] * S_EP + 1e-6)
        )
        return cost

    def align_history_features(self, lidar_seq, ego_poses_seq):
        B, T, C, H, W = lidar_seq.shape

        pose_cur = ego_poses_seq[:, -1, :]  # [B, 3]
        x_c, y_c, theta_c = pose_cur[:, 0], pose_cur[:, 1], pose_cur[:, 2]

        aligned_seq = []
        for t in range(T):
            if t == T - 1:
                aligned_seq.append(lidar_seq[:, t])
                continue

            pose_prev = ego_poses_seq[:, t, :]  # [B, 3]
            x_p, y_p, theta_p = pose_prev[:, 0], pose_prev[:, 1], pose_prev[:, 2]

            dx = x_c - x_p
            dy = y_c - y_p

            cos_p = torch.cos(theta_p)
            sin_p = torch.sin(theta_p)
            t_x = cos_p * dx + sin_p * dy
            t_y = -sin_p * dx + cos_p * dy

            delta_theta = theta_c - theta_p
            cos_dt = torch.cos(delta_theta)
            sin_dt = torch.sin(delta_theta)

            scale_x = self.x_max_meters  # 32.0
            scale_y = self.y_max_meters  # 32.0

            theta_mat = torch.zeros(B, 2, 3, dtype=torch.float32, device=lidar_seq.device)

            theta_mat[:, 0, 0] = cos_dt
            theta_mat[:, 0, 1] = sin_dt
            theta_mat[:, 0, 2] = t_y / scale_y

            theta_mat[:, 1, 0] = -sin_dt
            theta_mat[:, 1, 1] = cos_dt
            theta_mat[:, 1, 2] = t_x / scale_x

            grid = F.affine_grid(theta_mat, torch.Size([B, C, H, W]), align_corners=False)

            prev_feat = lidar_seq[:, t]
            aligned_feat = F.grid_sample(prev_feat, grid, align_corners=False, mode='bilinear', padding_mode='zeros')

            aligned_seq.append(aligned_feat)

        return torch.stack(aligned_seq, dim=1)


class AgentHead(nn.Module):
    def __init__(self, num_agents, d_ffn, d_model):
        super().__init__()
        self._mlp_states = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(),
                                         nn.Linear(d_ffn, BoundingBox2DIndex.size()))
        self._mlp_label = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, x):
        states = self._mlp_states(x)
        states[..., BoundingBox2DIndex.POINT] = states[..., BoundingBox2DIndex.POINT].tanh() * 32
        states[..., BoundingBox2DIndex.HEADING] = states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        labels = self._mlp_label(x).squeeze(-1)
        return {"agent_states": states, "agent_labels": labels}

class BEVUpsampleHead(nn.Module):
    def __init__(self, config, channel=64, c5_chs=512):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=config.bev_upsample_factor, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(
            size=(config.lidar_resolution_height // config.bev_down_sample_factor,
                  config.lidar_resolution_width // config.bev_down_sample_factor),
            mode="bilinear", align_corners=False
        )
        self.up_conv5 = nn.Conv2d(channel, channel, 3, padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.c5_conv = nn.Conv2d(c5_chs, channel, 1)

    def forward(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

class TrajectoryOffsetHead(nn.Module):
    def __init__(self, num_poses=8, d_ffn=1024, d_model=256):
        super().__init__()
        self._num_poses = num_poses
        self._mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(),
            nn.Linear(d_ffn, num_poses * StateSE2Index.size())
        )

    def forward(self, x):
        bz, num, _ = x.shape
        poses = self._mlp(x).reshape(bz, -1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return poses


class FutureBEVSEReadout(nn.Module):
    def __init__(self, hidden_dim=256, total_future_steps=8, selected_indices=None, reduction=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_future_steps = total_future_steps
        self.selected_indices = selected_indices if selected_indices is not None else list(range(total_future_steps))

        mid_dim = max(hidden_dim // reduction, 32)

        self.time_embedding = nn.Embedding(total_future_steps, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)

        self.excitation = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, hidden_dim),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.time_embedding.weight[0])  # current
        nn.init.normal_(self.time_embedding.weight[1:], mean=0.0, std=0.02)

    def forward(self, bev_st_flat, h=8, w=8, selected_indices=None):
        B, HW, C = bev_st_flat.shape
        bev_st = bev_st_flat.permute(0, 2, 1).reshape(B, C, h, w)

        context = F.adaptive_avg_pool2d(bev_st, output_size=1).flatten(1)

        selected_indices = self.selected_indices if selected_indices is None else selected_indices

        future_bev_seq = []
        for t_idx in selected_indices:
            t_idx = int(t_idx)
            if t_idx < 0 or t_idx >= self.total_future_steps:
                raise ValueError(
                    f"readout time index must be in [0, {self.total_future_steps - 1}], got {t_idx}"
                )
            t_embed = self.time_embedding.weight[t_idx].unsqueeze(0).expand(B, -1)
            cond = self.pre_norm(context + t_embed)

            gate = 2.0 * self.excitation(cond)
            gate = gate.view(B, C, 1, 1)

            bev_t = bev_st * gate
            future_bev_seq.append(bev_t)

        future_bev_seq = torch.stack(future_bev_seq, dim=1)   # [B, S, C, H, W]
        return future_bev_seq


class TrajectoryDeformableRewardEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_heads=8, num_points=4, pooling_type='max'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads
        self.pooling_type = pooling_type

        self.query_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        self.offset_linear = nn.Linear(hidden_dim, num_heads * num_points * 2)
        self.weight_linear = nn.Linear(hidden_dim, num_heads * num_points)

        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        if pooling_type == 'conv':
            self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )

        nn.init.constant_(self.offset_linear.weight, 0.)
        nn.init.constant_(self.offset_linear.bias, 0.)

    def get_reference_points(self, trajs_coords, H=8, W=8):
        delta_x = trajs_coords[..., 0]
        delta_y = trajs_coords[..., 1]
        grid_y = delta_y / 32.0
        grid_x = (delta_x / 16.0) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, trajs_coords, bev_features, status_context=None):
        B, N, T, _ = trajs_coords.shape
        _, C_bev, H, W = bev_features.shape

        queries = self.query_encoder(trajs_coords)  # [B, N, T, C]

        if status_context is not None:
            if status_context.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"status_context last dim must be {self.hidden_dim}, "
                    f"but got {status_context.shape[-1]}"
                )
            status_context = status_context.to(device=queries.device, dtype=queries.dtype)
            queries = queries + status_context[:, None, None, :]

        offsets = self.offset_linear(queries)
        offsets = offsets.view(B, N, T, self.num_heads, self.num_points, 2) * (1.0 / max(H, W))

        attn_weights = self.weight_linear(queries)
        attn_weights = attn_weights.view(B, N, T, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        ref_points = self.get_reference_points(trajs_coords, H, W)
        ref_points = ref_points.unsqueeze(3).unsqueeze(4)
        sample_locs = ref_points + offsets

        bev_v = bev_features.permute(0, 2, 3, 1)
        bev_v = self.value_proj(bev_v).permute(0, 3, 1, 2)
        bev_v = bev_v.view(B, self.num_heads, self.head_dim, H, W)
        bev_v = bev_v.reshape(B * self.num_heads, self.head_dim, H, W)

        sample_locs = sample_locs.permute(0, 3, 1, 2, 4, 5).contiguous()
        sample_locs = sample_locs.view(B * self.num_heads, N * T, self.num_points, 2)

        sampled = F.grid_sample(
            bev_v, sample_locs, mode='bilinear', padding_mode='zeros', align_corners=False
        )

        sampled = sampled.view(B, self.num_heads, self.head_dim, N, T, self.num_points)
        attn_weights = attn_weights.permute(0, 3, 1, 2, 4).unsqueeze(2)

        attended_features = (sampled * attn_weights).sum(dim=-1)
        attended_features = attended_features.permute(0, 3, 4, 1, 2).contiguous()
        attended_features = attended_features.view(B, N, T, self.hidden_dim)

        out_features = self.output_proj(attended_features)
        out_features = F.layer_norm(out_features + queries, [self.hidden_dim])
        out_features = F.relu(out_features)

        x = out_features.view(B * N, T, self.hidden_dim).permute(0, 2, 1)

        if self.pooling_type == 'max':
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        elif self.pooling_type == 'avg':
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif self.pooling_type == 'conv':
            x = self.conv1d(x)
            x = F.relu(x)
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        final_embed = self.output_mlp(x)
        return final_embed
