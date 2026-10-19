from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any


def compute_RFWorld_loss(
    targets,
    predictions,
    config,
) -> torch.Tensor:
    loss_dict = {}
    # offset regression
    traj_offset_loss = compute_traj_offset_loss(predictions, targets, config)
    loss_dict['traj_offset_loss'] = traj_offset_loss * config.traj_offset_loss_weight

    # offset reward
    trajectory_anchors = predictions["trajectory_anchors"]
    imitation_rewards = predictions["trajectory_offset_rewards"]
    offset_im_reward_loss = compute_im_reward_loss(targets, imitation_rewards, trajectory_anchors)
    loss_dict['offset_im_reward_loss'] = offset_im_reward_loss * config.offset_im_reward_weight

    # im & sim rewards
    candidate_trajectories = predictions["candidate_trajectories"]
    imitation_rewards = predictions["im_rewards"]
    im_reward_loss = compute_im_reward_loss(
        targets,
        imitation_rewards,
        candidate_trajectories,
    )

    sim_rewards_logits = predictions["sim_rewards_logits"]
    sim_rewards = predictions["sim_rewards"]
    sim_reward_loss = compute_sim_reward_loss(targets, sim_rewards_logits, config)
    sim_ep_ranking_margin = (
        config.sim_ep_ranking_margin
        if hasattr(config, "sim_ep_ranking_margin")
        else 0.05
    )
    sim_ep_ranking_loss = compute_sim_ep_margin_ranking_loss(
        targets,
        sim_rewards_logits,
        margin=sim_ep_ranking_margin,
    )
    sim_ep_ranking_loss_weight = (
        config.sim_ep_ranking_loss_weight
        if hasattr(config, "sim_ep_ranking_loss_weight")
        else 0.0
    )

    loss_dict['im_reward_loss'] = im_reward_loss
    loss_dict['sim_reward_loss'] = sim_reward_loss
    loss_dict['sim_ep_ranking_loss'] = sim_ep_ranking_loss * sim_ep_ranking_loss_weight

    use_agent_loss = config.use_agent_loss if hasattr(config, "use_agent_loss") else True
    if use_agent_loss:
        agent_cls_loss, agent_box_loss = compute_RFWorld_agent_loss(
                                            targets["agent_states"],
                                            targets["agent_labels"],
                                            predictions["agent_states"],
                                            predictions["agent_labels"],
                                            config,
                                            )
        agent_class_weight = config.agent_class_weight if hasattr(config, "agent_class_weight") else 0.0
        agent_box_weight = config.agent_box_weight if hasattr(config, "agent_box_weight") else 0.0

        loss_dict['agent_cls_loss'] = agent_cls_loss * agent_class_weight
        loss_dict['agent_box_loss'] = agent_box_loss * agent_box_weight

    if config.use_map_loss:
        use_focal_loss_for_map = config.use_focal_loss_for_map if hasattr(config, 'use_focal_loss_for_map') else False
        focal_loss_alpha = config.focal_loss_alpha if hasattr(config, 'focal_loss_alpha') else 0.25
        focal_loss_gamma = config.focal_loss_gamma if hasattr(config, 'focal_loss_gamma') else 2.0
        focal_loss_fn = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction='none')  

        map_loss_raw = focal_loss_fn(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        loss_dict['map_loss'] = map_loss_raw.mean() * config.bev_semantic_weight

        if "pred_future_maps_seq" in predictions and "future_semantic_maps_seq" in targets:
            pred_fut_maps = predictions["pred_future_maps_seq"]  # [B, S, C, H, W]

            total_steps = targets["future_semantic_maps_seq"].shape[1]

            # Prefer the indices sampled by the model in this forward step. This keeps
            # random future-map readout and GT supervision aligned. Fall back to the
            # legacy fixed config for backward compatibility.
            selected_indices = predictions.get("future_supervision_indices", None)
            if selected_indices is None:
                future_steps = getattr(config, "future_supervision_steps", None)
                if future_steps is None or len(future_steps) == 0:
                    selected_indices = list(range(total_steps))
                else:
                    selected_indices = list(future_steps)

            if torch.is_tensor(selected_indices):
                selected_indices = selected_indices.to(
                    device=targets["future_semantic_maps_seq"].device,
                    dtype=torch.long,
                )
            else:
                selected_indices = torch.tensor(
                    list(selected_indices),
                    device=targets["future_semantic_maps_seq"].device,
                    dtype=torch.long,
                )

            # focal raw: [B*S, H, W]
            B, S, C, H, W = pred_fut_maps.shape
            if selected_indices.numel() != S:
                raise ValueError(
                    "future supervision index count must match pred_future_maps_seq length: "
                    f"got {selected_indices.numel()} indices for S={S}"
                )
            if selected_indices.numel() > 0:
                if selected_indices.min().item() < 0 or selected_indices.max().item() >= total_steps:
                    raise ValueError(
                        f"future supervision indices must be in [0, {total_steps - 1}], "
                        f"got {selected_indices.detach().cpu().tolist()}"
                    )

            gt_fut_maps = targets["future_semantic_maps_seq"].index_select(1, selected_indices)  # [B, S, H, W]

            pred_flat = pred_fut_maps.reshape(B * S, C, H, W)
            gt_flat = gt_fut_maps.reshape(B * S, H, W)

            fut_map_loss_raw = focal_loss_fn(pred_flat, gt_flat.long()).view(B, S, H, W)

            if getattr(config, "use_future_valid_mask", False) and "future_valid_masks_seq" in targets:
                valid_mask = targets["future_valid_masks_seq"].index_select(1, selected_indices).float()  # [B, S, H, W]
            else:
                valid_mask = torch.ones_like(fut_map_loss_raw)

            per_step_loss = (fut_map_loss_raw * valid_mask).sum(dim=(0, 2, 3)) / valid_mask.sum(
                dim=(0, 2, 3)).clamp_min(1.0)

            fut_step_weights = getattr(config, "future_supervision_weights", [1.0] * S)
            if torch.is_tensor(fut_step_weights):
                fut_step_weights = fut_step_weights.to(dtype=per_step_loss.dtype, device=per_step_loss.device)
            else:
                fut_step_weights = torch.tensor(list(fut_step_weights), dtype=per_step_loss.dtype, device=per_step_loss.device)
            if fut_step_weights.numel() == total_steps:
                fut_step_weights = fut_step_weights.index_select(0, selected_indices.to(per_step_loss.device))
            elif fut_step_weights.numel() != S:
                raise ValueError(
                    "future_supervision_weights must have length S or total future steps: "
                    f"got {fut_step_weights.numel()}, S={S}, total_steps={total_steps}"
                )
            fut_step_weights = fut_step_weights / fut_step_weights.sum().clamp_min(1e-6)

            fut_map_loss = (per_step_loss * fut_step_weights).sum()

            fut_weight = config.fut_bev_semantic_weight if hasattr(config, "fut_bev_semantic_weight") else 1.0
            loss_dict["fut_map_loss"] = fut_map_loss * fut_weight
        else:
            loss_dict['fut_map_loss'] = torch.tensor(0.0, device=predictions["bev_semantic_map"].device)

    return loss_dict

def compute_traj_offset_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Any
) -> torch.Tensor:
    """
    Computes the trajectory offset loss using a Winner-Take-All (WTA) strategy.

    Args:
        predictions (Dict[str, torch.Tensor]): Model predictions containing:
            - "trajectory_anchors": Tensor of shape (batch_size, 256, traj_dim)
            - "trajectory_offset": Tensor of shape (batch_size, 256, traj_dim)
        targets (Dict[str, torch.Tensor]): Ground truth data containing:
            - "trajectory": Tensor of shape (batch_size, traj_dim)
        config (Any): Configuration object with the following attributes:
            - use_traj_offset (bool): Whether to compute the trajectory offset loss.
            - traj_offset_loss_weight (float): Weight for the trajectory offset loss.

    Returns:
        torch.Tensor: Weighted trajectory offset loss. Returns zero tensor if `use_traj_offset` is False.
    """
    # Extract necessary tensors from predictions and targets
    trajectory_anchors = predictions["trajectory_anchors"]       # Shape: (batch_size, 256, traj_dim)
    predicted_offsets = predictions["trajectory_offset"]      # Shape: (batch_size, 256, traj_dim)
    gt_trajectories = targets["trajectory"].squeeze(1)     # Shape: (batch_size, traj_dim)

    bz = gt_trajectories.shape[0]
    num_trajs = trajectory_anchors.shape[0]
    gt_trajectories = gt_trajectories.reshape(bz, -1)
    predicted_offsets = predicted_offsets.reshape(bz, num_trajs, -1)
    trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bz, 1, 1, 1).reshape(bz, num_trajs, -1)

    # Ensure that the batch sizes match
    batch_size, num_clusters, traj_dim = trajectory_anchors.shape
    assert gt_trajectories.shape == (batch_size, traj_dim), \
    f"Ground truth trajectories shape {gt_trajectories.shape} does not match expected {(batch_size, traj_dim)}"

    # Expand ground truth trajectories for broadcasting: (batch_size, 1, traj_dim)
    gt_expanded = gt_trajectories.unsqueeze(1)            # Shape: (batch_size, 1, traj_dim)

    # Compute L2 distances between each cluster center and the ground truth trajectory
    # Resulting shape: (batch_size, 256)
    distances = torch.norm(trajectory_anchors - gt_expanded, dim=2)  # Euclidean distance

    # Find the index of the closest cluster center for each sample in the batch
    # Shape: (batch_size,)
    min_indices = torch.argmin(distances, dim=1)            # Winner cluster index per sample

    # Create a batch index tensor for advanced indexing
    batch_indices = torch.arange(batch_size, device=trajectory_anchors.device)  # Shape: (batch_size,)

    # Gather the selected (winner) cluster centers based on min_indices
    # Shape: (batch_size, traj_dim)
    selected_trajectory_anchors = trajectory_anchors[batch_indices, min_indices]

    # Compute the ground truth offsets: gt_trajectory - selected_trajectory_anchor
    # Shape: (batch_size, traj_dim)
    gt_offsets = gt_trajectories - selected_trajectory_anchors

    # Gather the predicted offsets corresponding to the selected (winner) cluster centers
    # Shape: (batch_size, traj_dim)
    selected_predicted_offsets = predicted_offsets[batch_indices, min_indices]

    # Compute the L1 loss (Mean Absolute Error) between predicted offsets and ground truth offsets
    # Reduction is 'mean' to average over all samples and trajectory dimensions
    traj_offset_loss = F.l1_loss(selected_predicted_offsets, gt_offsets, reduction='mean')

    return traj_offset_loss

def compute_im_reward_loss(
    targets: Dict[str, torch.Tensor],
    prediction_rewards: torch.Tensor,
    trajectories: torch.Tensor,
    detach_trajectories_for_target: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    B = targets["trajectory"].shape[0]

    target_trajectory = targets["trajectory"][:, -1]
    target_trajectory = target_trajectory.reshape(B, -1).float()

    if trajectories.dim() == 3:
        # [N, T, D] -> [B, N, T*D]
        N = trajectories.shape[0]
        traj_flat = trajectories.reshape(N, -1).unsqueeze(0).expand(B, -1, -1)
    elif trajectories.dim() == 4:
        # [B, N, T, D] -> [B, N, T*D]
        traj_flat = trajectories.reshape(B, trajectories.shape[1], -1)
    else:
        raise ValueError(f"Unexpected trajectories shape: {trajectories.shape}")

    if detach_trajectories_for_target:
        traj_flat = traj_flat.detach()

    l2_distances = torch.norm(
        traj_flat - target_trajectory.unsqueeze(1),
        dim=-1,
    )  # [B, N]

    reward_targets = torch.softmax(-l2_distances / temperature, dim=-1)

    prediction_rewards = prediction_rewards.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    im_reward_loss = -(reward_targets * prediction_rewards.log()).sum(dim=-1).mean()

    return im_reward_loss

def compute_sim_reward_loss(
    targets: Dict[str, torch.Tensor],
    predicted_rewards_logits: torch.Tensor,
    config: Any = None,
) -> torch.Tensor:
    """
    Numerically stable simulation reward loss.

    Use logits with binary_cross_entropy_with_logits instead of applying log()
    to sigmoid probabilities. The previous probability-space BCE can produce
    NaN under AMP/FP16 because clamp(1e-6, 1 - 1e-6) rounds the upper bound
    back to 1.0, leading to 0 * log(0).
    """
    target_rewards = targets['sim_reward'][:, -1].to(
        device=predicted_rewards_logits.device,
        dtype=predicted_rewards_logits.dtype,
    )  # [B, 5, N], last frame

    if predicted_rewards_logits.shape != target_rewards.shape:
        raise ValueError(
            "predicted_rewards_logits and target_rewards must be aligned as [B, 5, N], "
            f"got pred={tuple(predicted_rewards_logits.shape)}, target={tuple(target_rewards.shape)}"
        )

    metric_dim = target_rewards.shape[1]
    metric_weights = [1.0] * metric_dim
    if config is not None:
        metric_weights = getattr(
            config,
            "sim_loss_metric_weights",
            getattr(config, "sim_reward_metric_weights", metric_weights),
        )

    if torch.is_tensor(metric_weights):
        metric_weights = metric_weights.to(
            dtype=predicted_rewards_logits.dtype,
            device=predicted_rewards_logits.device,
        )
    else:
        metric_weights = torch.tensor(
            list(metric_weights),
            dtype=predicted_rewards_logits.dtype,
            device=predicted_rewards_logits.device,
        )
    if metric_weights.numel() != metric_dim:
        raise ValueError(
            f"sim_loss_metric_weights must contain {metric_dim} values "
            f"in NC, DAC, EP, TTC, Comfort order, got {metric_weights.numel()}"
        )

    # for unavailable comfort labels; mask them instead of letting them corrupt BCE.
    valid_mask = (
        torch.isfinite(target_rewards)
        & torch.isfinite(predicted_rewards_logits)
        & (target_rewards >= 0.0)
        & (target_rewards <= 1.0)
    )

    if not valid_mask.any():
        return predicted_rewards_logits.new_tensor(0.0)

    logit_clamp = None
    if config is not None:
        logit_clamp = getattr(config, "sim_loss_logit_clamp", 20.0)
    if logit_clamp is not None and float(logit_clamp) > 0:
        safe_logits = predicted_rewards_logits.clamp(-float(logit_clamp), float(logit_clamp))
    else:
        safe_logits = predicted_rewards_logits

    safe_logits = torch.where(valid_mask, safe_logits, torch.zeros_like(safe_logits))
    safe_targets = torch.where(valid_mask, target_rewards, torch.zeros_like(target_rewards))

    bce = F.binary_cross_entropy_with_logits(
        safe_logits,
        safe_targets,
        reduction='none',
    )  # [B, 5, N]

    weights = metric_weights.view(1, metric_dim, 1)
    weighted_bce = bce * weights * valid_mask.to(dtype=bce.dtype)

    # Preserve legacy scale when all weights are 1 and all entries are valid:
    # mean over [B, 5, N] multiplied by 5.
    denom = valid_mask.to(dtype=bce.dtype).sum().clamp_min(1.0)
    sim_reward_loss = weighted_bce.sum() / denom * metric_dim

    return sim_reward_loss

def compute_sim_ep_margin_ranking_loss(
        targets: Dict[str, torch.Tensor],
        predicted_rewards_logits: torch.Tensor,
        margin: float = 0.05,
        ep_index: int = 2,
) -> torch.Tensor:
    """
    Compute pairwise margin ranking loss for the EP simulation reward from raw logits.

    Args:
        targets: Dictionary containing ``sim_reward`` with shape [B, T, 5, N].
        predicted_rewards_logits: Raw simulation reward logits with shape [B, 5, N].
        margin: Margin used by torch.nn.MarginRankingLoss in logit space.
        ep_index: Index of EP in the metric dimension. The order is
            NC, DAC, EP, TTC, Comfort, so EP is index 2.

    Returns:
        Scalar tensor. Returns 0 when there are no non-tied EP pairs.
    """
    target_rewards = targets['sim_reward'][:, -1].to(
        device=predicted_rewards_logits.device,
        dtype=predicted_rewards_logits.dtype,
    )

    if predicted_rewards_logits.dim() != 3:
        raise ValueError(
            f"predicted_rewards_logits should have shape [B, 5, N], got {tuple(predicted_rewards_logits.shape)}"
        )
    if target_rewards.dim() != 3:
        raise ValueError(
            f"target_rewards should have shape [B, 5, N], got {tuple(target_rewards.shape)}"
        )
    if predicted_rewards_logits.shape != target_rewards.shape:
        raise ValueError(
            "predicted_rewards_logits and target_rewards must be aligned as [B, 5, N], "
            f"got pred={tuple(predicted_rewards_logits.shape)}, target={tuple(target_rewards.shape)}"
        )
    if ep_index >= predicted_rewards_logits.shape[1]:
        raise ValueError(
            f"ep_index={ep_index} is out of range for metric dimension {predicted_rewards_logits.shape[1]}"
        )

    pred_ep = predicted_rewards_logits[:, ep_index, :]
    target_ep = target_rewards[:, ep_index, :]
    _, num_trajs = pred_ep.shape

    ep_valid = (
        torch.isfinite(pred_ep)
        & torch.isfinite(target_ep)
        & (target_ep >= 0.0)
        & (target_ep <= 1.0)
    )

    # Avoid inf - inf in MarginRankingLoss. The clamp value is intentionally
    # conservative because only the ordering of EP logits matters here.
    pred_ep = pred_ep.clamp(-20.0, 20.0)

    pred_i = pred_ep.unsqueeze(2).expand(-1, -1, num_trajs)
    pred_j = pred_ep.unsqueeze(1).expand(-1, num_trajs, -1)
    target_i = target_ep.unsqueeze(2).expand(-1, -1, num_trajs)
    target_j = target_ep.unsqueeze(1).expand(-1, num_trajs, -1)
    valid_i = ep_valid.unsqueeze(2).expand(-1, -1, num_trajs)
    valid_j = ep_valid.unsqueeze(1).expand(-1, num_trajs, -1)

    rank_targets = torch.sign(target_i - target_j)

    # Use each pair once and ignore ties in GT EP, because MarginRankingLoss
    # expects target values to be either 1 or -1.
    upper_tri_mask = torch.triu(
        torch.ones(num_trajs, num_trajs, dtype=torch.bool, device=predicted_rewards_logits.device),
        diagonal=1,
    ).unsqueeze(0)
    valid_pair_mask = upper_tri_mask & valid_i & valid_j & (rank_targets != 0) & torch.isfinite(rank_targets)

    if not valid_pair_mask.any():
        return predicted_rewards_logits.new_tensor(0.0)

    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction='mean')
    return ranking_loss_fn(
        pred_i[valid_pair_mask],
        pred_j[valid_pair_mask],
        rank_targets[valid_pair_mask],
    )

def compute_RFWorld_agent_loss(
    gt_states, 
    gt_valid,
    pred_states,
    pred_logits,
    config,
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    # gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    # pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    # visualize_agent_predictions(pred_states, gt_states, pred_logits, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    agent_class_weight = config.agent_class_weight if hasattr(config, "agent_class_weight") else 0.0
    agent_box_weight = config.agent_box_weight if hasattr(config, "agent_box_weight") else 0.0
    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Numerically stable Focal Loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_weight = (1.0 - pt) ** self.gamma

        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            else:
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
                alpha_weight = alpha_t[targets]
                focal_loss = alpha_weight * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
