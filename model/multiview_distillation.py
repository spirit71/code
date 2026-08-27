"""Training-only utilities for privileged multi-view place distillation."""

import torch
import torch.nn.functional as F


# 作用：构造地点原型与可靠性、计算原型/关系蒸馏，并对冲突梯度实施安全保护。
# 【相对 Baseline 新增】本模块只参与训练，不改变部署 Student 的网络层、参数量或推理路径。
@torch.no_grad()
def build_place_set_teacher(
        teacher_descriptors,
        num_places,
        views_per_place,
        view_temperature=0.1,
        reliability_threshold=0.2,
        reliability_margin_threshold=0.05,
        max_active_fraction=0.75,
        reliability_power=2.0):
    # This function is called from inside the training autocast region. Merely
    # converting its input to float32 is insufficient because autocast may cast
    # matrix operations back to a reduced precision. Keep all confidence and
    # prototype geometry explicitly in float32.
    with torch.autocast(
            device_type=teacher_descriptors.device.type, enabled=False):
        return _build_place_set_teacher_impl(
            teacher_descriptors.float(), num_places, views_per_place,
            view_temperature, reliability_threshold,
            reliability_margin_threshold, max_active_fraction,
            reliability_power)


def _build_place_set_teacher_impl(
        teacher_descriptors,
        num_places,
        views_per_place,
        view_temperature=0.1,
        reliability_threshold=0.2,
        reliability_margin_threshold=0.05,
        max_active_fraction=0.75,
        reliability_power=2.0):
    """Build robust global and leave-one-view-out teacher prototypes.

    The input order must match GSVCitiesDataset: all ``views_per_place`` images
    for a place are contiguous. A provisional mean identifies the views most
    consistent with the set; those views receive more weight in the final
    prototype. Each student view is supervised by a second prototype that
    excludes its matching teacher view, preventing same-image augmentation
    leakage. Reliability requires both cross-view consistency and separation
    from the nearest in-batch negative place.
    """
    if teacher_descriptors.ndim != 2:
        raise ValueError("teacher_descriptors must have shape [B * N, D]")
    if num_places < 1 or views_per_place < 2:
        raise ValueError("multi-view distillation requires B >= 1 and N >= 2")
    if teacher_descriptors.shape[0] != num_places * views_per_place:
        raise ValueError(
            "descriptor count does not match num_places * views_per_place: "
            f"{teacher_descriptors.shape[0]} != {num_places} * {views_per_place}")
    if view_temperature <= 0:
        raise ValueError("view_temperature must be positive")
    if not -1.0 <= reliability_threshold < 1.0:
        raise ValueError("reliability_threshold must be in [-1, 1)")
    if not -1.0 <= reliability_margin_threshold < 1.0:
        raise ValueError(
            "reliability_margin_threshold must be in [-1, 1)")
    if not 0.0 < max_active_fraction <= 1.0:
        raise ValueError("max_active_fraction must be in (0, 1]")
    if reliability_power <= 0:
        raise ValueError("reliability_power must be positive")

    # [B*N,D] 重排为 [B,N,D]，按视图与临时中心的一致性加权形成地点原型。
    # Compute teacher geometry in float32 even under autocast. The additional
    # memory is small relative to the model and avoids noisy confidence gates.
    views = F.normalize(
        teacher_descriptors.reshape(num_places, views_per_place, -1).float(),
        p=2, dim=-1)
    provisional = F.normalize(views.mean(dim=1), p=2, dim=-1)
    agreement = torch.einsum("bnd,bd->bn", views, provisional)
    view_weights = F.softmax(agreement / view_temperature, dim=1)
    prototypes = F.normalize(
        torch.einsum("bn,bnd->bd", view_weights, views), p=2, dim=-1)

    # 留一原型排除与 Student 对应的 Teacher 视图，防止退化为同图强弱增强捷径。
    # Target n is formed only from teacher views j != n. Recomputing the
    # provisional centre after exclusion makes this a genuine set-to-single
    # transfer rather than strong/weak self-view consistency.
    leave_one_out_centres = F.normalize(
        (views.sum(dim=1, keepdim=True) - views)
        / float(views_per_place - 1),
        p=2, dim=-1)
    leave_one_out_agreement = torch.einsum(
        "bjd,bnd->bnj", views, leave_one_out_centres)
    own_view_mask = torch.eye(
        views_per_place, device=views.device, dtype=torch.bool).unsqueeze(0)
    leave_one_out_agreement = leave_one_out_agreement.masked_fill(
        own_view_mask, torch.finfo(leave_one_out_agreement.dtype).min)
    leave_one_out_weights = F.softmax(
        leave_one_out_agreement / view_temperature, dim=-1)
    leave_one_out_prototypes = F.normalize(
        torch.einsum("bnj,bjd->bnd", leave_one_out_weights, views),
        p=2, dim=-1)

    # 地点内可靠性来自同地点视图的平均两两余弦相似度。
    pairwise = torch.bmm(views, views.transpose(1, 2))
    off_diagonal_sum = pairwise.sum(dim=(1, 2)) - pairwise.diagonal(
        dim1=1, dim2=2).sum(dim=1)
    mean_pairwise = off_diagonal_sum / (
        views_per_place * (views_per_place - 1))
    intra_reliability = (
        (mean_pairwise - reliability_threshold)
        / (1.0 - reliability_threshold)
    ).clamp(0.0, 1.0)

    if num_places == 1:
        nearest_negative_similarity = mean_pairwise.new_full((1,), -1.0)
    else:
        place_similarity = prototypes @ prototypes.transpose(0, 1)
        place_similarity.fill_diagonal_(
            torch.finfo(place_similarity.dtype).min)
        nearest_negative_similarity = place_similarity.max(dim=1).values
    # 判别可靠性 = 地点内多视角一致性 - 最近负地点相似度。
    discriminative_margin = mean_pairwise - nearest_negative_similarity
    margin_reliability = (
        (discriminative_margin - reliability_margin_threshold)
        / (1.0 - reliability_margin_threshold)
    ).clamp(0.0, 1.0)
    # The geometric combination retains the configured sharpening exponent
    # while requiring both conditions to be satisfied. With power=2 this is
    # the product of the two linearly calibrated confidence values.
    reliability = (intra_reliability * margin_reliability).pow(
        reliability_power / 2.0)
    # 【可靠性门控】仅保留 batch 内置信度最高的一部分地点，低质量 Teacher 权重归零。
    if max_active_fraction < 1.0 and num_places > 1:
        keep_count = max(1, int(num_places * max_active_fraction + 0.999999))
        keep_indices = reliability.topk(keep_count, largest=True).indices
        keep_mask = torch.zeros_like(reliability, dtype=torch.bool)
        keep_mask.scatter_(0, keep_indices, True)
        reliability = reliability * keep_mask

    diagnostics = {
        "mean_pairwise_similarity": mean_pairwise,
        "nearest_negative_similarity": nearest_negative_similarity,
        "discriminative_margin": discriminative_margin,
        "leave_one_out_weights": leave_one_out_weights,
    }
    return (prototypes, leave_one_out_prototypes, reliability, view_weights,
            diagnostics)


def privileged_multiview_distillation(
        student_descriptors,
        teacher_prototypes,
        leave_one_out_prototypes,
        place_reliability,
        views_per_place,
        temperature=0.2,
        relation_topk=16):
    # Preserve gradients through the student cast while preventing the outer
    # CUDA autocast context from reducing the precision of similarities/KL.
    with torch.autocast(
            device_type=student_descriptors.device.type, enabled=False):
        return _privileged_multiview_distillation_impl(
            student_descriptors.float(), teacher_prototypes.float(),
            leave_one_out_prototypes.float(), place_reliability.float(),
            views_per_place, temperature, relation_topk)


def _privileged_multiview_distillation_impl(
        student_descriptors,
        teacher_prototypes,
        leave_one_out_prototypes,
        place_reliability,
        views_per_place,
        temperature=0.2,
        relation_topk=16):
    # 【相对 Baseline 新增：两项蒸馏】单图对齐留一原型，并匹配困难负地点关系分布。
    """Distill place-set knowledge into independent single-image descriptors.

    ``prototype_loss`` pulls every student view toward a reliable prototype
    made only from the other teacher views of its place. ``relational_loss``
    transfers the teacher distribution over its hardest in-batch negative
    places, preventing easy negatives from dominating the objective.
    """
    if student_descriptors.ndim != 2 or teacher_prototypes.ndim != 2:
        raise ValueError("student descriptors and teacher prototypes must be 2-D")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if relation_topk <= 0:
        raise ValueError("relation_topk must be positive")

    num_places = teacher_prototypes.shape[0]
    if student_descriptors.shape[0] != num_places * views_per_place:
        raise ValueError("student descriptor count is incompatible with teacher prototypes")
    if leave_one_out_prototypes.shape != (
            num_places, views_per_place, teacher_prototypes.shape[-1]):
        raise ValueError(
            "leave_one_out_prototypes must have shape [B, N, D]")
    if place_reliability.shape != (num_places,):
        raise ValueError("place_reliability must have shape [B]")

    student = F.normalize(student_descriptors, p=2, dim=-1)
    prototypes = F.normalize(
        teacher_prototypes.detach(), p=2, dim=-1).to(
            device=student.device, dtype=student.dtype)
    view_targets = F.normalize(
        leave_one_out_prototypes.detach(), p=2, dim=-1).to(
            device=student.device, dtype=student.dtype).reshape(
                num_places * views_per_place, -1)
    sample_reliability = place_reliability.detach().repeat_interleave(
        views_per_place).to(dtype=student.dtype)
    # Normalize by sample count, not by reliability sum. Dividing by the
    # reliability sum would cancel a uniformly low confidence (e.g. changing
    # every reliability from 1.0 to 0.1), so an uncertain teacher would still
    # exert almost the same total force. A regular weighted mean makes
    # reliability control both relative place importance and global strength.
    normalizer = float(student.shape[0])

    positive_similarity = (student * view_targets).sum(dim=-1)
    prototype_loss = (
        (1.0 - positive_similarity) * sample_reliability).sum() / normalizer

    if num_places == 1:
        relational_loss = student.sum() * 0.0
    else:
        all_student_logits = student @ prototypes.transpose(0, 1)
        all_teacher_logits = prototypes @ prototypes.transpose(0, 1)
        all_teacher_logits.fill_diagonal_(
            torch.finfo(all_teacher_logits.dtype).min)
        effective_topk = min(relation_topk, num_places - 1)
        hard_negative_indices = all_teacher_logits.topk(
            effective_topk, dim=-1).indices
        sample_negative_indices = hard_negative_indices.repeat_interleave(
            views_per_place, dim=0)
        student_logits = all_student_logits.gather(
            dim=1, index=sample_negative_indices)
        teacher_logits = all_teacher_logits.gather(
            dim=1, index=hard_negative_indices).repeat_interleave(
                views_per_place, dim=0)

        teacher_probabilities = F.softmax(
            teacher_logits / temperature, dim=-1).detach()
        student_log_probabilities = F.log_softmax(
            student_logits / temperature, dim=-1)
        teacher_log_probabilities = torch.log(
            teacher_probabilities.clamp_min(torch.finfo(student.dtype).tiny))
        per_class_kl = (
            teacher_probabilities
            * (teacher_log_probabilities - student_log_probabilities)
        )
        per_sample_kl = per_class_kl.sum(dim=-1)
        relational_loss = (
            per_sample_kl * sample_reliability).sum() / normalizer
    return {
        "prototype_loss": prototype_loss,
        "relational_loss": relational_loss,
        "mean_reliability": place_reliability.mean(),
        "positive_place_fraction": (place_reliability > 0).float().mean(),
        "active_place_fraction": (place_reliability >= 0.05).float().mean(),
        "effective_place_fraction": (
            place_reliability.sum().square()
            / place_reliability.square().sum().clamp_min(
                torch.finfo(place_reliability.dtype).eps)
            / float(num_places)),
        "effective_relation_topk": student.new_tensor(
            0 if num_places == 1 else effective_topk),
    }


def conflict_safe_distillation(
        metric_loss,
        distillation_loss,
        student_descriptors,
        max_gradient_ratio=0.1):
    with torch.autocast(
            device_type=student_descriptors.device.type, enabled=False):
        return _conflict_safe_distillation_impl(
            metric_loss.float(), distillation_loss.float(),
            student_descriptors, max_gradient_ratio)


def _conflict_safe_distillation_impl(
        metric_loss,
        distillation_loss,
        student_descriptors,
        max_gradient_ratio=0.1):
    # 【相对 Baseline 新增：梯度安全层】冲突投影后再限制范数，保护原始度量学习方向。
    """Replace the distillation descriptor gradient with a safe direction.

    A conflicting component is projected away, then the remaining descriptor
    gradient is capped relative to the metric gradient. The returned scalar
    keeps the original distillation value for logging while carrying only the
    guarded gradient during backward.
    """
    if metric_loss.ndim != 0 or distillation_loss.ndim != 0:
        raise ValueError("metric and distillation losses must be scalars")
    if max_gradient_ratio <= 0:
        raise ValueError("max_gradient_ratio must be positive")

    metric_gradient = torch.autograd.grad(
        metric_loss, student_descriptors, retain_graph=True)[0].float()
    distillation_gradient = torch.autograd.grad(
        distillation_loss, student_descriptors, retain_graph=True)[0].float()
    # Project each view independently. A batch-global dot product can hide a
    # harmful view behind beneficial gradients from other views.
    metric_gradient_flat = metric_gradient.reshape(
        1 if metric_gradient.ndim == 1 else metric_gradient.shape[0], -1)
    distillation_gradient_flat = distillation_gradient.reshape_as(
        metric_gradient_flat)
    per_view_dot = (
        metric_gradient_flat * distillation_gradient_flat).sum(dim=1)
    per_view_metric_norm_sq = metric_gradient_flat.square().sum(dim=1)
    per_view_metric_norm = per_view_metric_norm_sq.sqrt()
    per_view_distillation_norm = distillation_gradient_flat.square().sum(dim=1).sqrt()
    epsilon = torch.finfo(torch.float32).eps
    per_view_gradient_cosine = per_view_dot / (
        per_view_metric_norm * per_view_distillation_norm).clamp_min(epsilon)

    projection_coefficient = (
        per_view_dot / per_view_metric_norm_sq.clamp_min(epsilon)
    ).clamp(max=0.0)
    safe_gradient_flat = (
        distillation_gradient_flat
        - projection_coefficient.unsqueeze(1) * metric_gradient_flat)
    safe_gradient = safe_gradient_flat.reshape_as(distillation_gradient)

    # Retain the conservative global norm cap after per-view projection.
    metric_norm = metric_gradient.square().sum().sqrt()
    projected_norm = safe_gradient.square().sum().sqrt()
    gradient_cap_scale = (
        max_gradient_ratio * metric_norm
        / projected_norm.clamp_min(epsilon)
    ).clamp(max=1.0)
    safe_gradient = safe_gradient * gradient_cap_scale

    gradient_surrogate = (
        student_descriptors.float() * safe_gradient.detach()).sum()
    safe_loss = (
        distillation_loss.detach()
        + gradient_surrogate
        - gradient_surrogate.detach())
    return safe_loss, {
        "gradient_cosine": per_view_gradient_cosine.mean().detach(),
        "minimum_gradient_cosine": per_view_gradient_cosine.min().detach(),
        "gradient_cap_scale": gradient_cap_scale.detach(),
        "gradient_conflict": (per_view_dot < 0).float().mean().detach(),
        "metric_gradient_norm": metric_norm.detach(),
        "safe_distillation_gradient_norm": safe_gradient.square().sum().sqrt().detach(),
    }


def get_distillation_scale(epoch, warmup_epochs, ramp_epochs):
    """Return the epoch-level coefficient for a warm-up plus linear ramp."""
    if epoch < 0 or warmup_epochs < 0 or ramp_epochs <= 0:
        raise ValueError(
            "epoch/warmup_epochs must be non-negative and ramp_epochs positive")
    if epoch < warmup_epochs:
        return 0.0
    return min((epoch - warmup_epochs + 1) / float(ramp_epochs), 1.0)


@torch.no_grad()
# 【相对 Baseline 新增】EMA 更新包含参数和缓冲区；Teacher 仅训练时存在。
def update_ema_teacher(teacher_model, student_model, decay):
    """Update a training-only teacher, including normalization buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")

    for teacher_parameter, student_parameter in zip(
            teacher_model.parameters(), student_model.parameters()):
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay)

    for teacher_buffer, student_buffer in zip(
            teacher_model.buffers(), student_model.buffers()):
        teacher_buffer.copy_(student_buffer.detach())
