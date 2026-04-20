import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from kilosort import CCG
from kilosort.preprocessing import get_drift_matrix

logger = logging.getLogger(__name__)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def templates_from_wall(Wall, wPCA):
    wall = _to_numpy(Wall)
    wpca = _to_numpy(wPCA)
    templates = np.einsum("ncp,pt->nct", wall, wpca)
    return np.transpose(templates, (0, 2, 1))


def wall_from_templates(templates, wPCA, device=None):
    waves = _to_numpy(templates)
    wpca = _to_numpy(wPCA)
    wall = np.einsum("ntc,pt->ncp", waves, wpca).astype(np.float32)
    wall = torch.from_numpy(wall)
    if device is not None:
        wall = wall.to(device)
    return wall


def template_channel_norms(Wall):
    wall = _to_numpy(Wall)
    return np.linalg.norm(wall, axis=-1)


def template_peak_channels(Wall):
    return np.argmax(template_channel_norms(Wall), axis=1).astype(np.int32)


def template_positions_from_wall(Wall, xc, yc):
    peak_chans = template_peak_channels(Wall)
    return peak_chans, _to_numpy(xc)[peak_chans], _to_numpy(yc)[peak_chans]


def compute_feature_ind(Wall, nearest_chans):
    chan_norms = template_channel_norms(Wall)
    order = np.argsort(chan_norms, axis=1)[:, ::-1]
    return order[:, :nearest_chans].astype(np.uint32)


def build_spike_positions(clu, anchor_x, anchor_y):
    clu = np.asarray(clu, dtype=np.int32)
    xs = np.asarray(anchor_x, dtype=np.float32)[clu]
    ys = np.asarray(anchor_y, dtype=np.float32)[clu]
    return np.column_stack([xs, ys]).astype(np.float32)


def build_fingerprint_from_ops(ops):
    depths = ops.get("drift_fingerprint_depths", None)
    fingerprint = ops.get("drift_fingerprint", None)
    if depths is None or fingerprint is None:
        return None, None
    return _to_numpy(depths).astype(np.float32), _to_numpy(fingerprint).astype(np.float32)


def build_alignment_reference_from_ops(ops):
    return {
        "fingerprint_depths": np.asarray(
            _to_numpy(ops.get("drift_fingerprint_depths", np.zeros(0, dtype=np.float32))),
            dtype=np.float32,
        ),
        "fingerprint": np.asarray(
            _to_numpy(ops.get("drift_fingerprint", np.zeros((0, 0), dtype=np.float32))),
            dtype=np.float32,
        ),
        "fingerprint_centered": np.asarray(
            _to_numpy(ops.get("drift_fingerprint_centered", np.zeros((0, 0), dtype=np.float32))),
            dtype=np.float32,
        ),
        "yblk": np.asarray(
            _to_numpy(ops.get("yblk", np.zeros(0, dtype=np.float32))),
            dtype=np.float32,
        ),
    }


def _interp_fingerprint(depths_src, fingerprint_src, depths_dst, shift_um):
    shifted_depths = np.asarray(depths_src) + shift_um
    cols = []
    for col in range(fingerprint_src.shape[1]):
        cols.append(
            np.interp(
                depths_dst,
                shifted_depths,
                fingerprint_src[:, col],
                left=0.0,
                right=0.0,
            )
        )
    return np.stack(cols, axis=1)


def _interp_fingerprint_with_shifts(depths_src, fingerprint_src, depths_dst, shifts_um):
    source_depths = np.asarray(depths_dst, dtype=np.float32) - np.asarray(
        shifts_um, dtype=np.float32
    )
    cols = []
    for col in range(fingerprint_src.shape[1]):
        cols.append(
            np.interp(
                source_depths,
                depths_src,
                fingerprint_src[:, col],
                left=0.0,
                right=0.0,
            )
        )
    return np.stack(cols, axis=1).astype(np.float32)


def alignment_score(aligned_prior, current_fingerprint):
    if aligned_prior is None or current_fingerprint is None:
        return 0.0
    aligned_prior = np.asarray(aligned_prior, dtype=np.float32)
    current_fingerprint = np.asarray(current_fingerprint, dtype=np.float32)
    if aligned_prior.size == 0 or current_fingerprint.size == 0:
        return 0.0

    prior = aligned_prior - aligned_prior.mean(axis=0, keepdims=True)
    cur = current_fingerprint - current_fingerprint.mean(axis=0, keepdims=True)
    denom = (np.linalg.norm(prior) + 1e-6) * (np.linalg.norm(cur) + 1e-6)
    return float(np.sum(prior * cur) / denom)


def estimate_rigid_shift(
    prior_depths,
    prior_fingerprint,
    current_depths,
    current_fingerprint,
    binning_depth,
    max_shift_um=None,
):
    if prior_depths is None or prior_fingerprint is None:
        debug = {
            "candidate_shifts_um": np.array([0.0], dtype=np.float32),
            "scores": np.array([0.0], dtype=np.float32),
            "aligned_prior_fingerprint": None,
        }
        return 0.0, debug

    prior_depths = np.asarray(prior_depths, dtype=np.float32)
    current_depths = np.asarray(current_depths, dtype=np.float32)
    prior_fingerprint = np.asarray(prior_fingerprint, dtype=np.float32)
    current_fingerprint = np.asarray(current_fingerprint, dtype=np.float32)

    if max_shift_um is None:
        probe_span = float(current_depths.max() - current_depths.min())
        max_shift_um = max(100.0, min(probe_span / 3.0, 500.0))

    step_um = max(float(binning_depth), 1.0)
    candidate_shifts = np.arange(-max_shift_um, max_shift_um + step_um, step_um)

    cur = current_fingerprint - current_fingerprint.mean(axis=0, keepdims=True)
    cur_std = np.linalg.norm(cur) + 1e-6

    scores = np.zeros(candidate_shifts.size, dtype=np.float32)
    aligned = None
    for idx, shift_um in enumerate(candidate_shifts):
        shifted = _interp_fingerprint(prior_depths, prior_fingerprint, current_depths, shift_um)
        shifted = shifted - shifted.mean(axis=0, keepdims=True)
        denom = (np.linalg.norm(shifted) + 1e-6) * cur_std
        scores[idx] = float(np.sum(shifted * cur) / denom)

    best_idx = int(np.argmax(scores))
    best_shift = float(candidate_shifts[best_idx])
    aligned = _interp_fingerprint(prior_depths, prior_fingerprint, current_depths, best_shift)
    debug = {
        "candidate_shifts_um": candidate_shifts.astype(np.float32),
        "scores": scores.astype(np.float32),
        "aligned_prior_fingerprint": aligned.astype(np.float32),
        "best_score": float(scores[best_idx]),
    }
    return best_shift, debug


def warp_templates_rigid(prior_templates, ops, shift_um, device=None):
    templates = _to_numpy(prior_templates).astype(np.float32)
    if templates.size == 0:
        result = torch.zeros((0, templates.shape[-1], templates.shape[-2]), dtype=torch.float32)
        return result.to(device) if device is not None else result

    M = get_drift_matrix(ops, shift_um, device=device or ops["iKxx"].device)
    M_cpu = M.detach().cpu()
    warped = []
    for template in templates:
        chans_by_time = torch.from_numpy(template.T).float()
        warped.append((M_cpu @ chans_by_time).T.unsqueeze(0))
    warped = torch.cat(warped, dim=0)
    if device is not None:
        warped = warped.to(device)
    return warped


def build_nonrigid_block_grid(depths, nblocks):
    depths = np.asarray(depths, dtype=np.float32)
    if depths.size == 0:
        return np.zeros(0, dtype=np.float32)
    if nblocks <= 1:
        return np.array([float(depths.mean())], dtype=np.float32)

    nybins = depths.size
    yl = max(1, nybins // nblocks)
    ifirst = np.round(np.linspace(0, max(0, nybins - yl), 2 * nblocks - 1)).astype(np.int32)
    ilast = np.minimum(ifirst + yl, nybins)
    return np.array(
        [float(depths[start:stop].mean()) for start, stop in zip(ifirst, ilast)],
        dtype=np.float32,
    )


def estimate_nonrigid_shift(
    prior_bundle,
    current_depths,
    current_fingerprint,
    rigid_shift_um,
    ops,
):
    current_depths = np.asarray(current_depths, dtype=np.float32)
    current_fingerprint = np.asarray(current_fingerprint, dtype=np.float32)
    current_yblk = np.asarray(_to_numpy(ops.get("yblk", np.zeros(0, dtype=np.float32))), dtype=np.float32)
    settings = ops["settings"]

    prior_depths = prior_bundle.get("fingerprint_depths", None)
    prior_fingerprint = prior_bundle.get("fingerprint", None)
    prior_centered = prior_bundle.get("fingerprint_centered", None)
    prior_yblk = np.asarray(prior_bundle.get("yblk", np.zeros(0, dtype=np.float32)), dtype=np.float32)

    rigid_aligned = _interp_fingerprint(
        prior_depths,
        prior_fingerprint,
        current_depths,
        rigid_shift_um,
    )
    rigid_score = alignment_score(rigid_aligned, current_fingerprint)

    debug = {
        "rigid_score": float(rigid_score),
        "aligned_rigid_fingerprint": rigid_aligned.astype(np.float32),
        "accepted": False,
        "used_mode": "rigid",
        "fallback_reason": "",
        "nonrigid_block_depths": np.zeros(0, dtype=np.float32),
        "nonrigid_block_shifts_um": np.zeros(0, dtype=np.float32),
        "nonrigid_depth_shifts_um": np.zeros(0, dtype=np.float32),
        "aligned_nonrigid_fingerprint": rigid_aligned.astype(np.float32),
        "nonrigid_score": float(rigid_score),
    }

    if (
        prior_depths is None
        or prior_fingerprint is None
        or prior_centered is None
        or prior_yblk.size == 0
        or current_depths.size == 0
        or current_fingerprint.size == 0
    ):
        debug["fallback_reason"] = "missing_nonrigid_reference"
        return np.full(current_yblk.shape, rigid_shift_um, dtype=np.float32), debug

    if current_yblk.size == 0:
        debug["fallback_reason"] = "missing_current_yblk"
        return np.array([rigid_shift_um], dtype=np.float32), debug

    requested_nblocks = int(settings.get("closed_loop_nonrigid_nblocks", 0))
    effective_nblocks = requested_nblocks if requested_nblocks > 0 else max(int(ops.get("nblocks", 1)), 1)
    block_depths = build_nonrigid_block_grid(current_depths, effective_nblocks)
    if block_depths.size == 0:
        debug["fallback_reason"] = "empty_block_grid"
        return np.full(current_yblk.shape, rigid_shift_um, dtype=np.float32), debug

    residual_max = float(settings["closed_loop_nonrigid_max_shift_um"])
    residual_candidates = np.arange(
        -residual_max,
        residual_max + max(float(ops["binning_depth"]), 1.0),
        max(float(ops["binning_depth"]), 1.0),
        dtype=np.float32,
    )
    prior_for_local = np.asarray(prior_centered, dtype=np.float32)
    if prior_for_local.size == 0:
        prior_for_local = np.asarray(prior_fingerprint, dtype=np.float32)
    current_centered = current_fingerprint - current_fingerprint.mean(axis=0, keepdims=True)

    spacing = float(np.median(np.diff(block_depths))) if block_depths.size > 1 else float(
        max(np.median(np.diff(current_depths)) if current_depths.size > 1 else ops["binning_depth"], 1.0)
    )
    sigma_um = max(spacing * 1.5, float(ops["binning_depth"]) * 2.0)
    weights = np.exp(-0.5 * ((current_depths[:, np.newaxis] - block_depths[np.newaxis, :]) / sigma_um) ** 2)
    block_residuals = np.zeros(block_depths.size, dtype=np.float32)

    for block_idx in range(block_depths.size):
        w = weights[:, block_idx:block_idx + 1]
        cur_local = current_centered * w
        cur_norm = np.linalg.norm(cur_local) + 1e-6
        best_score = -np.inf
        best_residual = 0.0
        for residual_um in residual_candidates:
            aligned = _interp_fingerprint(
                prior_depths,
                prior_for_local,
                current_depths,
                rigid_shift_um + float(residual_um),
            )
            aligned_local = aligned * w
            score = float(np.sum(aligned_local * cur_local) / ((np.linalg.norm(aligned_local) + 1e-6) * cur_norm))
            if score > best_score:
                best_score = score
                best_residual = float(residual_um)
        block_residuals[block_idx] = best_residual

    smoothing = float(settings["closed_loop_nonrigid_smoothing"])
    if smoothing > 0 and block_residuals.size > 1:
        block_residuals = gaussian_filter1d(block_residuals, sigma=smoothing, mode="nearest")
    block_residuals = np.clip(
        block_residuals,
        -float(settings["closed_loop_nonrigid_max_residual_um"]),
        float(settings["closed_loop_nonrigid_max_residual_um"]),
    ).astype(np.float32)

    if block_depths.size == 1:
        residual_on_current_yblk = np.full(current_yblk.shape, block_residuals[0], dtype=np.float32)
        residual_on_depths = np.full(current_depths.shape, block_residuals[0], dtype=np.float32)
    else:
        residual_on_current_yblk = np.interp(
            current_yblk,
            block_depths,
            block_residuals,
            left=block_residuals[0],
            right=block_residuals[-1],
        ).astype(np.float32)
        residual_on_depths = np.interp(
            current_depths,
            block_depths,
            block_residuals,
            left=block_residuals[0],
            right=block_residuals[-1],
        ).astype(np.float32)

    total_depth_shifts = rigid_shift_um + residual_on_depths
    aligned_nonrigid = _interp_fingerprint_with_shifts(
        prior_depths,
        prior_fingerprint,
        current_depths,
        total_depth_shifts,
    )
    nonrigid_score = alignment_score(aligned_nonrigid, current_fingerprint)
    min_gain = float(settings["closed_loop_nonrigid_min_score_gain"])
    accepted = bool(nonrigid_score >= rigid_score + min_gain)

    debug.update({
        "nonrigid_block_depths": block_depths.astype(np.float32),
        "nonrigid_block_shifts_um": (rigid_shift_um + block_residuals).astype(np.float32),
        "nonrigid_depth_shifts_um": total_depth_shifts.astype(np.float32),
        "aligned_nonrigid_fingerprint": aligned_nonrigid.astype(np.float32),
        "nonrigid_score": float(nonrigid_score),
        "accepted": accepted,
        "used_mode": "nonrigid" if accepted else "rigid",
        "fallback_reason": "" if accepted else "insufficient_score_gain",
        "score_gain": float(nonrigid_score - rigid_score),
    })

    chosen_shift = (
        rigid_shift_um + residual_on_current_yblk
        if accepted else
        np.full(current_yblk.shape, rigid_shift_um, dtype=np.float32)
    )
    return np.asarray(chosen_shift, dtype=np.float32), debug


def warp_templates_nonrigid(prior_templates, ops, yblk, shifts_um, device=None):
    templates = _to_numpy(prior_templates).astype(np.float32)
    if templates.size == 0:
        result = torch.zeros((0, templates.shape[-1], templates.shape[-2]), dtype=torch.float32)
        return result.to(device) if device is not None else result

    target_yblk = np.asarray(yblk, dtype=np.float32)
    target_shifts = np.asarray(shifts_um, dtype=np.float32)
    if target_shifts.ndim == 0:
        return warp_templates_rigid(templates, ops, float(target_shifts), device=device)

    warp_ops = ops.copy()
    warp_ops["yblk"] = target_yblk
    warp_ops["nblocks"] = int(max(1, target_yblk.size))
    M = get_drift_matrix(warp_ops, target_shifts, device=device or ops["iKxx"].device)
    M_cpu = M.detach().cpu()
    warped = []
    for template in templates:
        chans_by_time = torch.from_numpy(template.T).float()
        warped.append((M_cpu @ chans_by_time).T.unsqueeze(0))
    warped = torch.cat(warped, dim=0)
    if device is not None:
        warped = warped.to(device)
    return warped


def load_prior_bundle(path):
    path = Path(path)
    bundle = np.load(path, allow_pickle=True)
    metadata_raw = bundle["metadata_json"]
    if isinstance(metadata_raw, np.ndarray):
        metadata_raw = metadata_raw.item()
    metadata = json.loads(str(metadata_raw))
    result = {k: bundle[k] for k in bundle.files if k != "metadata_json"}
    result["metadata"] = metadata
    result["path"] = str(path)
    return result


def next_loop_index(prior_bundle):
    if prior_bundle is None:
        return 0
    return int(prior_bundle["metadata"].get("loop_index", 0)) + 1


def select_matching_prior_mask(prior_bundle):
    global_ids = np.asarray(prior_bundle.get("global_unit_ids", np.zeros(0, dtype=np.int32)))
    n_prior = global_ids.size
    if n_prior == 0:
        return np.zeros(0, dtype=bool)

    status = np.asarray(
        prior_bundle.get("status", np.array(["active"] * n_prior, dtype="U16"))
    ).astype("U16")
    keep_status = status != "dropped"

    is_ref = np.asarray(
        prior_bundle.get("is_ref", np.ones(n_prior, dtype=np.float32)),
        dtype=np.float32,
    ) > 0.5
    if np.any(keep_status & is_ref):
        return keep_status & is_ref

    est_contam = np.asarray(
        prior_bundle.get("est_contam_rate", np.zeros(n_prior, dtype=np.float32)),
        dtype=np.float32,
    )
    low_contam = est_contam <= 0.2
    if np.any(keep_status & low_contam):
        return keep_status & low_contam

    return keep_status


def combined_similarity(Wall_a, Wall_b, wPCA):
    if len(Wall_a) == 0 or len(Wall_b) == 0:
        return np.zeros((len(Wall_a), len(Wall_b)), dtype=np.float32)
    wall_a = Wall_a.detach().cpu()
    wall_b = Wall_b.detach().cpu()
    combined = torch.cat([wall_a, wall_b], dim=0)
    sims = CCG.similarity(combined, wPCA.detach().cpu().contiguous(), nt=wPCA.shape[1])
    n_a = wall_a.shape[0]
    return sims[:n_a, n_a:].astype(np.float32)


def spike_overlap_matrix(prior_st, discovery_st, discovery_clu, n_prior, n_disc, tol=1):
    overlaps = np.zeros((n_prior, n_disc), dtype=np.float32)
    if n_prior == 0 or n_disc == 0:
        return overlaps

    prior_times = prior_st[:, 0].astype(np.int64)
    prior_ids = prior_st[:, 1].astype(np.int32)
    disc_times = discovery_st[:, 0].astype(np.int64)
    disc_ids = discovery_clu.astype(np.int32)

    for p in range(n_prior):
        pt = np.sort(prior_times[prior_ids == p])
        if pt.size == 0:
            continue
        for d in range(n_disc):
            dt = np.sort(disc_times[disc_ids == d])
            if dt.size == 0:
                continue
            i = 0
            j = 0
            matches = 0
            while i < pt.size and j < dt.size:
                diff = pt[i] - dt[j]
                if abs(diff) <= tol:
                    matches += 1
                    i += 1
                    j += 1
                elif diff < 0:
                    i += 1
                else:
                    j += 1
            overlaps[p, d] = matches / max(1, min(pt.size, dt.size))
    return overlaps


def reconcile_discovery_units(
    prior_wall,
    discovery_wall,
    prior_st,
    discovery_st,
    discovery_clu,
    prior_anchor_y,
    discovery_anchor_y,
    ops,
):
    n_prior = len(prior_wall)
    n_disc = len(discovery_wall)
    if n_disc == 0:
        return {
            "duplicate_pairs": {},
            "kept_new_units": np.zeros(0, dtype=bool),
            "similarity": np.zeros((n_prior, 0), dtype=np.float32),
            "overlap": np.zeros((n_prior, 0), dtype=np.float32),
        }

    similarity = combined_similarity(prior_wall, discovery_wall, ops["wPCA"])
    overlap = spike_overlap_matrix(prior_st, discovery_st, discovery_clu, n_prior, n_disc)
    prior_anchor_y = np.asarray(prior_anchor_y, dtype=np.float32)
    discovery_anchor_y = np.asarray(discovery_anchor_y, dtype=np.float32)
    depth_diff = np.abs(prior_anchor_y[:, np.newaxis] - discovery_anchor_y[np.newaxis, :])
    max_depth_diff = max(float(ops.get("dmin", 20)) * 2.0, 30.0)

    duplicate_pairs = {}
    kept_new = np.ones(n_disc, dtype=bool)

    for d in range(n_disc):
        best_prior = int(np.argmax(similarity[:, d])) if n_prior else -1
        if best_prior < 0:
            continue
        sim = similarity[best_prior, d]
        ov = overlap[best_prior, d]
        depth_ok = depth_diff[best_prior, d] <= max_depth_diff
        if depth_ok and ((sim >= 0.9) or (sim >= 0.8 and ov >= 0.05)):
            duplicate_pairs[d] = {
                "prior_index": best_prior,
                "similarity": float(sim),
                "overlap": float(ov),
                "depth_diff_um": float(depth_diff[best_prior, d]),
            }
            kept_new[d] = False

    return {
        "duplicate_pairs": duplicate_pairs,
        "kept_new_units": kept_new,
        "similarity": similarity,
        "overlap": overlap,
    }


def adapt_carried_templates(
    aligned_templates,
    aligned_wall,
    prior_counts,
    duplicate_pairs,
    discovery_wall,
    alpha,
    min_spikes,
    wPCA,
    device=None,
):
    templates = _to_numpy(aligned_templates).copy()
    discovery_templates = templates_from_wall(discovery_wall, wPCA)
    adapted = np.zeros(prior_counts.size, dtype=bool)
    adaptation_source = np.full(prior_counts.size, -1, dtype=np.int32)
    for disc_idx, info in duplicate_pairs.items():
        prior_idx = info["prior_index"]
        if prior_counts[prior_idx] < min_spikes:
            continue
        templates[prior_idx] = (
            (1.0 - alpha) * templates[prior_idx]
            + alpha * discovery_templates[disc_idx]
        ).astype(np.float32)
        adapted[prior_idx] = True
        adaptation_source[prior_idx] = disc_idx
    adapted_wall = wall_from_templates(templates, wPCA, device=device)
    return templates, adapted_wall, adapted, adaptation_source


def save_prior_bundle(
    path,
    ops,
    Wall,
    global_ids,
    is_ref,
    est_contam_rate,
    results_dir,
    status=None,
    source_loop=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    global_ids = np.asarray(global_ids, dtype=np.int32)
    templates = templates_from_wall(Wall, ops["wPCA"]).astype(np.float32)
    peak_channel, anchor_x, anchor_y = template_positions_from_wall(Wall, ops["xc"], ops["yc"])
    template_amp = np.linalg.norm(_to_numpy(Wall), axis=(1, 2)).astype(np.float32)
    alignment_reference = build_alignment_reference_from_ops(ops)

    status = np.asarray(
        status if status is not None else np.array(["active"] * len(global_ids), dtype="U16")
    )
    source_loop = np.asarray(
        source_loop if source_loop is not None else np.full(len(global_ids), ops.get("closed_loop_loop_index", 0))
    )

    metadata = {
        "loop_index": int(ops.get("closed_loop_loop_index", 0)),
        "identity_mode": ops["settings"].get("closed_loop_identity_mode", "off"),
        "results_dir": str(results_dir),
        "alignment_reference_nblocks": int(alignment_reference["yblk"].size),
    }

    np.savez(
        path,
        global_unit_ids=global_ids,
        templates=templates,
        peak_channel=peak_channel.astype(np.int32),
        anchor_x=np.asarray(anchor_x, dtype=np.float32),
        anchor_y=np.asarray(anchor_y, dtype=np.float32),
        template_amplitude=template_amp,
        is_ref=np.asarray(is_ref, dtype=np.float32),
        est_contam_rate=np.asarray(est_contam_rate, dtype=np.float32),
        status=status.astype("U16"),
        source_loop=np.asarray(source_loop, dtype=np.int32),
        probe_xc=_to_numpy(ops["xc"]).astype(np.float32),
        probe_yc=_to_numpy(ops["yc"]).astype(np.float32),
        fingerprint_depths=np.asarray(
            alignment_reference["fingerprint_depths"],
            dtype=np.float32,
        ),
        fingerprint=np.asarray(
            alignment_reference["fingerprint"],
            dtype=np.float32,
        ),
        fingerprint_centered=np.asarray(
            alignment_reference["fingerprint_centered"],
            dtype=np.float32,
        ),
        yblk=np.asarray(
            alignment_reference["yblk"],
            dtype=np.float32,
        ),
        metadata_json=json.dumps(metadata),
    )


def save_cross_loop_outputs(results_dir, ops):
    results_dir = Path(results_dir)
    map_rows = ops.get("closed_loop_map_rows", None)
    if map_rows is None:
        return

    fieldnames = [
        "local_cluster_id",
        "global_unit_id",
        "source_loop",
        "status",
        "match_confidence",
        "adapted",
        "n_spikes_matched",
        "duplicate_similarity",
        "duplicate_overlap",
    ]
    with open(results_dir / "cross_loop_unit_map.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in map_rows:
            writer.writerow(row)

    summary = ops.get("closed_loop_summary", {})
    with open(results_dir / "cross_loop_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    debug = ops.get("closed_loop_debug", {})
    if debug:
        np.savez(results_dir / "cross_loop_debug.npz", **debug)


def default_closed_loop_state(ops):
    ops["closed_loop_enabled"] = False
    ops["closed_loop_loop_index"] = 0
    ops["closed_loop_rigid_shift_um"] = 0.0
    ops["closed_loop_num_carried"] = 0
    ops["closed_loop_num_new"] = 0
    ops["closed_loop_num_stale"] = 0
    return ops
