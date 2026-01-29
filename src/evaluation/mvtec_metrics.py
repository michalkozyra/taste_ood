"""
MVTec AD evaluation utilities for per-pixel anomaly maps.

Implements:
- Image-level AUROC/AP (image score derived from anomaly map; default reducer=max).
- Pixel-level AUROC/AP.
- PRO / AUPRO (Per-Region Overlap) integrated over FPR in [0, fpr_limit].

Notes / conventions:
- Higher score == more anomalous.
- Masks are binary (0/1) at GT resolution.
- If your model produces a lower-res map, upsample to mask resolution before calling these metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import numpy as np


TailMode = Literal["upper", "two_sided"]

# region agent log
def _agent_log(payload: dict) -> None:
    """
    Append one NDJSON line to the debug log. Best-effort only.
    """
    try:
        import json  # noqa

        payload = dict(payload)
        payload["timestamp"] = 0
        # IMPORTANT: exact path per debug-mode config
        log_path = "/Users/michalkozyra/Developer/PhD/stein_shift_detection/.cursor/debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _agent_run_id() -> str:
    try:
        import os  # noqa

        return str(os.environ.get("AGENT_RUN_ID", "run1"))
    except Exception:
        return "run1"


# endregion agent log


def _validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape={m.shape}")
    return (m > 0).astype(np.uint8)


def percentile_thresholds(
    ref_scores: np.ndarray,
    *,
    alpha: float,
    tail: TailMode = "upper",
) -> tuple[Optional[float], float]:
    """
    Fit percentile-based threshold(s) from a 1D reference score sample.

    Conventions:
    - Higher score == more anomalous.
    - 'upper': reject if score >= q_{1-alpha}
    - 'two_sided': reject if score <= q_{alpha/2} OR score >= q_{1-alpha/2}

    Returns:
        (thr_low, thr_high)
        - For 'upper', thr_low is None and thr_high is the upper threshold.
        - For 'two_sided', both are finite thresholds.
    """
    _agent_log(
        {
            "sessionId": "debug-session",
            "runId": _agent_run_id(),
            "hypothesisId": "H1_nonnegative_maps_or_lower_tail_unused",
            "location": "src/evaluation/mvtec_metrics.py:percentile_thresholds:entry",
            "message": "percentile_thresholds entry",
            "data": {"alpha": float(alpha), "tail": str(tail), "ref_scores_shape": list(np.asarray(ref_scores).shape)},
        }
    )
    a = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("ref_scores must contain at least one finite value")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0,1). Got {alpha}")

    t = str(tail)
    if t == "upper":
        thr_high = float(np.quantile(a, 1.0 - float(alpha)))
        if not np.isfinite(thr_high):
            raise RuntimeError(f"Non-finite upper threshold. thr_high={thr_high}")
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H2_thresholds_nearly_identical",
                "location": "src/evaluation/mvtec_metrics.py:percentile_thresholds:upper",
                "message": "percentile_thresholds upper",
                "data": {
                    "alpha": float(alpha),
                    "tail": "upper",
                    "n": int(a.size),
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                    "q_1m_alpha": float(thr_high),
                },
            }
        )
        return None, thr_high
    if t == "two_sided":
        a2 = float(alpha) / 2.0
        thr_low = float(np.quantile(a, a2))
        thr_high = float(np.quantile(a, 1.0 - a2))
        if not (np.isfinite(thr_low) and np.isfinite(thr_high)):
            raise RuntimeError(f"Non-finite two-sided thresholds. thr_low={thr_low} thr_high={thr_high}")
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H2_thresholds_nearly_identical",
                "location": "src/evaluation/mvtec_metrics.py:percentile_thresholds:two_sided",
                "message": "percentile_thresholds two_sided",
                "data": {
                    "alpha": float(alpha),
                    "tail": "two_sided",
                    "n": int(a.size),
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                    "q_alpha_over_2": float(thr_low),
                    "q_1m_alpha_over_2": float(thr_high),
                },
            }
        )
        return thr_low, thr_high

    raise ValueError(f"Unsupported tail='{tail}'. Use 'upper' or 'two_sided'.")


def anomaly_mask_from_thresholds(
    scores_hw: np.ndarray,
    *,
    thr_low: Optional[float],
    thr_high: float,
    tail: TailMode = "upper",
) -> np.ndarray:
    """
    Convert an HxW score map into a binary anomaly mask using fitted percentile thresholds.
    Returns uint8 mask in {0,1}.
    """
    s = np.asarray(scores_hw, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores_hw must be 2D, got shape={s.shape}")
    t = str(tail)
    if t == "upper":
        pred = (s >= float(thr_high))
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H3_lower_tail_never_triggers",
                "location": "src/evaluation/mvtec_metrics.py:anomaly_mask_from_thresholds:upper",
                "message": "mask upper stats",
                "data": {
                    "tail": "upper",
                    "thr_high": float(thr_high),
                    "scores_min": float(np.min(s)),
                    "scores_max": float(np.max(s)),
                    "frac_ge_thr_high": float(np.mean(pred)),
                },
            }
        )
        return pred.astype(np.uint8)
    if t == "two_sided":
        if thr_low is None:
            raise ValueError("thr_low must be provided for two_sided tail")
        low = (s <= float(thr_low))
        high = (s >= float(thr_high))
        pred = (low | high)
        _agent_log(
            {
                "sessionId": "debug-session",
                "runId": _agent_run_id(),
                "hypothesisId": "H1_nonnegative_maps_or_lower_tail_unused",
                "location": "src/evaluation/mvtec_metrics.py:anomaly_mask_from_thresholds:two_sided",
                "message": "mask two_sided stats",
                "data": {
                    "tail": "two_sided",
                    "thr_low": float(thr_low),
                    "thr_high": float(thr_high),
                    "scores_min": float(np.min(s)),
                    "scores_max": float(np.max(s)),
                    "frac_le_thr_low": float(np.mean(low)),
                    "frac_ge_thr_high": float(np.mean(high)),
                    "frac_flagged": float(np.mean(pred)),
                },
            }
        )
        return pred.astype(np.uint8)
    raise ValueError(f"Unsupported tail='{tail}'. Use 'upper' or 'two_sided'.")


def percentile_oodness_from_reference(
    scores: np.ndarray,
    *,
    ref_scores: np.ndarray,
    tail: TailMode,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Percentile / p-value based score transform (higher => more anomalous).

    We compute an empirical CDF F_ref using ref_scores (finite values only).
    Then:
      - upper tail p(x) = P_ref(S >= x) = 1 - F_ref(x)
      - two-sided p(x) = 2 * min(F_ref(x), 1 - F_ref(x))
    And return oodness(x) = -log(max(p(x), eps)).

    This makes AUROC/AP tail-dependent (as requested).
    """
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(ref_scores, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("ref_scores must contain at least one finite value")
    r_sorted = np.sort(r, kind="mergesort")
    n = int(r_sorted.size)

    # Empirical CDF using searchsorted; F(x) in [0,1]
    x = s.reshape(-1)
    idx = np.searchsorted(r_sorted, x, side="right")
    F = idx.astype(np.float64) / float(n)

    t = str(tail)
    if t == "upper":
        p = 1.0 - F
    elif t == "two_sided":
        p = 2.0 * np.minimum(F, 1.0 - F)
    else:
        raise ValueError(f"Unsupported tail='{tail}'. Use 'upper' or 'two_sided'.")

    # Saturation / clamping diagnostics: if many pixels hit eps, ordering can collapse.
    p_min_raw = float(np.min(p)) if p.size else float("nan")
    p_max_raw = float(np.max(p)) if p.size else float("nan")
    frac_p_le_eps = float(np.mean(p <= float(eps))) if p.size else float("nan")
    frac_p_ge_1 = float(np.mean(p >= 1.0)) if p.size else float("nan")
    p = np.clip(p, float(eps), 1.0)
    out = (-np.log(p)).reshape(s.shape)

    _agent_log(
        {
            "sessionId": "debug-session",
            "runId": _agent_run_id(),
            "hypothesisId": "H4_auc_should_change_with_transform",
            "location": "src/evaluation/mvtec_metrics.py:percentile_oodness_from_reference",
            "message": "percentile_oodness transform",
            "data": {
                "tail": t,
                "eps": float(eps),
                "ref_n": int(r.size),
                "ref_min": float(np.min(r_sorted)),
                "ref_max": float(np.max(r_sorted)),
                "scores_shape": list(s.shape),
                "scores_min": float(np.nanmin(s)),
                "scores_max": float(np.nanmax(s)),
                "p_min_raw": float(p_min_raw),
                "p_max_raw": float(p_max_raw),
                "frac_p_le_eps": float(frac_p_le_eps),
                "frac_p_ge_1": float(frac_p_ge_1),
                "p_min": float(np.min(p)),
                "p_max": float(np.max(p)),
                "ood_min": float(np.min(out)),
                "ood_max": float(np.max(out)),
            },
        }
    )

    return out


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    """
    ROC curve assuming labels in {0,1} with 1=positive (anomaly) and higher score=more positive.
    Returns (fpr, tpr, thresholds_desc).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must be 1D arrays of same length")
    if scores.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = float(tps[-1])
    N = float(fps[-1])
    if P == 0 or N == 0:
        # Degenerate: only one class present
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    distinct_last = np.r_[s[1:] != s[:-1], True]
    tps = tps[distinct_last]
    fps = fps[distinct_last]
    thresh = s[distinct_last]
    fpr = fps / N
    tpr = tps / P

    # Ensure curve includes endpoints
    if fpr.size == 0 or fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.r_[0.0, fpr]
        tpr = np.r_[0.0, tpr]
        thresh = np.r_[np.inf, thresh]
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.r_[fpr, 1.0]
        tpr = np.r_[tpr, 1.0]
        thresh = np.r_[thresh, -np.inf]

    return fpr, tpr, thresh


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    if fpr.size < 2:
        return float("nan")
    return float(np.trapezoid(tpr, fpr))


def auc_upto_fpr_limit(
    fpr: np.ndarray, tpr: np.ndarray, *, fpr_limit: float = 0.30, normalize: bool = True
) -> float:
    """
    Area under ROC curve up to FPR <= fpr_limit.

    If normalize=True, returns area / fpr_limit (so result is in [0,1] assuming tpr in [0,1]).
    This mirrors the normalization used in compute_pro_auc (area/limit).
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    limit = float(fpr_limit)
    if fpr.size < 2 or limit <= 0:
        return float("nan")

    # Ensure sorted by FPR
    order = np.argsort(fpr, kind="mergesort")
    f = fpr[order]
    t = tpr[order]

    in_mask = f <= limit
    if not np.any(in_mask):
        return float("nan")
    f_in = f[in_mask]
    t_in = t[in_mask]

    # Add interpolated point at limit if needed
    if f_in[-1] < limit and np.any(~in_mask):
        j = int(np.argmax(~in_mask))
        f0, f1 = float(f_in[-1]), float(f[j])
        t0, t1 = float(t_in[-1]), float(t[j])
        if f1 > f0:
            w = (limit - f0) / (f1 - f0)
            t_lim = t0 + w * (t1 - t0)
        else:
            t_lim = t0
        f_in = np.r_[f_in, limit]
        t_in = np.r_[t_in, t_lim]
    elif f_in[-1] > limit:
        # Shouldn't happen given in_mask, but keep safe.
        f_in[-1] = limit

    area = float(np.trapezoid(t_in, f_in))
    return (area / limit) if normalize else area

def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Average precision (area under precision-recall curve) without sklearn.
    Higher score = more positive (anomaly).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    P = float(np.sum(y == 1))
    if P == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / P
    # integrate precision w.r.t recall (step-wise)
    # ensure recall starts at 0
    recall0 = np.r_[0.0, recall]
    prec0 = np.r_[precision[0], precision]
    # compute area with trapezoid on the step curve
    return float(np.trapezoid(prec0, recall0))


def average_precision_upto_fpr_limit(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fpr_limit: float = 0.30,
) -> float:
    """
    "Partial" AP over operating points with FPR <= fpr_limit.

    Implementation:
    - Sort by score descending (threshold moves from strict -> lenient).
    - Compute (precision, recall, fpr) at each prefix.
    - Keep points up to FPR<=limit; if needed, linearly interpolate one point at exactly FPR=limit.
    - Integrate precision w.r.t recall with trapezoid (same convention as average_precision()).

    Notes:
    - This is not the standard sklearn AP; it's a convenient "low-FPR AP" counterpart to AUROC@FPR<=limit.
    - Returns NaN if labels are degenerate or if no operating point reaches FPR<=limit.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        return float("nan")
    limit = float(fpr_limit)
    if limit <= 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    if P == 0 or N == 0:
        return float("nan")

    tp = np.cumsum(y == 1).astype(np.float64)
    fp = np.cumsum(y == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / P
    fpr = fp / N

    in_mask = fpr <= limit
    if not np.any(in_mask):
        return float("nan")

    # Keep points within limit.
    last = int(np.max(np.where(in_mask)[0]))
    prec_in = precision[: last + 1].copy()
    rec_in = recall[: last + 1].copy()
    fpr_in = fpr[: last + 1].copy()

    # If last point is below limit and we have a next point, interpolate at limit.
    if fpr_in[-1] < limit and (last + 1) < fpr.size:
        f0 = float(fpr_in[-1])
        f1 = float(fpr[last + 1])
        r0 = float(rec_in[-1])
        r1 = float(recall[last + 1])
        p0 = float(prec_in[-1])
        p1 = float(precision[last + 1])
        if f1 > f0:
            w = (limit - f0) / (f1 - f0)
            rec_lim = r0 + w * (r1 - r0)
            prec_lim = p0 + w * (p1 - p0)
        else:
            rec_lim = r0
            prec_lim = p0
        rec_in = np.r_[rec_in, rec_lim]
        prec_in = np.r_[prec_in, prec_lim]

    # Ensure recall starts at 0 for integration.
    rec0 = np.r_[0.0, rec_in]
    prec0 = np.r_[prec_in[0], prec_in]
    return float(np.trapezoid(prec0, rec0))


def fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    FPR at the first operating point where TPR >= target_tpr.
    (Common in OOD literature; no interpolation.)
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    idx = np.where(tpr >= float(target_tpr))[0]
    if idx.size == 0:
        return float("nan")
    return float(fpr[int(idx[0])])


def image_scores_from_maps(
    anomaly_maps: Iterable[np.ndarray],
    reducer: Literal["max", "mean", "p95"] = "max",
) -> np.ndarray:
    vals = []
    for m in anomaly_maps:
        a = np.asarray(m, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"anomaly_map must be HxW, got shape={a.shape}")
        if reducer == "max":
            vals.append(float(np.max(a)))
        elif reducer == "mean":
            vals.append(float(np.mean(a)))
        elif reducer == "p95":
            vals.append(float(np.percentile(a, 95.0)))
        else:
            raise ValueError(reducer)
    return np.asarray(vals, dtype=np.float64)


@dataclass(frozen=True)
class DetectionMetrics:
    auroc: float
    auroc_full: float
    ap: float
    ap_full: float
    fpr95: float


@dataclass(frozen=True)
class LocalizationMetrics:
    pixel_auroc: float
    pixel_ap: float
    aupro: float
    aupro_fpr_limit: float


def compute_image_level_metrics(
    *,
    anomaly_maps: list[np.ndarray],
    image_labels: np.ndarray,  # 0/1 per image
    reducer: Literal["max", "mean", "p95"] = "max",
    target_tpr: float = 0.95,
    fpr_limit: float = 0.30,
) -> DetectionMetrics:
    image_labels = np.asarray(image_labels, dtype=np.int64)
    scores = image_scores_from_maps(anomaly_maps, reducer=reducer)
    fpr, tpr, _ = _roc_curve(scores, image_labels)
    auroc_full = auc_from_roc(fpr, tpr)
    auroc_030 = auc_upto_fpr_limit(fpr, tpr, fpr_limit=float(fpr_limit), normalize=True)
    ap_full = average_precision(scores, image_labels)
    ap_030 = average_precision_upto_fpr_limit(scores, image_labels, fpr_limit=float(fpr_limit))
    return DetectionMetrics(
        auroc=float(auroc_030),
        auroc_full=float(auroc_full),
        ap=float(ap_030),
        ap_full=float(ap_full),
        fpr95=fpr_at_tpr(fpr, tpr, target_tpr=target_tpr),
    )


def compute_pixel_level_metrics(
    *,
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    pixel_subsample: Optional[int] = None,
    seed: int = 0,
    fpr_limit: float = 0.30,
) -> tuple[float, float, float, float]:
    """
    Pixel AUROC/AP. Optionally subsample pixels for memory/speed.
    """
    if len(anomaly_maps) != len(masks):
        raise ValueError("anomaly_maps and masks length mismatch")

    rng = np.random.default_rng(seed)
    all_scores = []
    all_labels = []
    for m, gt in zip(anomaly_maps, masks):
        a = np.asarray(m, dtype=np.float64)
        g = _validate_binary_mask(gt)
        if a.shape != g.shape:
            raise ValueError(f"shape mismatch: map {a.shape} vs mask {g.shape}")
        s = a.reshape(-1)
        y = g.reshape(-1).astype(np.int64)
        if pixel_subsample is not None and pixel_subsample > 0 and s.size > pixel_subsample:
            idx = rng.choice(s.size, size=int(pixel_subsample), replace=False)
            s = s[idx]
            y = y[idx]
        all_scores.append(s)
        all_labels.append(y)

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    fpr, tpr, _ = _roc_curve(scores, labels)
    auroc_full = auc_from_roc(fpr, tpr)
    auroc_030 = auc_upto_fpr_limit(fpr, tpr, fpr_limit=float(fpr_limit), normalize=True)
    ap_full = average_precision(scores, labels)
    ap_030 = average_precision_upto_fpr_limit(scores, labels, fpr_limit=float(fpr_limit))
    return float(auroc_030), float(ap_030), float(auroc_full), float(ap_full)


def compute_pixel_level_metrics_mean_over_images(
    *,
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    fpr_limit: float = 0.30,
    normalize: bool = True,
) -> tuple[float, float, float, float, int]:
    """
    Per-image pixel metrics, averaged across images (unweighted mean).

    Important: For a given image, pixel ROC/AP is only defined if the GT mask contains
    both positives and negatives. We skip degenerate images and compute the mean over
    the remaining ones.

    Returns:
        (mean_auroc_upto_limit, mean_ap, mean_auroc_full, n_valid_images)
    """
    if len(anomaly_maps) != len(masks):
        raise ValueError("anomaly_maps and masks length mismatch")

    aurocs_030 = []
    aurocs_full = []
    aps_030 = []
    aps_full = []
    n_valid = 0
    for m, gt in zip(anomaly_maps, masks):
        a = np.asarray(m, dtype=np.float64)
        g = _validate_binary_mask(gt)
        if a.shape != g.shape:
            raise ValueError(f"shape mismatch: map {a.shape} vs mask {g.shape}")
        y = g.reshape(-1).astype(np.int64)
        P = int(np.sum(y == 1))
        N = int(np.sum(y == 0))
        if P == 0 or N == 0:
            continue
        s = a.reshape(-1)
        fpr, tpr, _ = _roc_curve(s, y)
        aurocs_full.append(auc_from_roc(fpr, tpr))
        aurocs_030.append(auc_upto_fpr_limit(fpr, tpr, fpr_limit=float(fpr_limit), normalize=bool(normalize)))
        aps_full.append(average_precision(s, y))
        aps_030.append(average_precision_upto_fpr_limit(s, y, fpr_limit=float(fpr_limit)))
        n_valid += 1

    if n_valid == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    return (
        float(np.mean(aurocs_030)),
        float(np.mean(aps_030)),
        float(np.mean(aurocs_full)),
        float(np.mean(aps_full)),
        int(n_valid),
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Return list of boolean masks for each connected component in a binary mask.
    Uses scipy if available (fast). Falls back to a simple BFS otherwise.
    """
    m = _validate_binary_mask(mask)
    if m.sum() == 0:
        return []
    try:
        from scipy.ndimage import label as cc_label  # type: ignore
    except Exception:
        cc_label = None

    if cc_label is not None:
        labeled, n = cc_label(m)
        comps = [(labeled == (i + 1)) for i in range(int(n))]
        return comps

    # Fallback: naive BFS (4-connectivity). Slower, but avoids hard dependency.
    H, W = m.shape
    seen = np.zeros((H, W), dtype=np.uint8)
    comps = []
    for i in range(H):
        for j in range(W):
            if m[i, j] == 0 or seen[i, j]:
                continue
            q = [(i, j)]
            seen[i, j] = 1
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and m[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = 1
                        q.append((nx, ny))
            comp = np.zeros((H, W), dtype=bool)
            for x, y in pts:
                comp[x, y] = True
            comps.append(comp)
    return comps


def compute_pro_auc(
    *,
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    fpr_limit: float = 0.30,
    num_thresholds: int = 200,
    threshold_strategy: Literal["linspace", "quantile"] = "quantile",
    pixel_subsample_for_thresholds: int = 200_000,
    seed: int = 0,
) -> float:
    """
    Compute AUPRO (area under PRO vs FPR curve) up to fpr_limit.

    PRO at a threshold:
      - For each GT connected component region R in each anomalous image:
          overlap = |pred ∩ R| / |R|
      - PRO is mean overlap across all GT regions.

    FPR at a threshold:
      - FP pixels / total negative pixels, across all test images.

    This follows common open-source implementations (PatchCore/anomalib-style).
    """
    fprs, pros = _compute_pro_curve(
        anomaly_maps=anomaly_maps,
        masks=masks,
        num_thresholds=int(num_thresholds),
        threshold_strategy=str(threshold_strategy),  # type: ignore[arg-type]
        pixel_subsample_for_thresholds=int(pixel_subsample_for_thresholds),
        seed=int(seed),
    )
    return _aupro_area_normalized(fprs, pros, limit=float(fpr_limit))


def compute_pro_auc_limits(
    *,
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    fpr_limit: float = 0.30,
    num_thresholds: int = 200,
    threshold_strategy: Literal["linspace", "quantile"] = "quantile",
    pixel_subsample_for_thresholds: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Convenience wrapper to compute both:
      - AUPRO@FPR<=fpr_limit (normalized by fpr_limit)
      - AUPRO@FPR<=1.0 (normalized by 1.0; "full")
    using a single PRO curve computation (much faster than calling compute_pro_auc twice).
    """
    fprs, pros = _compute_pro_curve(
        anomaly_maps=anomaly_maps,
        masks=masks,
        num_thresholds=int(num_thresholds),
        threshold_strategy=str(threshold_strategy),  # type: ignore[arg-type]
        pixel_subsample_for_thresholds=int(pixel_subsample_for_thresholds),
        seed=int(seed),
    )
    return (
        _aupro_area_normalized(fprs, pros, limit=float(fpr_limit)),
        _aupro_area_normalized(fprs, pros, limit=1.0),
    )


def _compute_pro_curve(
    *,
    anomaly_maps: list[np.ndarray],
    masks: list[np.ndarray],
    num_thresholds: int,
    threshold_strategy: str,
    pixel_subsample_for_thresholds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(anomaly_maps) != len(masks):
        raise ValueError("anomaly_maps and masks length mismatch")

    maps = [np.asarray(m, dtype=np.float64) for m in anomaly_maps]
    gts = [_validate_binary_mask(gt) for gt in masks]
    for a, g in zip(maps, gts):
        if a.shape != g.shape:
            raise ValueError(f"shape mismatch: map {a.shape} vs mask {g.shape}")

    rng = np.random.default_rng(int(seed))
    all_scores = np.concatenate([a.reshape(-1) for a in maps], axis=0)
    if all_scores.size > int(pixel_subsample_for_thresholds):
        idx = rng.choice(all_scores.size, size=int(pixel_subsample_for_thresholds), replace=False)
        score_sample = all_scores[idx]
    else:
        score_sample = all_scores

    if str(threshold_strategy) == "linspace":
        lo = float(np.min(score_sample))
        hi = float(np.max(score_sample))
        thresholds = np.linspace(lo, hi, int(num_thresholds), dtype=np.float64)
    elif str(threshold_strategy) == "quantile":
        qs = np.linspace(0.0, 1.0, int(num_thresholds), dtype=np.float64)
        thresholds = np.quantile(score_sample, qs)
    else:
        raise ValueError(threshold_strategy)

    total_neg = float(np.sum([np.sum(g == 0) for g in gts]))
    if total_neg <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    gt_regions_per_image: list[list[np.ndarray]] = []
    for g in gts:
        gt_regions_per_image.append(_connected_components(g))

    fprs = []
    pros = []
    for thr in thresholds:
        fp = 0.0
        overlaps = []
        for a, g, regions in zip(maps, gts, gt_regions_per_image):
            pred = (a >= thr)
            fp += float(np.sum(pred & (g == 0)))
            for r in regions:
                denom = float(np.sum(r))
                if denom <= 0:
                    continue
                overlaps.append(float(np.sum(pred & r)) / denom)
        fprs.append(fp / total_neg)
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs, kind="mergesort")
    return fprs[order], pros[order]


def _aupro_area_normalized(fprs: np.ndarray, pros: np.ndarray, *, limit: float) -> float:
    """
    Integrate PRO vs FPR up to `limit` and return area/limit (normalized).
    """
    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    if fprs.size == 0 or pros.size == 0 or fprs.size != pros.size:
        return float("nan")
    limit = float(limit)
    if limit <= 0:
        return float("nan")

    in_mask = fprs <= limit
    if not np.any(in_mask):
        return float("nan")

    f_in = fprs[in_mask]
    p_in = pros[in_mask]

    if f_in[-1] < limit and np.any(~in_mask):
        j = int(np.argmax(~in_mask))
        f0, f1 = float(f_in[-1]), float(fprs[j])
        p0, p1 = float(p_in[-1]), float(pros[j])
        if f1 > f0:
            w = (limit - f0) / (f1 - f0)
            p_lim = p0 + w * (p1 - p0)
        else:
            p_lim = p0
        f_in = np.r_[f_in, limit]
        p_in = np.r_[p_in, p_lim]

    area = float(np.trapezoid(p_in, f_in))
    return area / limit

