"""Ridge readout + cluster bootstrap (PROTOCOL_v3.md section 18).

Readout: ridge regression (lambda 1.0), standardized features, on the
mean-pooled block-4 representation; predict the 24 fact labels; macro-F1.
Split: 12 templates train / 8 templates evaluation, by template ID.
Uncertainty: 10,000 paired cluster-bootstrap replicates over template IDs
(descriptive only). Registered minimum effect: +3 macro-F1 points.

Features come from a trained model at the selected checkpoint; this module is a
pure function of (features, labels, template_ids) so it is frozen and testable
independent of any run.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

TRAIN_TEMPLATES = set(range(12))     # first 12 template IDs
EVAL_TEMPLATES = set(range(12, 20))  # last 8
RIDGE_LAMBDA = 1.0
N_BOOTSTRAP = 10_000
MIN_EFFECT_F1 = 3.0


def _fit_predict(feat, labels, tids):
    tr = np.array([t in TRAIN_TEMPLATES for t in tids])
    ev = ~tr
    scaler = StandardScaler().fit(feat[tr])
    Xtr, Xev = scaler.transform(feat[tr]), scaler.transform(feat[ev])
    model = Ridge(alpha=RIDGE_LAMBDA).fit(Xtr, labels[tr])
    pred = (model.predict(Xev) >= 0.5).astype(int)
    return pred, labels[ev], tids[ev]


def macro_f1_readout(feat: np.ndarray, labels: np.ndarray, tids: np.ndarray) -> float:
    """Point-estimate macro-F1 on the evaluation templates."""
    pred, ytrue, _ = _fit_predict(np.asarray(feat, float), labels, tids)
    return float(f1_score(ytrue, pred, average="macro", zero_division=0))


def cluster_bootstrap_ci(feat, labels, tids, *, seed: int, n: int = N_BOOTSTRAP,
                         ci: float = 0.95):
    """Paired cluster bootstrap over evaluation template IDs (resample templates,
    keep all their variants). Descriptive CI only."""
    feat = np.asarray(feat, float)
    pred, ytrue, ev_tids = _fit_predict(feat, labels, tids)
    eval_templates = np.unique(ev_tids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        chosen = rng.choice(eval_templates, size=eval_templates.size, replace=True)
        idx = np.concatenate([np.where(ev_tids == c)[0] for c in chosen])
        stats[b] = f1_score(ytrue[idx], pred[idx], average="macro", zero_division=0)
    lo = float(np.quantile(stats, (1 - ci) / 2))
    hi = float(np.quantile(stats, 1 - (1 - ci) / 2))
    return {"point": float(f1_score(ytrue, pred, average="macro", zero_division=0)),
            "ci_low": lo, "ci_high": hi, "n_bootstrap": n}


def paired_delta(feat_pre, feat_post, labels, tids, *, seed: int) -> dict:
    """Acceptance Check 4: a readout update must move the paired point estimate
    by >= +3 macro-F1 to count as measurable (section 18)."""
    f_pre = macro_f1_readout(feat_pre, labels, tids)
    f_post = macro_f1_readout(feat_post, labels, tids)
    delta = 100.0 * (f_post - f_pre)
    return {"f1_pre": f_pre, "f1_post": f_post,
            "delta_f1_points": delta, "measurable": delta >= MIN_EFFECT_F1}
