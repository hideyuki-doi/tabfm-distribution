"""Evaluation metrics and statistics used by Valavi et al. (2022, Ecol. Monogr. 92:e01486).

- AUC_PRG: area under the precision-recall-gain curve; port of the R package `prg`
  (Flach & Kull 2015; github.com/meeliskull/prg, R_package/prg/R/prg.R).
- Friedman aligned-rank test and its post-hoc pairwise test with Shaffer's static
  correction (García et al. 2010), as in the R package `scmamp`.
"""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata


# ---------------------------------------------------------------------------
# AUC_PRG
# ---------------------------------------------------------------------------
def _precision_gain(TP, FN, FP, TN):
    n_pos, n_neg = TP + FN, FP + TN
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 1 - (n_pos * FP) / (n_neg * TP)
    return np.where(TN + FN == 0, 0.0, g)


def _recall_gain(TP, FN, FP, TN):
    n_pos, n_neg = TP + FN, FP + TN
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 1 - (n_pos * FN) / (n_neg * TP)
    return np.where(TN + FN == 0, 1.0, g)


def _gains(P):
    P["precision_gain"] = _precision_gain(P["TP"], P["FN"], P["FP"], P["TN"])
    P["recall_gain"] = _recall_gain(P["TP"], P["FN"], P["FP"], P["TN"])
    return P


def create_prg_curve(labels, pos_scores):
    labels = (np.asarray(labels) == 1).astype(int)
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = -pos_scores
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    # segments: groups of tied scores, in decreasing score order
    order = np.lexsort((neg_scores, -pos_scores))
    s, lab = pos_scores[order], labels[order]
    new = np.r_[True, s[1:] != s[:-1]]
    seg = np.cumsum(new) - 1
    pos_count = np.bincount(seg, weights=lab)
    neg_count = np.bincount(seg, weights=1 - lab)
    P = pd.DataFrame({
        "index": np.arange(1, len(pos_count) + 2, dtype=float),
        "TP": np.r_[0, np.cumsum(pos_count)],
        "FP": np.r_[0, np.cumsum(neg_count)],
    })
    P["FN"] = n_pos - P["TP"]
    P["TN"] = n_neg - P["FP"]
    P = _gains(P)
    cols = ["index", "TP", "FP", "FN", "TN"]
    # crossing point through the y-axis (recall gain = 0)
    j = int(np.argmax(P["recall_gain"].values >= 0))
    if P["recall_gain"].iloc[j] > 0 and j > 0:
        d = P.loc[j, cols] - P.loc[j - 1, cols]
        alpha = (n_pos * n_pos / (n_pos + n_neg) - P["TP"].iloc[j - 1]) / d["TP"] if d["TP"] > 0 else 0.5
        newp = P.loc[j - 1, cols] + alpha * d
        newp = _gains(pd.DataFrame([newp]))
        newp["recall_gain"] = 0.0
        P = pd.concat([P, newp], ignore_index=True).sort_values("index", kind="stable").reset_index(drop=True)
    # crossings through the non-negative part of the x-axis (precision gain = 0)
    x, y = P["recall_gain"].values, P["precision_gain"].values
    cross = []
    yy = np.r_[y, 0] * np.r_[0, y]
    xx = np.r_[1, x]
    for i in np.where((yy < 0) & (xx >= 0))[0]:
        if i == 0 or i >= len(P):
            continue
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        d = P.loc[i, cols] - P.loc[i - 1, cols]
        n = n_pos + n_neg
        if d["TP"] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - P["TP"].iloc[i - 1]) / d["TP"]
        else:
            alpha = (n_neg / n_pos * P["TP"].iloc[i - 1] - P["FP"].iloc[i - 1]) / d["FP"]
        newp = _gains(pd.DataFrame([P.loc[i - 1, cols] + alpha * d]))
        newp["precision_gain"] = 0.0
        cross.append(newp)
    if cross:
        P = pd.concat([P] + cross, ignore_index=True)
    P = P.sort_values(["index", "recall_gain"], kind="stable").reset_index(drop=True)
    return P


def calc_auprg(P):
    rg, pg = P["recall_gain"].values, P["precision_gain"].values
    area = 0.0
    for i in range(1, len(P)):
        if not np.isnan(rg[i - 1]) and rg[i - 1] >= 0:
            area += (rg[i] - rg[i - 1]) * (pg[i] + pg[i - 1]) / 2
    return area


def auc_prg(labels, scores):
    return calc_auprg(create_prg_curve(labels, scores))


# ---------------------------------------------------------------------------
# Friedman aligned ranks + Shaffer
# ---------------------------------------------------------------------------
def friedman_aligned_ranks(data: pd.DataFrame):
    """data: rows = problems (species), columns = algorithms; higher = better.
    Returns (statistic, p_value, mean aligned rank per algorithm [low = better])."""
    X = data.values.astype(float)
    n, k = X.shape
    aligned = X - X.mean(axis=1, keepdims=True)
    # rank 1 = best (largest aligned value), average ties
    R = rankdata(-aligned.ravel()).reshape(n, k)
    Rj, Ri = R.sum(axis=0), R.sum(axis=1)
    num = (k - 1) * (np.sum(Rj ** 2) - (k * n ** 2 / 4.0) * (k * n + 1) ** 2)
    den = (k * n * (k * n + 1) * (2 * k * n + 1) / 6.0) - np.sum(Ri ** 2) / k
    stat = num / den
    return stat, chi2.sf(stat, k - 1), pd.Series(R.mean(axis=0), index=data.columns)


def _shaffer_sets(k, _cache={0: {0}, 1: {0}}):
    if k not in _cache:
        _cache[k] = {comb(j, 2) + x for j in range(1, k + 1) for x in _shaffer_sets(k - j)}
    return _cache[k]


def shaffer_adjust(raw_p: pd.DataFrame):
    """Shaffer's static correction for all pairwise comparisons (scmamp::adjustShaffer)."""
    k = raw_p.shape[0]
    pairs = list(combinations(range(k), 2))
    p = np.array([raw_p.values[i, j] for i, j in pairs])
    m = len(p)
    S = np.array(sorted(_shaffer_sets(k)))
    order = np.argsort(p)
    t = np.array([S[S <= m - i].max() for i in range(m)])  # i = 0..m-1  -> m - (i+1) + 1
    adj_sorted = np.minimum(1, np.maximum.accumulate(t * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    out = pd.DataFrame(np.nan, index=raw_p.index, columns=raw_p.columns)
    for (i, j), a in zip(pairs, adj):
        out.iat[i, j] = out.iat[j, i] = a
    return out


def posthoc_aligned_ranks(data: pd.DataFrame, correct="shaffer"):
    """Pairwise post-hoc for the Friedman aligned-rank test (García et al. 2010)."""
    n, k = data.shape
    _, _, mr = friedman_aligned_ranks(data)
    se = np.sqrt(k * (n * k + 1) / 6.0)  # as implemented in scmamp::friedmanAlignedRanksPost
    z = (mr.values[:, None] - mr.values[None, :]) / se
    raw = pd.DataFrame(2 * norm.sf(np.abs(z)), index=data.columns, columns=data.columns)
    return shaffer_adjust(raw) if correct == "shaffer" else raw
