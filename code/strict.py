"""Strict evaluation helpers (2026-10-31).

- Leave-one-person-out (LOPO) with everything fitted on the training people only:
  median imputation, scaling and the regularisation strength (chosen by an inner
  LOPO on the training people). No value from the test person is used before predicting.
- Cluster bootstrap by person for 95% confidence intervals (people, not reps, are resampled,
  because reps of the same person are not independent).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = (0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _clf(c):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _reg(a):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a))


def _inner_pick(X, y, groups, kind):
    best, best_v = None, -np.inf
    for h in (CS if kind == "clf" else ALPHAS):
        pred = np.full(len(y), np.nan)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            if kind == "clf" and len(np.unique(y[tr])) < 2:
                continue
            m = (_clf(h) if kind == "clf" else _reg(h)).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te])
        ok = np.isfinite(pred)
        if kind == "clf":
            v = roc_auc_score(y[ok], pred[ok]) if len(np.unique(y[ok])) == 2 else -np.inf
        else:
            v = spearmanr(pred[ok], y[ok])[0]
        if v > best_v:
            best, best_v = h, v
    return best


def lopo_predict(X, y, groups, kind="clf"):
    """Strict LOPO predictions. kind='clf' -> P(correct); 'reg' -> predicted score."""
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    X = np.where(np.isfinite(X), X, np.nan)
    pred, chosen = np.full(len(y), np.nan), []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        h = _inner_pick(X[tr], y[tr], groups[tr], kind)
        chosen.append(h)
        m = (_clf(h) if kind == "clf" else _reg(h)).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te])
    return pred, chosen


def cluster_ci(metric, y, pred, groups, n=2000, seed=0):
    """95% CI of metric(y, pred) by resampling people with replacement."""
    rng = np.random.default_rng(seed)
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    ids = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ids}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(ids, len(ids), replace=True)])
        try:
            v = metric(y[take], pred[take])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def rho(y, p):
    return spearmanr(p, y)[0]
