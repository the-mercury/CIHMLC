"""
Calibration and Uncertainty Analysis

Edit the CONFIG block below, then run this file in PyCharm.
It will:
1) Fit Temperature Scaling (global T) on validation logits or inferred logits.
2) Compute ECE, NLL, Brier on the test set pre and post calibration.
3) Save reliability diagrams for selected labels.
4) Optionally compute hierarchical metrics (CVR and conditional ECE) if a hierarchy JSON is provided.
5) Optionally summarize Monte Carlo Dropout predictions if provided.

Expected arrays:
- y_val:  shape [N_val, L], binary {0,1}
- p_val:  shape [N_val, L], logits by default or probabilities if FROM_PROBS is True
- y_test: shape [N_test, L], binary {0,1}
- p_test: shape [N_test, L], logits by default or probabilities if FROM_PROBS is True
- Optional MCD: npz with key 'P' of shape [K, N_test, L]

Notes:
- labels.txt should contain exactly L lines, one label per line.
- hierarchy.json example: { "Abnormal": ["Effusion","Consolidation"], "Opacity": ["Infiltration","Pneumonia"] }
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config

# -------------------------
# CONFIG: edit these paths
# -------------------------
IS_BASELINE = False

if IS_BASELINE:
    YVAL_PATH = 'default_model_files/scores/test_val_scores_baseline/y_true_val.npy'
    PVAL_PATH = 'default_model_files/scores/test_val_scores_baseline/y_pred_val.npy'  # probs by default; set FROM_PROBS if logits
    YTEST_PATH = 'default_model_files/scores/test_val_scores_baseline/y_true_test.npy'
    PTEST_PATH = 'default_model_files/scores/test_val_scores_baseline/y_pred_test.npy'  # probs by default; set FROM_PROBS if logits

else:
    YVAL_PATH = 'default_model_files/scores/test_val_scores/y_true_val.npy'
    PVAL_PATH = 'default_model_files/scores/test_val_scores/y_pred_val.npy'  # probs by default; set FROM_PROBS if logits
    YTEST_PATH = 'default_model_files/scores/test_val_scores/y_true_test.npy'
    PTEST_PATH = 'default_model_files/scores/test_val_scores/y_pred_test.npy'  # probs by default; set FROM_PROBS if logits

FROM_PROBS = True  # True if PVAL_PATH and PTEST_PATH contain probabilities
LABELS_PATH = 'default_model_files/scores/test_val_scores/labels.txt'  # e.g., 'path/to/labels.txt' or None
HIERARCHY_JSON = 'default_model_files/scores/test_val_scores/hierarchy.json'  # e.g., 'path/to/hierarchy.json' or None
MCD_PATH = None  # e.g., 'path/to/mcd_test.npz' with key 'P': [K,N,L] or None

N_BINS = 15
SELECT_LABELS = 21
INIT_T = 1.0
TS_STEPS = 1000
TS_LR = 0.05
TS_TOL = 1e-7
CVR_THRESHOLD = 0.5
OUTPATH = f'calibration_results_{N_BINS}_bins_baseline' if IS_BASELINE else f'calibration_results_{N_BINS}_bins'
OUTDIR = os.path.join(Config.CALIBRATION_ANALYSIS_DIR, OUTPATH)
PLOTS_DIR = os.path.join(OUTDIR, 'plots')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------

def ensure_dir(d: str) -> None:
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def bce(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    return float(bce(y, p).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((y - p) ** 2).mean())


def ece_single(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for a single label vector.
    y and p are 1D arrays of the same length.
    """
    conf = p.reshape(-1)
    y_flat = y.reshape(-1)
    pred_flat = (conf >= 0.5).astype(np.float32)
    acc_flat = (pred_flat == y_flat).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        conf_bin = float(conf[m].mean())
        acc_bin = float(acc_flat[m].mean())
        w = float(m.mean())
        ece_val += w * abs(acc_bin - conf_bin)
    return float(ece_val)


# def per_label_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Tuple[List[dict], dict]:
#     L = y.shape[1]
#     rows = []
#     for j in range(L):
#         rows.append({
#             'label_index': j,
#             'ECE': ece_single(y[:, j], p[:, j], n_bins=n_bins),
#             'NLL': nll(y[:, j], p[:, j]),
#             'Brier': brier(y[:, j], p[:, j]),
#         })
#     macro = {
#         'label_index': 'macro',
#         'ECE': float(np.mean([r['ECE'] for r in rows])),
#         'NLL': float(np.mean([r['NLL'] for r in rows])),
#         'Brier': float(np.mean([r['Brier'] for r in rows])),
#     }
#     return rows, macro


def ece_prob(y, p, n_bins=15):
    """
    Standard binary ECE: bin by predicted *positive* probability p,
    compare bin mean confidence vs empirical positive rate (mean of y).
    """
    y = y.reshape(-1)
    p = p.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1] if i < n_bins - 1 else 1.0000001
        m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        conf_bin = float(p[m].mean())
        freq_pos = float(y[m].mean())  # empirical positive rate
        w = float(m.mean())
        ece += w * abs(freq_pos - conf_bin)
    return float(ece)


def ece_balanced(y, p, n_bins=15):
    """
    Balanced ECE: average of positive-side ECE on (y, p) and
    negative-side ECE on (1-y, 1-p). Helpful under heavy imbalance.
    """
    pos = ece_prob(y, p, n_bins=n_bins)
    neg = ece_prob(1 - y, 1 - p, n_bins=n_bins)
    return 0.5 * (pos + neg)


def reliability_diagram_prob(y, p, out_path, title, n_bins=15):
    """
    Reliability diagram for the positive class:
    x = mean predicted probability in bin
    y = empirical positive rate in bin
    """
    y = y.reshape(-1)
    p = p.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1] if i < n_bins - 1 else 1.0000001
        m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', label='ideal')
    if xs:
        plt.plot(xs, ys, marker='o', linewidth=1, label='model')
    plt.xlabel('Confidence (P[y=1])')
    plt.ylabel('Empirical positive rate')
    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def per_label_metrics(y, p, n_bins=15):
    L = y.shape[1]
    rows = []
    for j in range(L):
        rows.append({
            'label_index': j,
            'ECE_prob': ece_prob(y[:, j], p[:, j], n_bins=n_bins),
            'ECE_bal': ece_balanced(y[:, j], p[:, j], n_bins=n_bins),
            'NLL': nll(y[:, j], p[:, j]),
            'Brier': brier(y[:, j], p[:, j]),
        })
    macro = {
        'label_index': 'macro',
        'ECE_prob': float(np.mean([r['ECE_prob'] for r in rows])),
        'ECE_bal': float(np.mean([r['ECE_bal'] for r in rows])),
        'NLL': float(np.mean([r['NLL'] for r in rows])),
        'Brier': float(np.mean([r['Brier'] for r in rows])),
    }
    return rows, macro


class TemperatureScaler:
    """
    Scalar temperature scaling with Adam on s = log T.
    logits = logits / T. Optimize T on validation set by minimizing NLL.
    """

    def __init__(self, init_T: float = 1.0, lr: float = 0.05, steps: int = 1000, tol: float = 1e-7):
        self.s = float(math.log(max(init_T, 1e-6)))
        self.lr = float(lr)
        self.steps = int(steps)
        self.tol = float(tol)
        self.m = 0.0
        self.v = 0.0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    @property
    def T(self) -> float:
        return float(math.exp(self.s))

    def _grad(self, logits: np.ndarray, y: np.ndarray) -> float:
        T = self.T
        z = logits / T
        p = sigmoid(z)
        g = (p - y) * (-logits / T)
        return float(g.mean())

    def _loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        p = sigmoid(logits / self.T)
        return nll(y, p)

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray, verbose: bool = True) -> None:
        t = 0
        prev_step = None
        for t in range(1, self.steps + 1):
            g = self._grad(logits_val, y_val)
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
            mhat = self.m / (1.0 - self.beta1 ** t)
            vhat = self.v / (1.0 - self.beta2 ** t)
            step = self.lr * mhat / (math.sqrt(vhat) + self.eps)
            self.s -= step
            if prev_step is not None and abs(step) < self.tol:
                break
            prev_step = step
        if verbose:
            print(f"[TS] Steps: {t}, T: {self.T:.6f}, val NLL: {self._loss(logits_val, y_val):.6f}")

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.T


def reliability_diagram(y: np.ndarray, p: np.ndarray, out_path: str, title: str, n_bins: int = 15) -> None:
    conf = p.reshape(-1)
    y_flat = y.reshape(-1)
    pred_flat = (conf >= 0.5).astype(np.float32)
    acc_flat = (pred_flat == y_flat).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        xs.append(float(conf[m].mean()))
        ys.append(float(acc_flat[m].mean()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', label='ideal')
    if xs:
        plt.plot(xs, ys, marker='o', linewidth=1, label='model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_labels(path: Optional[str], n_classes: int) -> List[str]:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) == n_classes:
            return labels
        print('[WARN] labels.txt length does not match number of classes. Falling back to indices.')
    return [f'label_{i}' for i in range(n_classes)]


def parse_hierarchy(h_path: Optional[str], labels: List[str]) -> Tuple[List[int], List[List[int]]]:
    if not h_path or not os.path.exists(h_path):
        return [], []
    with open(h_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    name_to_idx = {name: i for i, name in enumerate(labels)}
    parent_indices, child_indices = [], []
    for parent_name, children_names in mapping.items():
        if parent_name not in name_to_idx:
            print(f'[WARN] Parent "{parent_name}" not found in labels. Skipping.')
            continue
        pidx = name_to_idx[parent_name]
        kids = []
        for cn in children_names:
            if cn not in name_to_idx:
                print(f'[WARN] Child "{cn}" not found in labels. Skipping.')
                continue
            kids.append(name_to_idx[cn])
        if kids:
            parent_indices.append(pidx)
            child_indices.append(kids)
    return parent_indices, child_indices


def cvr(probs: np.ndarray, parent_indices: List[int], child_indices: List[List[int]], thr: float = 0.5) -> float:
    if not parent_indices:
        return float('nan')
    n = probs.shape[0]
    violations = 0
    for pidx, kids in zip(parent_indices, child_indices):
        parent_pos = probs[:, pidx] >= thr
        child_any = (probs[:, kids] >= thr).any(axis=1)
        violations += np.logical_and(~parent_pos, child_any).sum()
    denom = n * len(parent_indices)
    return float(violations) / float(denom) if denom > 0 else float('nan')


def conditional_ece(y: np.ndarray, p: np.ndarray, child: int, parent: int, parent_state: int = 1,
                    n_bins: int = 15) -> float:
    mask = (y[:, parent] == parent_state)
    if not np.any(mask):
        return float('nan')
    # return ece_single(y[mask, child], p[mask, child], n_bins=n_bins)
    return ece_prob(y[mask, child], p[mask, child], n_bins=n_bins)


def load_array(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        z = np.load(path)
        for k in ['arr', 'P', 'data', 'preds', 'probs', 'logits', 'y', 'x']:
            if k in z:
                return z[k]
        return z[list(z.files)[0]]
    raise ValueError(f'Unsupported file format: {path}')


# -------------------------
# Main run
# -------------------------

def run():
    print('[INFO] Loading arrays...')
    y_val = load_array(YVAL_PATH).astype(np.float32)
    y_test = load_array(YTEST_PATH).astype(np.float32)
    P_val_in = load_array(PVAL_PATH).astype(np.float32)
    P_test_in = load_array(PTEST_PATH).astype(np.float32)
    pos_counts = y_val.sum(axis=0).astype(int)
    neg_counts = (y_val.shape[0] - pos_counts).astype(int)
    few = [(j, int(pos_counts[j])) for j in range(len(pos_counts)) if pos_counts[j] < 5]
    print(
        f'[VAL] Positives per label (min/median/max): {pos_counts.min()} / {np.median(pos_counts)} / {pos_counts.max()}')
    if few:
        print('[VAL] Labels with <5 positives:', few[:10], '...' if len(few) > 10 else '')

    if y_val.ndim != 2 or y_test.ndim != 2:
        raise ValueError('y_val and y_test must be 2D arrays [N, L].')
    if P_val_in.shape != y_val.shape or P_test_in.shape != y_test.shape:
        raise ValueError('Prediction shapes must match ground truth shapes.')

    n_classes = y_test.shape[1]
    labels = parse_labels(LABELS_PATH, n_classes)
    parent_idxs, child_groups = parse_hierarchy(HIERARCHY_JSON, labels)

    ensure_dir(OUTDIR)

    # Determine logits and probabilities
    if FROM_PROBS:
        logits_val = logit(P_val_in)
        logits_test = logit(P_test_in)
        probs_test_raw = P_test_in
    else:
        logits_val = P_val_in
        logits_test = P_test_in
        probs_test_raw = sigmoid(P_test_in)

    # Fit temperature on validation
    print('[INFO] Fitting temperature scaling...')
    ts = TemperatureScaler(init_T=INIT_T, lr=TS_LR, steps=TS_STEPS, tol=TS_TOL)
    ts.fit(logits_val, y_val, verbose=True)

    # Compute val baseline and TS NLLs
    val_pred_base = P_val_in if FROM_PROBS else sigmoid(P_val_in)
    val_nll_base = nll(y_val, val_pred_base)
    val_nll_ts = nll(y_val, sigmoid(logits_val / ts.T))

    # Keep TS only if it helps on validation
    if val_nll_ts >= val_nll_base - 1e-8:
        print('[TS] No validation improvement; reverting to T=1.0')
        ts.s = 0.0  # log(1)=0

    probs_test_ts = sigmoid(ts.transform_logits(logits_test))

    # Metrics pre and post
    print('[INFO] Computing metrics...')
    rows_pre, macro_pre = per_label_metrics(y_test, probs_test_raw, n_bins=N_BINS)
    rows_post, macro_post = per_label_metrics(y_test, probs_test_ts, n_bins=N_BINS)

    df_pre = pd.DataFrame(rows_pre)
    df_pre['phase'] = 'pre_TS'
    df_post = pd.DataFrame(rows_post)
    df_post['phase'] = 'post_TS'
    df_all = pd.concat([df_pre, df_post], ignore_index=True)
    df_all['label'] = df_all['label_index'].apply(lambda idx: labels[idx] if isinstance(idx, int) else 'macro')

    per_label_path = os.path.join(OUTDIR, 'metrics_per_label.csv')
    macro_path = os.path.join(OUTDIR, 'metrics_macro.csv')
    df_all.to_csv(per_label_path, index=False)

    macro_df = pd.DataFrame([
        {'phase': 'pre_TS', **macro_pre},
        {'phase': 'post_TS', **macro_post},
    ])
    macro_df.to_csv(macro_path, index=False)

    print('--- Macro metrics (test) ---')
    print(macro_df.to_string(index=False))

    # Reliability diagrams for selected labels: worst by pre-TS ECE
    print('[INFO] Saving reliability diagrams...')
    # worst = df_pre.sort_values('ECE', ascending=False).head(SELECT_LABELS)
    worst = df_pre.sort_values('ECE_prob', ascending=False).head(SELECT_LABELS)
    worst_indices = [int(i) for i in worst['label_index'].tolist()]
    for j in worst_indices:
        title_pre = f'{labels[j]} reliability (pre TS)'
        out_pre = os.path.join(PLOTS_DIR, f'reliability_{labels[j]}_pre.png')
        reliability_diagram_prob(y_test[:, j], probs_test_raw[:, j], out_pre, title_pre, n_bins=N_BINS)

        title_post = f'{labels[j]} reliability (post TS)'
        out_post = os.path.join(PLOTS_DIR, f'reliability_{labels[j]}_post.png')
        reliability_diagram_prob(y_test[:, j], probs_test_ts[:, j], out_post, title_post, n_bins=N_BINS)

    # Hierarchical metrics
    summary_lines = []
    summary_lines.append(f'Fitted temperature T = {ts.T:.6f}')

    if parent_idxs:
        cvr_pre = cvr(probs_test_raw, parent_idxs, child_groups, thr=CVR_THRESHOLD)
        cvr_post = cvr(probs_test_ts, parent_idxs, child_groups, thr=CVR_THRESHOLD)
        summary_lines.append(f'CVR (thr={CVR_THRESHOLD}) pre_TS:  {cvr_pre:.6f}')
        summary_lines.append(f'CVR (thr={CVR_THRESHOLD}) post_TS: {cvr_post:.6f}')
        summary_lines.append('Conditional ECE (child | parent=1):')
        for pidx, cidxs in zip(parent_idxs, child_groups):
            for c in cidxs:
                cECE_pre = conditional_ece(y_test, probs_test_raw, c, pidx, parent_state=1, n_bins=N_BINS)
                cECE_post = conditional_ece(y_test, probs_test_ts, c, pidx, parent_state=1, n_bins=N_BINS)
                summary_lines.append(f'  {labels[c]} | {labels[pidx]}=1 : pre {cECE_pre:.6f} -> post {cECE_post:.6f}')
    else:
        summary_lines.append('No hierarchy provided. Skipping CVR and conditional ECE.')

    if MCD_PATH and os.path.exists(MCD_PATH):
        try:
            z = np.load(MCD_PATH)
            Pm = z['P']
            if Pm.ndim != 3:
                raise ValueError('MCD array must be [K, N, L].')
            mu = Pm.mean(axis=0)
            var = Pm.var(axis=0)
            abs_err = np.abs(y_test - mu)
            summary_lines.append(f'MCD mean variance: {float(var.mean()):.6f}')
            summary_lines.append(f'Mean absolute error (μ probs): {float(abs_err.mean()):.6f}')

            # Spearman correlation
            def rank(a: np.ndarray) -> np.ndarray:
                temp = a.argsort().argsort()
                return temp.astype(np.float32)

            rhos = []
            for j in range(n_classes):
                v = var[:, j]
                e = abs_err[:, j]
                rv = rank(v)
                re = rank(e)
                rv = (rv - rv.mean()) / (rv.std() + 1e-8)
                re = (re - re.mean()) / (re.std() + 1e-8)
                rho = float(np.mean(rv * re))
                if not np.isnan(rho):
                    rhos.append(rho)
            if rhos:
                summary_lines.append(f'Spearman corr(var, abs_err) median across labels: {float(np.median(rhos)):.4f}')
        except Exception as e:
            summary_lines.append(f'[WARN] MCD analysis failed: {e}')

    ensure_dir(OUTDIR)
    with open(os.path.join(OUTDIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print(f'\nSaved per-label metrics to: {per_label_path}')
    print(f'Saved macro metrics to: {macro_path}')
    print(f'Reliability diagrams saved in: {PLOTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    run()
