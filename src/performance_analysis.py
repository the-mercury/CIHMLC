"""
Lambda Penalty Analysis
-----------------------
Reproduces the cross-validated comparisons between penalty modes (fixed vs. data-driven)
and the flat baseline using the fold-level results you provided.

Outputs:
- analysis_results/summary_by_strategy_lambda.csv
- analysis_results/dd_vs_fixed_paired_tests_per_lambda.csv
- analysis_results/fixed_best_vs_flat.csv
- analysis_results/dd_best_vs_flat.csv
- analysis_results/bootstrap_cis_key_contrasts.csv
- analysis_results/plots/acc_vs_lambda.png
- analysis_results/plots/f1_vs_lambda.png
- analysis_results/plots/auc_vs_lambda.png

Dependencies:
    pip install numpy pandas scipy matplotlib
"""

import os
from typing import List, Tuple, Dict, Tuple as Tup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config


# -----------------------------
# Data loading
# -----------------------------

def pct_to_float(x):
    """Convert '87.2%' -> 0.872; pass through floats."""
    if isinstance(x, str) and x.endswith('%'):
        return float(x.strip('%')) / 100.0
    return float(x)


def make_dataframe(rows: List[Tuple], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=['lambda', 'fold', 'acc', 'f1', 'auc'])
    for col in ['acc', 'f1', 'auc']:
        df[col] = df[col].map(pct_to_float)
    df['strategy'] = name
    return df[['strategy', 'lambda', 'fold', 'acc', 'f1', 'auc']]


def load_embedded() -> Tup[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the fold-level results embedded below."""

    # Fixed penalty (beta=1)
    fixed_rows = [
        (0.3, 1, '86.23%', '83.22%', '91.47%'),
        (0.3, 2, '87.18%', '83.87%', '90.34%'),
        (0.3, 3, '86.91%', '84.44%', '92.12%'),
        (0.3, 4, '85.47%', '84.99%', '90.77%'),
        (0.3, 5, '87.71%', '85.64%', '91.13%'),
        (0.5, 1, '86.43%', '87.04%', '93.32%'),
        (0.5, 2, '87.16%', '85.11%', '91.14%'),
        (0.5, 3, '87.62%', '84.64%', '92.03%'),
        (0.5, 4, '88.08%', '86.25%', '90.74%'),
        (0.5, 5, '88.81%', '84.68%', '92.92%'),
        (0.7, 1, '86.00%', '83.84%', '90.68%'),
        (0.7, 2, '84.75%', '82.36%', '88.67%'),
        (0.7, 3, '85.49%', '84.08%', '89.91%'),
        (0.7, 4, '86.24%', '83.22%', '91.15%'),
        (0.7, 5, '84.98%', '82.60%', '89.14%'),
        (1.0, 1, '83.69%', '82.70%', '89.64%'),
        (1.0, 2, '82.50%', '80.89%', '87.73%'),
        (1.0, 3, '84.02%', '82.52%', '88.76%'),
        (1.0, 4, '85.54%', '81.06%', '87.88%'),
        (1.0, 5, '84.35%', '81.79%', '89.79%'),
    ]

    # Data-driven penalty (corrected values)
    dd_rows = [
        (0.3, 1, '87.40%', '85.04%', '91.53%'),
        (0.3, 2, '88.02%', '86.23%', '92.17%'),
        (0.3, 3, '88.62%', '86.59%', '92.68%'),
        (0.3, 4, '89.19%', '87.38%', '93.47%'),
        (0.3, 5, '89.60%', '87.51%', '94.46%'),
        (0.5, 1, '89.92%', '88.54%', '94.31%'),
        (0.5, 2, '89.72%', '88.20%', '94.18%'),
        (0.5, 3, '90.09%', '88.68%', '94.22%'),
        (0.5, 4, '89.53%', '87.48%', '94.01%'),
        (0.5, 5, '90.57%', '88.64%', '95.20%'),
        (0.7, 1, '90.66%', '88.88%', '95.18%'),
        (0.7, 2, '90.83%', '89.34%', '95.24%'),
        (0.7, 3, '91.39%', '90.33%', '95.62%'),
        (0.7, 4, '91.88%', '90.77%', '96.16%'),
        (0.7, 5, '91.97%', '90.63%', '96.17%'),
        (1.0, 1, '85.81%', '83.16%', '89.41%'),
        (1.0, 2, '86.18%', '84.05%', '90.08%'),
        (1.0, 3, '86.93%', '84.70%', '90.69%'),
        (1.0, 4, '87.10%', '84.43%', '91.27%'),
        (1.0, 5, '87.14%', '84.56%', '91.34%'),
    ]

    # Flat (lambda = 0)
    flat_rows = [
        (0.0, 1, '85.80%', '82.94%', '89.35%'),
        (0.0, 2, '85.89%', '82.87%', '89.54%'),
        (0.0, 3, '86.26%', '83.41%', '90.11%'),
        (0.0, 4, '86.89%', '85.42%', '90.98%'),
        (0.0, 5, '87.87%', '85.65%', '92.27%'),
    ]

    fixed = make_dataframe(fixed_rows, 'fixed')
    dd = make_dataframe(dd_rows, 'data_driven')
    flat = make_dataframe(flat_rows, 'flat')
    return fixed, dd, flat


# -----------------------------
# Analysis helpers
# -----------------------------

def summarize_by_lambda(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(['strategy', 'lambda'], as_index=False)
        .agg(
            folds=('fold', 'nunique'),
            acc_mean=('acc', 'mean'), acc_sd=('acc', 'std'),
            f1_mean=('f1', 'mean'), f1_sd=('f1', 'std'),
            auc_mean=('auc', 'mean'), auc_sd=('auc', 'std'),
        )
    )
    return g


def cohens_dz(paired_diffs: np.ndarray) -> float:
    return float(np.mean(paired_diffs) / np.std(paired_diffs, ddof=1))


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Return Holm–Bonferroni adjusted p-values in original order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = [None] * m
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        adj_p = (m - rank + 1) * pvals[i]
        running_max = max(running_max, adj_p)
        adj[i] = min(1.0, running_max)
    return adj


def paired_tests_dd_vs_fixed_per_lambda(fixed: pd.DataFrame, dd: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Returns rows with one metric; Holm adjustment is applied later after concatenation."""
    rows = []
    for lam in sorted(set(fixed['lambda']).intersection(dd['lambda'])):
        f = fixed[fixed['lambda'] == lam].sort_values('fold')
        d = dd[dd['lambda'] == lam].sort_values('fold')
        if len(f) == 0 or len(d) == 0:
            continue
        diff = d[metric].values - f[metric].values  # DD - Fixed
        _, t_p = stats.ttest_rel(d[metric], f[metric])
        try:
            _, w_p = stats.wilcoxon(d[metric], f[metric], zero_method='wilcox', alternative='two-sided', mode='auto')
        except ValueError:
            w_p = np.nan
        rows.append({
            'lambda': lam,
            'metric': metric,
            'mean_diff': float(np.mean(diff)),
            'sd_diff': float(np.std(diff, ddof=1)),
            'cohens_dz': cohens_dz(diff),
            't_pvalue': float(t_p),
            'wilcoxon_pvalue': float(w_p) if not np.isnan(w_p) else np.nan,
        })
    return pd.DataFrame(rows)


def paired_vs_flat(strategy_df: pd.DataFrame, lam: float, flat_df: pd.DataFrame, metric: str) -> Dict:
    s = strategy_df[strategy_df['lambda'] == lam].sort_values('fold')
    fl = flat_df.sort_values('fold')
    diff = s[metric].values - fl[metric].values
    _, t_p = stats.ttest_rel(s[metric], fl[metric])
    try:
        _, w_p = stats.wilcoxon(s[metric], fl[metric], zero_method='wilcox', alternative='two-sided', mode='auto')
    except ValueError:
        w_p = np.nan
    return {
        'metric': metric,
        'lambda': lam,
        'mean_diff': float(np.mean(diff)),
        'sd_diff': float(np.std(diff, ddof=1)),
        'cohens_dz': cohens_dz(diff),
        't_pvalue': float(t_p),
        'wilcoxon_pvalue': float(w_p) if not np.isnan(w_p) else np.nan,
    }


def bootstrap_ci(diff: np.ndarray, n_boot: int = 20000,
                 alpha: float = 0.05, seed: int = 42) -> Tup[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idxs = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idxs].mean(axis=1)
    return float(np.quantile(boot_means, alpha / 2)), float(np.quantile(boot_means, 1 - alpha / 2))


def plot_metric_vs_lambda(summary: pd.DataFrame, metric_mean_col: str, metric_baseline: float, out_path: str):
    plt.figure()
    for strat in ['fixed', 'data_driven']:
        sub = summary[summary['strategy'] == strat].sort_values('lambda')
        if len(sub) == 0:
            continue
        plt.plot(sub['lambda'], sub[metric_mean_col], marker='o', label=strat)
    plt.xlabel('lambda')
    plt.hlines(y=metric_baseline, xmin=0.25, xmax=1.05, linewidth=1, color='gray', linestyles='--', label='lambda=0')
    ylabel = metric_mean_col.replace('_mean', '').upper()
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs lambda (5-fold means)')
    plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def to_pp(x: float) -> float:
    """Convert proportion to percentage points."""
    return 100.0 * x


# -----------------------------
# Main
# -----------------------------

def main():
    # save_dir = os.path.join('logs/logfiles', 'analysis_results')
    save_dir = Config.PERFORMANCE_ANALYSIS_DIR
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    fixed, dd, flat = load_embedded()

    # Summary by strategy and lambda
    summary = summarize_by_lambda(pd.concat([fixed, dd, flat], ignore_index=True))
    summary.to_csv(f'{save_dir}/summary_by_strategy_lambda.csv', index=False)

    # Paired tests: Data-driven vs Fixed at the same lambda (per metric)
    acc_tests = paired_tests_dd_vs_fixed_per_lambda(fixed, dd, 'acc')
    f1_tests = paired_tests_dd_vs_fixed_per_lambda(fixed, dd, 'f1')
    auc_tests = paired_tests_dd_vs_fixed_per_lambda(fixed, dd, 'auc')
    dd_fixed_tests = pd.concat([acc_tests, f1_tests, auc_tests], ignore_index=True)

    # Holm adjust across metrics within each lambda (SAFE ASSIGNMENT)
    dd_fixed_tests['t_pvalue_holm'] = np.nan
    for lam, idx in dd_fixed_tests.groupby('lambda').groups.items():
        pvals = dd_fixed_tests.loc[idx, 't_pvalue'].tolist()
        adj = holm_bonferroni(pvals)
        dd_fixed_tests.loc[idx, 't_pvalue_holm'] = adj

    # Present deltas in percentage points for readability
    dd_fixed_tests['mean_diff_pp'] = dd_fixed_tests['mean_diff'].apply(to_pp)
    dd_fixed_tests['sd_diff_pp'] = dd_fixed_tests['sd_diff'].apply(to_pp)
    dd_fixed_tests.to_csv(f'{save_dir}/dd_vs_fixed_paired_tests_per_lambda.csv', index=False)

    # Choose best lambda per strategy by macro-F1 mean (tie-breakers: AUC, then ACC)
    strat_means = summary[summary['strategy'].isin(['fixed', 'data_driven'])]
    fixed_best_lam = (
        strat_means[strat_means['strategy'] == 'fixed']
        .sort_values(['f1_mean', 'auc_mean', 'acc_mean'], ascending=False)
        .iloc[0]['lambda']
    )
    dd_best_lam = (
        strat_means[strat_means['strategy'] == 'data_driven']
        .sort_values(['f1_mean', 'auc_mean', 'acc_mean'], ascending=False)
        .iloc[0]['lambda']
    )

    # Compare each best setting vs flat
    fixed_vs_flat = pd.DataFrame(
        [paired_vs_flat(fixed, float(fixed_best_lam), flat, m) for m in ['acc', 'f1', 'auc']]
    )
    dd_vs_flat = pd.DataFrame(
        [paired_vs_flat(dd, float(dd_best_lam), flat, m) for m in ['acc', 'f1', 'auc']]
    )

    # Holm adjust within each 3-metric family
    fixed_vs_flat['t_pvalue_holm'] = holm_bonferroni(list(fixed_vs_flat['t_pvalue'].values))
    dd_vs_flat['t_pvalue_holm'] = holm_bonferroni(list(dd_vs_flat['t_pvalue'].values))

    for df in (fixed_vs_flat, dd_vs_flat):
        df['mean_diff_pp'] = df['mean_diff'].apply(to_pp)
        df['sd_diff_pp'] = df['sd_diff'].apply(to_pp)

    fixed_vs_flat.to_csv(f'{save_dir}/fixed_best_vs_flat.csv', index=False)
    dd_vs_flat.to_csv(f'{save_dir}/dd_best_vs_flat.csv', index=False)

    # -----------------------------
    # Bootstrap CIs for key contrasts
    # -----------------------------
    def paired_vector(df_a, lam_a, df_b, lam_b, metric):
        a = df_a[df_a['lambda'] == lam_a].sort_values('fold')[metric].values
        b = df_b[df_b['lambda'] == lam_b].sort_values('fold')[metric].values
        if a.shape != b.shape:
            raise ValueError(
                f"Fold mismatch for {metric}: λ_a={lam_a} (shape {a.shape}) vs λ_b={lam_b} (shape {b.shape})")
        return a - b

    rows = []

    # DD − Fixed at λ = fixed_best and λ = dd_best
    for lam in [float(fixed_best_lam), float(dd_best_lam)]:
        for metric in ['acc', 'f1', 'auc']:
            dvals = paired_vector(dd, lam, fixed, lam, metric)
            lo, hi = bootstrap_ci(dvals, n_boot=20000, alpha=0.05, seed=1 if lam == float(fixed_best_lam) else 11)
            rows.append({
                'contrast': 'DD - Fixed', 'lambda': lam, 'metric': metric,
                'mean_diff': float(np.mean(dvals)), 'ci_low': lo, 'ci_high': hi
            })

    # DD(best) − Flat(0.0)
    dlam = float(dd_best_lam)
    for metric in ['acc', 'f1', 'auc']:
        dvals = paired_vector(dd, dlam, flat, 0.0, metric)
        dlo, dhi = bootstrap_ci(dvals, n_boot=20000, alpha=0.05, seed=2)
        rows.append({
            'contrast': f'DD({dlam}) - Flat(0)', 'lambda': dlam, 'metric': metric,
            'mean_diff': float(np.mean(dvals)), 'ci_low': dlo, 'ci_high': dhi
        })

    # Fixed(best) − Flat(0.0)
    flam = float(fixed_best_lam)
    for metric in ['acc', 'f1', 'auc']:
        fdvals = paired_vector(fixed, flam, flat, 0.0, metric)
        flo, fhi = bootstrap_ci(fdvals, n_boot=20000, alpha=0.05, seed=3)
        rows.append({
            'contrast': f'Fixed({flam}) - Flat(0)', 'lambda': flam, 'metric': metric,
            'mean_diff': float(np.mean(fdvals)), 'ci_low': flo, 'ci_high': fhi
        })

    cis = pd.DataFrame(rows)
    # human-friendly pp columns
    for c in ['mean_diff', 'ci_low', 'ci_high']:
        cis[c + '_pp'] = cis[c] * 100.0

    # annotate method and sizes
    cis['alpha'] = 0.05
    cis['n_boot'] = 20000
    cis['n_folds'] = 5

    cis = cis.sort_values(['contrast', 'lambda', 'metric']).reset_index(drop=True)
    cis.to_csv(f'{save_dir}/bootstrap_cis_key_contrasts.csv', index=False)

    # Plots (save under analysis_results/plots)
    plot_metric_vs_lambda(
        summary[summary['strategy'].isin(['fixed', 'data_driven'])],
        'acc_mean', 0.8654, f'{plot_dir}/acc_vs_lambda.png'
    )
    plot_metric_vs_lambda(
        summary[summary['strategy'].isin(['fixed', 'data_driven'])],
        'f1_mean', 0.8406, f'{plot_dir}/f1_vs_lambda.png'
    )
    plot_metric_vs_lambda(
        summary[summary['strategy'].isin(['fixed', 'data_driven'])],
        'auc_mean', 0.9045, f'{plot_dir}/auc_vs_lambda.png'
    )

    # Console summary
    print('=== Summary by strategy & lambda (means across 5 folds) ===')
    print(summary.to_string(index=False))

    print('\n=== Data-driven vs Fixed per-λ: paired tests (Δ = DD - Fixed) ===')
    cols = ['lambda', 'metric', 'mean_diff_pp', 'sd_diff_pp', 'cohens_dz', 't_pvalue', 't_pvalue_holm']
    print(dd_fixed_tests[cols].to_string(index=False))

    print(f'\nBest lambda by macro-F1: fixed={fixed_best_lam}, data_driven={dd_best_lam}')

    print('\n=== Fixed(best) vs Flat ===')
    cols = ['metric', 'lambda', 'mean_diff_pp', 'sd_diff_pp', 'cohens_dz', 't_pvalue', 't_pvalue_holm']
    print(fixed_vs_flat[cols].to_string(index=False))

    print('\n=== Data-driven(best) vs Flat ===')
    print(dd_vs_flat[cols].to_string(index=False))

    print('\n=== Bootstrap CIs for key contrasts (pp) ===')
    cols = ['contrast', 'lambda', 'metric', 'mean_diff_pp', 'ci_low_pp', 'ci_high_pp']
    print(cis[cols].to_string(index=False))


if __name__ == '__main__':
    main()
