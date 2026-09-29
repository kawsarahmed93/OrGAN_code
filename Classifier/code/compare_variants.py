#!/usr/bin/env python3
"""Paired per-fold comparison of FusionNet input variants.

Each cross-validation fold index corresponds to the same training split in
every run, so fold i of variant A and fold i of variant B are a matched pair.
Pairing matters: fold-to-fold variance is shared by both variants and cancels
in the difference, so a gap that sits well inside each variant's own s.d. can
still be consistent across every fold. "Ahead at 7/7 thresholds but each under
1 s.d." is not a claim a reviewer accepts; "ahead in all five folds at every
threshold" is.

With five folds the sign test bottoms out at p = 1/2^5 = 0.031 one-sided
(0.0625 two-sided) - that is the most significance five folds can carry, and
it is only reached on a clean 5-0 sweep. Wilcoxon signed-rank is reported
alongside; at n=5 its exact minimum is the same, but it uses the size of the
differences and not just their sign.

Usage
-----
  python compare_variants.py --a FusionNet_xl_224_bce --b FusionNet_x0_224_bce_3
  python compare_variants.py --a FusionNet_xl_224_bce \
                             --b FusionNet_x0_224_bce_3 FusionNet_xx_224_bce_2 \
                             --metric macro micro
Reads the 'per_fold' sheet written by test_iou.py, so run that first for every
variant being compared.
"""

from argparse import ArgumentParser
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

WEIGHTS_ROOT = '../weights_cv'
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def parse_args():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--a', required=True,
                    help='Reference variant (run directory name under --weights-root)')
    ap.add_argument('--b', required=True, nargs='+',
                    help='One or more variants to compare against --a')
    ap.add_argument('--config', default='proposed',
                    help="Config name in the workbook filename (default: proposed)")
    ap.add_argument('--weights-root', default=WEIGHTS_ROOT)
    ap.add_argument('--metric', nargs='+', default=['macro', 'micro'],
                    help="'macro', 'micro', 'all-findings', or explicit finding names")
    ap.add_argument('--out', default=None,
                    help='Output .xlsx (default: paired_comparison_<a>.xlsx in --a run dir)')
    return ap.parse_args()


def load_per_fold(run, root, config):
    path = os.path.join(root, run, f'final_iou_accuracy_{config}.xlsx')
    if not os.path.exists(path):
        raise SystemExit(f"missing: {path}\n  run test_iou.py for '{run}' first")
    try:
        df = pd.read_excel(path, sheet_name='per_fold')
    except ValueError:
        raise SystemExit(
            f"'{path}' has no 'per_fold' sheet - it predates per-fold saving.\n"
            f"  re-run test_iou.py for '{run}'")
    df['run'] = run
    return df


def paired_test(a_vals, b_vals):
    """Sign test + Wilcoxon on paired per-fold values. Returns a dict."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    nz = d[d != 0]
    n_eff, n_pos = len(nz), int((nz > 0).sum())

    out = {'n_folds': n, 'a_wins': n_pos, 'b_wins': n_eff - n_pos, 'ties': n - n_eff,
           'mean_diff': round(float(d.mean()), 6) if n else np.nan,
           'min_diff': round(float(d.min()), 6) if n else np.nan,
           'max_diff': round(float(d.max()), 6) if n else np.nan}

    if n_eff == 0:
        out.update(sign_p_one_sided=np.nan, sign_p_two_sided=np.nan,
                   wilcoxon_p_two_sided=np.nan, verdict='all ties')
        return out

    # one-sided in the observed direction, then the symmetric two-sided form
    k = max(n_pos, n_eff - n_pos)
    p_one = float(stats.binom.sf(k - 1, n_eff, 0.5))
    out['sign_p_one_sided'] = round(p_one, 5)
    out['sign_p_two_sided'] = round(min(1.0, 2 * p_one), 5)
    try:
        out['wilcoxon_p_two_sided'] = round(
            float(stats.wilcoxon(nz, alternative='two-sided', mode='exact').pvalue), 5)
    except Exception:
        out['wilcoxon_p_two_sided'] = np.nan

    swept = (n_pos == n_eff) or (n_pos == 0)
    direction = 'A' if n_pos > n_eff - n_pos else ('B' if n_pos < n_eff - n_pos else 'split')
    out['verdict'] = (f'{direction} wins {k}/{n_eff}' + (' (sweep)' if swept else ''))
    return out


def main():
    args = parse_args()
    a_df = load_per_fold(args.a, args.weights_root, args.config)

    findings = [m for m in sorted(a_df['metric'].unique())
                if m not in ('macro', 'micro')]
    metrics = []
    for m in args.metric:
        metrics.extend(findings if m == 'all-findings' else [m])

    rows = []
    for b_run in args.b:
        b_df = load_per_fold(b_run, args.weights_root, args.config)
        for metric in metrics:
            for thr in THRESHOLDS:
                sa = a_df[(a_df.metric == metric) & (a_df.threshold == thr)].sort_values('fold')
                sb = b_df[(b_df.metric == metric) & (b_df.threshold == thr)].sort_values('fold')
                if sa.empty or sb.empty:
                    continue
                folds = sorted(set(sa.fold) & set(sb.fold))
                sa = sa[sa.fold.isin(folds)].sort_values('fold')
                sb = sb[sb.fold.isin(folds)].sort_values('fold')
                r = {'A': args.a, 'B': b_run, 'metric': metric, 'threshold': thr,
                     'A_mean': round(float(sa.value.mean()), 4),
                     'B_mean': round(float(sb.value.mean()), 4)}
                r.update(paired_test(sa.value.values, sb.value.values))
                for f, va, vb in zip(folds, sa.value.values, sb.value.values):
                    r[f'fold{f}_diff'] = round(float(va - vb), 5)
                rows.append(r)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit('no overlapping (metric, threshold) rows to compare')

    lead = ['A', 'B', 'metric', 'threshold', 'A_mean', 'B_mean', 'mean_diff',
            'a_wins', 'b_wins', 'ties', 'n_folds',
            'sign_p_one_sided', 'sign_p_two_sided', 'wilcoxon_p_two_sided', 'verdict']
    res = res[[c for c in lead if c in res.columns] +
              [c for c in res.columns if c not in lead]]

    # console summary
    for (A, B, metric), g in res.groupby(['A', 'B', 'metric'], sort=False):
        sweeps = int(((g.a_wins == g.n_folds) | (g.b_wins == g.n_folds)).sum())
        print(f"\n=== {A}  vs  {B}   [{metric}] ===")
        print(g[['threshold', 'A_mean', 'B_mean', 'mean_diff', 'a_wins', 'b_wins',
                 'sign_p_one_sided', 'sign_p_two_sided', 'verdict']].to_string(index=False))
        print(f"  clean sweeps: {sweeps}/{len(g)} thresholds"
              + (f"   [Holm-adjusted across {len(g)} thresholds: "
                 f"smallest adjusted one-sided p = "
                 f"{min(1.0, g.sign_p_one_sided.min() * len(g)):.4f}]" if sweeps else ""))

    out = args.out or os.path.join(args.weights_root, args.a,
                                   f'paired_comparison_{args.a}.xlsx')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        res.to_excel(xl, sheet_name='paired_tests', index=False)
        pd.concat([a_df] + [load_per_fold(b, args.weights_root, args.config)
                            for b in args.b]).to_excel(xl, sheet_name='per_fold_raw', index=False)
    print(f"\nSaved paired comparison to {out} (sheets: paired_tests, per_fold_raw)")


if __name__ == '__main__':
    main()
