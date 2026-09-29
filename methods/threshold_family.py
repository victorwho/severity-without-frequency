#!/usr/bin/env python3
"""Threshold-family contrasts for the manuscript (Table 5, v1.1).

For each cut c in {30, 40, 65, 80, 90, 100} fit the paper's specification
(NB2, alpha by MLE, city FE, free exposure elasticity) on the indicator
risk >= c, reporting model-based and region-cluster-robust CIs, Poisson HC1,
and E-values. Companion to paper_analysis.py, which runs the pre-specified
>=65 contrast and the restricted >100 vs <30 comparison.

Usage: python validation/threshold_family.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
H = Path(__file__).resolve().parent


def evalue(rr: float) -> float:
    """VanderWeele & Ding E-value for an observed risk ratio."""
    r = rr if rr >= 1 else 1.0 / rr
    if r <= 1:
        return 1.0
    return r + np.sqrt(r * (r - 1.0))


def design(d: pd.DataFrame):
    ct = pd.get_dummies(d.city, drop_first=True).astype(float).values
    return np.column_stack([np.log(d.passages.values), ct])


def nb2(y, x, X0, groups=None):
    X = sm.add_constant(np.column_stack([x, X0]))
    if groups is not None:
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=300, cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True})
    else:
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=300)
    b, se = m.params[1], m.bse[1]
    p = 2 * stats.norm.sf(abs(b / se))
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


def poisson_hc(y, x, X0):
    X = sm.add_constant(np.column_stack([x, X0]))
    m = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    b, se = m.params[1], m.bse[1]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[1]


def main() -> int:
    d = pd.read_csv(H / "out" / "panel_final.csv")
    d = d[(d.passages > 0) & d.risk.notna()].reset_index(drop=True)
    d["region"] = d.city.astype(str)
    X0 = design(d)
    print(f"panel: {len(d)} sites, {d.region.nunique()} regions, "
          f"{d.passages.sum()/1e9:.2f}B passages, {int(d.all_crashes.sum())} crashes, "
          f"{int(d.ksi.sum())} KSI\n")

    cuts = [30, 40, 65, 80, 90, 100]
    print("descriptives per cut (above-cut totals, crude rates below|above):")
    for c in cuts:
        hi, lo = d[d.risk >= c], d[d.risk < c]
        print(f"  cut {c:>3}: above {len(hi):>4} sites {hi.passages.sum()/1e6:7.0f}M "
              f"{int(hi.all_crashes.sum()):>5} cr {int(hi.ksi.sum()):>4} ksi | "
              f"crash/M {lo.all_crashes.sum()/lo.passages.sum()*1e6:5.2f}|{hi.all_crashes.sum()/hi.passages.sum()*1e6:5.2f} | "
              f"ksi/M {lo.ksi.sum()/lo.passages.sum()*1e6:6.3f}|{hi.ksi.sum()/hi.passages.sum()*1e6:6.3f}")
    print()

    for oc in ("all_crashes", "ksi"):
        y = d[oc].values.astype(float)
        print(f"=== outcome: {oc} " + "=" * 46)
        for c in cuts:
            x = (d.risk >= c).astype(float).values
            r1 = nb2(y, x, X0)
            r2 = nb2(y, x, X0, groups=d.region.values)
            r3 = poisson_hc(y, x, X0)
            print(f"  cut >= {c:<3} (n_hi={int(x.sum())})")
            print(f"     NB2 model    RR {r1[0]:5.2f} [{r1[1]:4.2f}, {r1[2]:4.2f}]  p={r1[3]:.4g}")
            print(f"     NB2 cluster  RR {r2[0]:5.2f} [{r2[1]:4.2f}, {r2[2]:4.2f}]  p={r2[3]:.4g}")
            print(f"     Poisson HC1  RR {r3[0]:5.2f} [{r3[1]:4.2f}, {r3[2]:4.2f}]  p={r3[3]:.4g}")
            lo_bound = evalue(r1[1]) if r1[1] > 1 else 1.0
            print(f"     E-value point {evalue(r1[0]):.2f} / CI {lo_bound:.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
