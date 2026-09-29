#!/usr/bin/env python3
"""Inference-grade re-analysis for the manuscript.

Everything so far used sm.GLM(family=NegativeBinomial(alpha=1.0)) -- dispersion
FIXED at 1 rather than estimated, model-based SEs, and no clustering. That is
adequate for ranking effects but not for reported confidence intervals. This
re-runs the headline contrasts as they must appear in a paper:

  * NB2 with alpha estimated by MLE (statsmodels.discrete NegativeBinomial)
  * cluster-robust (region) covariance alongside model-based
  * Poisson + HC1 as an estimator-robustness check
  * leave-one-region-out influence
  * E-values (VanderWeele & Ding 2017) for unmeasured confounding
  * overdispersion diagnostics

Usage: python validation/paper_analysis.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
H = Path(__file__).resolve().parent


def evalue(rr: float) -> float:
    """VanderWeele & Ding E-value: minimum confounder association that could
    explain away an observed risk ratio."""
    r = rr if rr >= 1 else 1.0 / rr
    return r + np.sqrt(r * (r - 1.0))


def design(d: pd.DataFrame, expo_col: str = "passages"):
    ct = pd.get_dummies(d.city, drop_first=True).astype(float).values
    return np.column_stack([np.log(d[expo_col].values), ct])


def nb2(y, x, X0, groups=None):
    """NB2 with alpha estimated by MLE; returns (rr, lo, hi, p, alpha) using
    cluster-robust cov if groups given."""
    X = sm.add_constant(np.column_stack([x, X0]))
    m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=300)
    if groups is not None:
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=300, cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True})
    b, se = m.params[1], m.bse[1]
    z = b / se
    p = 2 * (1 - sm.distributions.norm_gen().cdf(abs(z))) if False else 2 * sm.stats.stattools.stats.norm.sf(abs(z))
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p, m.params[-1]


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

    # --- dispersion diagnostic -------------------------------------------
    for oc in ("all_crashes", "ksi"):
        y = d[oc].values.astype(float)
        mp = sm.GLM(y, sm.add_constant(np.column_stack([d.risk.values, X0])),
                    family=sm.families.Poisson()).fit()
        print(f"{oc:<12} Poisson deviance/df = {mp.deviance/mp.df_resid:6.2f}  "
              f"Pearson chi2/df = {mp.pearson_chi2/mp.df_resid:6.2f}  "
              f"-> {'over-dispersed, NB required' if mp.pearson_chi2/mp.df_resid > 1.5 else 'ok'}")

    CONTRASTS = [
        ("PRIMARY (pre-specified): risk >=65 vs <65", (d.risk >= 65).astype(float).values, None),
        ("EXPLORATORY: risk >=80 vs <80", (d.risk >= 80).astype(float).values, None),
        ("EXPLORATORY: risk >100 vs <30", None, None),
    ]
    print()
    for oc in ("all_crashes", "ksi"):
        y = d[oc].values.astype(float)
        print(f"=== outcome: {oc} " + "=" * 46)
        for label, x, _ in CONTRASTS:
            if x is None:                       # >100 vs <30 restricted comparison
                s = d[(d.risk > 100) | (d.risk < 30)].reset_index(drop=True)
                ys = s[oc].values.astype(float)
                xs = (s.risk > 100).astype(float).values
                Xs = design(s)
                r1 = nb2(ys, xs, Xs)
                r2 = nb2(ys, xs, Xs, groups=s.region.values)
                r3 = poisson_hc(ys, xs, Xs)
                n = len(s)
            else:
                r1 = nb2(y, x, X0)
                r2 = nb2(y, x, X0, groups=d.region.values)
                r3 = poisson_hc(y, x, X0)
                n = len(d)
            print(f"  {label}   (n={n})")
            print(f"     NB2 model SE      RR {r1[0]:5.2f} [{r1[1]:4.2f}, {r1[2]:4.2f}]  "
                  f"p={r1[3]:.4f}   alpha={r1[4]:.3f}")
            print(f"     NB2 cluster SE    RR {r2[0]:5.2f} [{r2[1]:4.2f}, {r2[2]:4.2f}]  p={r2[3]:.4f}")
            print(f"     Poisson HC1       RR {r3[0]:5.2f} [{r3[1]:4.2f}, {r3[2]:4.2f}]  p={r3[3]:.4f}")
            print(f"     E-value point {evalue(r1[0]):.2f}   E-value CI-bound "
                  f"{evalue(max(r1[1],1.0001)) if r1[1] > 1 else 1.0:.2f}")
        print()

    # --- leave-one-region-out on the primary + exploratory ---------------
    print("=== leave-one-region-out (KSI) " + "=" * 38)
    for label, sel in (("risk>=65 vs <65", None), ("risk>100 vs <30", "tail")):
        print(f"  {label}")
        base = d if sel is None else d[(d.risk > 100) | (d.risk < 30)]
        for reg in ["(none)"] + sorted(base.region.unique()):
            s = base if reg == "(none)" else base[base.region != reg]
            s = s.reset_index(drop=True)
            if s.ksi.sum() < 30 or s.city.nunique() < 2:
                continue
            x = ((s.risk >= 65) if sel is None else (s.risk > 100)).astype(float).values
            if x.sum() < 5 or (1 - x).sum() < 5:
                continue
            try:
                rr, lo, hi, p, _ = nb2(s.ksi.values.astype(float), x, design(s))
                flag = "  <-- sign/sig change" if (lo < 1 and sel is None) else ""
                print(f"     drop {reg:<20} RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}] "
                      f"p={p:.3f}  (KSI {int(s.ksi.sum())}){flag}")
            except Exception as e:
                print(f"     drop {reg:<20} failed: {type(e).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
