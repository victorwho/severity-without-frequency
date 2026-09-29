#!/usr/bin/env python3
"""Headline validation contrasts re-run on the b46v1 re-scored panel.

Same specification as paper_analysis.py / threshold_family.py (NB2 with alpha by
MLE, city fixed effects, free log-exposure elasticity; region-cluster SEs), run
four ways:

  A  stored b36v1 risk        -- must reproduce the published §8.31/§8.32 numbers
  B  refetched b36v1 risk     -- pipeline-fidelity check (same SQL as C)
  C  b46v1 risk, fixed cuts   -- the current algorithm on the product's absolute scale
  D  b46v1 risk, matched cuts -- cuts placed at the same site-share quantiles as
                                 the b36v1 cuts, so discrimination is compared
                                 like-for-like despite the scale inflation
                                 (b46 mean +12.9, SD 28.4 -> 37.9)

Plus per-SD (z-scored within panel) continuous contrasts -- the preflight metric
-- and the §8.32-style band rate table on both scores.

Usage: python validation/rescore_analysis.py
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

CUTS = (65, 80, 90, 100)
BANDS = [-np.inf, 30, 42, 65, 80, 90, 100, np.inf]
BAND_LABELS = ["<30", "30-42", "42-65", "65-80", "80-90", "90-100", ">100"]


def design(d: pd.DataFrame) -> np.ndarray:
    ct = pd.get_dummies(d.city, drop_first=True).astype(float).values
    return np.column_stack([np.log(d.passages.values), ct])


def nb2(y, x, X0, groups):
    from scipy import stats
    X = sm.add_constant(np.column_stack([x, X0]))
    try:
        base = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=500, method="bfgs")
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=500, start_params=base.params, cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True})
        b, se = m.params[1], m.bse[1]
        if not (np.isfinite(b) and np.isfinite(se) and abs(b) < 20):
            raise ValueError("degenerate NB2 fit")
    except Exception:
        # Poisson with cluster-robust SEs as the documented fallback
        m = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        b, se = m.params[1], m.bse[1]
    p = 2 * stats.norm.sf(abs(b / se))
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


def contrasts(d: pd.DataFrame, risk_col: str, cuts, label: str) -> None:
    d = d[d[risk_col].notna() & (d.passages > 0)].reset_index(drop=True)
    X0 = design(d)
    g = d.city.values
    print(f"\n--- {label}  (n={len(d)}, KSI={int(d.ksi.sum())}) " + "-" * 30)
    z = (d[risk_col] - d[risk_col].mean()) / d[risk_col].std()
    for oc in ("all_crashes", "ksi"):
        y = d[oc].values.astype(float)
        rr, lo, hi, p = nb2(y, z.values, X0, g)
        print(f"  {oc:<11} per-SD           RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}]  p={p:.4f}")
        for cut, cval in cuts:
            x = (d[risk_col] >= cval).astype(float).values
            n_hi = int(x.sum())
            rr, lo, hi, p = nb2(y, x, X0, g)
            print(f"  {oc:<11} >={cut:>3} (@{cval:6.1f}) RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}]"
                  f"  p={p:.4f}  (sites {n_hi})")
        # tail vs very-safe reference: >100-equivalent vs <30-equivalent
        top = dict(cuts).get(100)
        ref = dict(cuts).get(30, 30.0)
        s = d[(d[risk_col] > top) | (d[risk_col] < ref)].reset_index(drop=True)
        if s.city.nunique() >= 2 and len(s) > 50:
            xs = (s[risk_col] > top).astype(float).values
            rr, lo, hi, p = nb2(s[oc].values.astype(float), xs, design(s), s.city.values)
            print(f"  {oc:<11} >100eq vs <30eq  RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}]"
                  f"  p={p:.4f}  (n={len(s)}, top {int(xs.sum())})")


def band_table(d: pd.DataFrame, risk_col: str, label: str) -> None:
    d = d[d[risk_col].notna() & (d.passages > 0)]
    b = pd.cut(d[risk_col], BANDS, labels=BAND_LABELS, right=False)
    t = d.groupby(b, observed=False).agg(
        sites=("risk", "size"), crashes=("all_crashes", "sum"),
        ksi=("ksi", "sum"), Mpass=("passages", lambda v: v.sum() / 1e6))
    t["perM_all"] = t.crashes / t.Mpass
    t["perM_ksi"] = t.ksi / t.Mpass
    print(f"\n{label} band table:")
    print(t.round(3).to_string())


def loro(d: pd.DataFrame, risk_col: str, cval: float, oc: str = "ksi") -> None:
    d = d[d[risk_col].notna() & (d.passages > 0)].reset_index(drop=True)
    print(f"\nleave-one-region-out, {oc} >= {cval:.0f} (b46):")
    for reg in ["(none)"] + sorted(d.city.unique()):
        s = d if reg == "(none)" else d[d.city != reg]
        s = s.reset_index(drop=True)
        x = (s[risk_col] >= cval).astype(float).values
        if x.sum() < 5 or s.city.nunique() < 2:
            continue
        try:
            rr, lo, hi, p = nb2(s[oc].values.astype(float), x, design(s), s.city.values)
            flag = "  <-- crosses 1" if lo < 1 else ""
            print(f"  drop {reg:<20} RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}] p={p:.3f}{flag}")
        except Exception as e:
            print(f"  drop {reg:<20} failed: {type(e).__name__}")


def main() -> int:
    d = pd.read_csv(H / "out" / "panel_final_b46.csv")
    d = d[d.passages > 0].reset_index(drop=True)

    fixed = [(c, float(c)) for c in (30,) + CUTS]
    contrasts(d, "risk_old_stored", fixed, "A: stored b36v1 (published replication)")
    contrasts(d, "risk_old_refetch", fixed, "B: refetched b36v1 (pipeline check)")
    contrasts(d, "risk_b46", fixed, "C: b46v1, fixed cuts")

    # D: percentile-matched cuts -- same share of sites above each cut as b36v1
    both = d[d.risk_b46.notna() & d.risk_old_refetch.notna()]
    matched = []
    for c in (30,) + CUTS:
        share = (both.risk_old_refetch >= c).mean()
        matched.append((c, float(both.risk_b46.quantile(1 - share))))
    print("\nmatched cuts (b36 cut -> b46 equivalent): "
          + ", ".join(f"{c}->{v:.1f}" for c, v in matched))
    contrasts(d, "risk_b46", matched, "D: b46v1, percentile-matched cuts")

    band_table(d, "risk_old_refetch", "b36v1 (refetch)")
    band_table(d, "risk_b46", "b46v1 (fixed absolute bands)")

    loro(d, "risk_b46", 100.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
