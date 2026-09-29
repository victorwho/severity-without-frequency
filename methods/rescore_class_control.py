#!/usr/bin/env python3
"""Road-class-controlled contrasts on the b46v1 re-scored panel (§8.34 follow-up).

§8.31 caveat 1: every full-panel number so far is unadjusted for road class, and
§8.17 showed class adjustment halved the b36v1 KSI effect (1.36 -> 1.20 ns) with
only the mid-block component surviving (§8.18: 1.48 [1.02,2.13]). Class was never
harvested for the 797 sites added after §8.17. The b46v1 rescore fetched each
site's carriageway `highway` tag, so the class-controlled version now runs on the
FULL 2,033-site panel, for both generations, with the same spec as
rescore_analysis.py (NB2 alpha-MLE, city FE, free elasticity, cluster SEs) plus
ordered-class dummies (major/mid/minor, final_analysis.py's CLASS_MAP).

Interpretation guard (plan §1, confound 2): the score is partly built FROM class,
so conditioning on class is conservative by construction; report both.

Also: mid-block (link) KSI with class control on the 1,186-site subset that
carries the junction/link split (panel_v2), replicating §8.18 under b46v1.

Usage: python validation/rescore_class_control.py
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

CLASS_MAP = {
    "motorway": "major", "trunk": "major", "trunk_link": "major",
    "primary": "major", "primary_link": "major",
    "secondary": "mid", "secondary_link": "mid", "tertiary": "mid",
    "tertiary_link": "mid", "busway": "mid",
    "residential": "minor", "unclassified": "minor", "service": "minor",
    "living_street": "minor", "platform": "minor",
}
# matched cuts from rescore_analysis.py (same site share as the b36v1 cuts)
MATCHED = {"risk_old_refetch": [(65, 65.0), (80, 80.0), (100, 100.0)],
           "risk_b46": [(65, 79.5), (80, 104.6), (100, 129.5)]}


def fit(y, x, d, with_class, extra=None):
    cols = [np.log(d.passages.values),
            pd.get_dummies(d.city, drop_first=True).astype(float).values]
    if with_class:
        cols.append(pd.get_dummies(d.klass, drop_first=True).astype(float).values)
    if extra is not None:
        cols.append(extra)
    X = sm.add_constant(np.column_stack([np.asarray(x, float)] +
                                        [np.column_stack([c]) if c.ndim == 1 else c
                                         for c in cols]))
    try:
        base = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=500, method="bfgs")
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=500, start_params=base.params, cov_type="cluster",
            cov_kwds={"groups": d.city.values, "use_correction": True})
        b, se = m.params[1], m.bse[1]
        if not (np.isfinite(b) and np.isfinite(se) and abs(b) < 20):
            raise ValueError
    except Exception:
        m = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": d.city.values})
        b, se = m.params[1], m.bse[1]
    p = 2 * stats.norm.sf(abs(b / se))
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


def block(d, risk_col, label):
    d = d[d[risk_col].notna()].reset_index(drop=True)
    print(f"\n--- {label}  (n={len(d)}, KSI={int(d.ksi.sum())}) " + "-" * 28)
    z = ((d[risk_col] - d[risk_col].mean()) / d[risk_col].std()).values
    for oc in ("all_crashes", "ksi"):
        y = d[oc].values.astype(float)
        for name, x in [("per-SD", z)] + [
                (f">={c:>3} (@{v:5.1f})", (d[risk_col] >= v).astype(float).values)
                for c, v in MATCHED[risk_col]]:
            r0 = fit(y, x, d, with_class=False)
            r1 = fit(y, x, d, with_class=True)
            print(f"  {oc:<11} {name:<15} marginal {r0[0]:5.2f} [{r0[1]:4.2f},{r0[2]:4.2f}] "
                  f"p={r0[3]:.4f}   +class {r1[0]:5.2f} [{r1[1]:4.2f},{r1[2]:4.2f}] p={r1[3]:.4f}")


def midblock(d):
    v2 = pd.read_csv(H / "out" / "panel_v2.csv")[
        ["city", "passages", "j_all", "j_ksi", "l_all", "l_ksi"]]
    m = d.merge(v2, on=["city", "passages"], how="inner")
    m = m[m.risk_b46.notna() & m.klass.notna()].reset_index(drop=True)
    print(f"\n--- mid-block / junction split, b46v1, class-controlled "
          f"(n={len(m)}, link KSI={int(m.l_ksi.sum())}, junction KSI={int(m.j_ksi.sum())}) ---")
    for risk_col in ("risk_old_refetch", "risk_b46"):
        z = ((m[risk_col] - m[risk_col].mean()) / m[risk_col].std()).values
        cuts = MATCHED[risk_col]
        for oc in ("l_ksi", "j_ksi", "l_all", "j_all"):
            y = m[oc].values.astype(float)
            for name, x in [("per-SD", z)] + [
                    (f">={c:>3} (@{v:5.1f})", (m[risk_col] >= v).astype(float).values)
                    for c, v in cuts]:
                r1 = fit(y, x, m, with_class=True)
                print(f"  {risk_col:<16} {oc:<6} {name:<15} +class "
                      f"{r1[0]:5.2f} [{r1[1]:4.2f},{r1[2]:4.2f}] p={r1[3]:.4f}")
            print()


def loro(d):
    print("\nleave-one-region-out, class-controlled KSI, b46v1 >=104.6 (top 11.2%):")
    for reg in ["(none)"] + sorted(d.city.unique()):
        s = d if reg == "(none)" else d[d.city != reg]
        s = s[s.risk_b46.notna()].reset_index(drop=True)
        x = (s.risk_b46 >= 104.6).astype(float).values
        if x.sum() < 5 or s.city.nunique() < 2:
            continue
        try:
            rr, lo, hi, p = fit(s.ksi.values.astype(float), x, s, with_class=True)
            flag = "  <-- crosses 1" if lo < 1 else ""
            print(f"  drop {reg:<20} RR {rr:5.2f} [{lo:4.2f}, {hi:4.2f}] p={p:.3f}{flag}")
        except Exception as e:
            print(f"  drop {reg:<20} failed: {type(e).__name__}")


def main() -> int:
    d = pd.read_csv(H / "out" / "panel_final_b46.csv")
    d = d[(d.passages > 0)].copy()
    d["klass"] = d.hw_road.map(CLASS_MAP)
    d = d[d.klass.notna()].reset_index(drop=True)
    print("class composition:", d.klass.value_counts().to_dict())
    print("mean risk_b46 by class:",
          d.groupby("klass").risk_b46.mean().round(1).to_dict())

    block(d, "risk_old_refetch", "b36v1 (refetch), matched=fixed cuts")
    block(d, "risk_b46", "b46v1, percentile-matched cuts")
    midblock(d)
    loro(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
