#!/usr/bin/env python3
"""Per-factor KSI audit of the b46v1 (current production) risk weights.

Re-runs the §8.20-8.22 factor audit against the live generation. The §38
recalibration acted on the previous audit (oneway zeroed, four factors retired,
Maxspeed<51 raised, traffic re-weighted), so this is the calibration CHECK:
under the new weights, does each factor still carry the hazard its points claim?

Design follows factor_severity.py / factor_weights.py:
  * Germany primary (379 sites, one KSI definition, 10-yr Unfallatlas window);
    France and Belgium as replication panels.
  * Per factor: presence indicator -> NB2 KSI-per-passage RR with city fixed
    effects and free exposure elasticity (model SEs; BH-FDR at q=0.10 across
    the factor set per region, as in the original).
  * Per-point calibration (Germany): NB2 beta per POINT of each factor's
    contribution. A calibrated score buys equal hazard per point everywhere;
    the composite score's own per-point slope is the reference line.
  * Road-class control on the German survivors.

Inputs: out/panel_b46_factors_raw.csv (harvested risk_factors JSON per site,
carriageway attribution) + out/panel_final_b46.csv (outcomes).

Usage: python validation/rescore_factor_audit_b46.py
"""
from __future__ import annotations

import json
import re
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
OUT = H / "out"

REGIONS = {
    "de": ("baden-wuerttemberg", "berlin", "berlin-bzm"),
    "fr": ("paris", "toulouse", "nantes", "bordeaux", "telraam-fr"),
    "be": ("belgium", "belgium-new"),
}
CYCLE_KEYS = ("Cycle Track", "Cyclestreet", "Sharrow", "Shared Lane",
              "Bicycle Lanes", "Bicycle Road", "Bicycle Designated",
              "Share Busway")
MIN_EXPOSED = 15
CLASS_MAP = {
    "motorway": "major", "trunk": "major", "trunk_link": "major",
    "primary": "major", "primary_link": "major",
    "secondary": "mid", "secondary_link": "mid", "tertiary": "mid",
    "tertiary_link": "mid", "busway": "mid",
    "residential": "minor", "unclassified": "minor", "service": "minor",
    "living_street": "minor", "platform": "minor",
}


def norm_key(k: str) -> str:
    if k.startswith("Incline"):
        return "Incline"
    if k.startswith("Downhill"):
        return "Downhill"
    return k.replace(" (default)", "")


def load() -> pd.DataFrame:
    raw = pd.read_csv(OUT / "panel_b46_factors_raw.csv")
    panel = pd.read_csv(OUT / "panel_final_b46.csv").reset_index()
    d = panel.merge(raw.rename(columns={"sid": "index"}), on="index",
                    how="inner", suffixes=("", "_f"))
    rows = []
    for _, r in d.iterrows():
        if pd.isna(r.factors):
            continue
        rec = {"index": r["index"], "city": r.city, "passages": r.passages,
               "all_crashes": r.all_crashes, "ksi": r.ksi,
               "risk": r.risk_b46, "klass": CLASS_MAP.get(r.hw_road_f)}
        for k, v in json.loads(r.factors).items():
            k = norm_key(k)
            rec[k] = rec.get(k, 0.0) + float(v)
        rows.append(rec)
    f = pd.DataFrame(rows).fillna(0.0)
    f["Cycle infra (any)"] = (
        f[[c for c in CYCLE_KEYS if c in f]].abs().sum(axis=1) > 0).astype(float) * -1.0
    return f[f.passages > 0].reset_index(drop=True)


def nb2(y, x, d):
    X0 = np.column_stack([np.log(d.passages.values),
                          pd.get_dummies(d.city, drop_first=True).astype(float).values])
    X = sm.add_constant(np.column_stack([np.asarray(x, float), X0]))
    try:
        m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, maxiter=500, method="bfgs")
        b, se = m.params[1], m.bse[1]
        if not (np.isfinite(b) and np.isfinite(se) and abs(b) < 20):
            raise ValueError
    except Exception:
        m = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
        b, se = m.params[1], m.bse[1]
    p = 2 * stats.norm.sf(abs(b / se))
    return b, se, p


def bh(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    passed = pvals[order] <= q * np.arange(1, n + 1) / n
    keep = np.zeros(n, bool)
    if passed.any():
        keep[order[:np.max(np.flatnonzero(passed)) + 1]] = True
    return keep


def factor_cols(d: pd.DataFrame):
    skip = {"index", "city", "passages", "all_crashes", "ksi", "risk", "klass"}
    return [c for c in d.columns
            if c not in skip and (d[c] != 0).sum() >= MIN_EXPOSED]


def presence_block(d: pd.DataFrame, region: str) -> pd.DataFrame:
    cols = factor_cols(d)
    res = []
    y = d.ksi.values.astype(float)
    for c in cols:
        x = (d[c] != 0).astype(float).values
        if x.sum() < MIN_EXPOSED or (1 - x).sum() < MIN_EXPOSED:
            continue
        b, se, p = nb2(y, x, d)
        res.append(dict(factor=c, n_exposed=int(x.sum()),
                        points=round(float(d.loc[d[c] != 0, c].median()), 1),
                        rr=np.exp(b), lo=np.exp(b - 1.96 * se),
                        hi=np.exp(b + 1.96 * se), p=p))
    r = pd.DataFrame(res).sort_values("p").reset_index(drop=True)
    r["bh_pass"] = bh(r.p.values)
    print(f"\n=== {region.upper()} presence audit: KSI per passage "
          f"(n={len(d)}, KSI={int(d.ksi.sum())}) ===")
    print(f"{'factor':<26} {'pts':>6} {'n_exp':>5}  RR [95% CI]           p      FDR")
    for _, x in r.iterrows():
        print(f"{x.factor:<26} {x.points:>6} {x.n_exposed:>5}  "
              f"{x.rr:5.2f} [{x.lo:4.2f}, {x.hi:4.2f}]   {x.p:7.4f}  "
              f"{'*' if x.bh_pass else ''}")
    return r


def perpoint_block(d: pd.DataFrame) -> None:
    cols = factor_cols(d)
    y = d.ksi.values.astype(float)
    print(f"\n=== DE per-point calibration: log-KSI-hazard per point of each "
          f"factor (univariate) ===")
    b0, se0, _ = nb2(y, d.risk.values, d)
    print(f"{'composite score (reference)':<26} beta/pt {b0:+.4f} "
          f"(SE {se0:.4f}) -> RR/10pt {np.exp(10*b0):.2f}")
    rows = []
    for c in cols:
        v = d[c].values.astype(float)
        if np.std(v) == 0:
            continue
        b, se, p = nb2(y, v, d)
        rows.append((c, b, se, p, np.exp(10 * b)))
    for c, b, se, p, rr10 in sorted(rows, key=lambda t: -t[1]):
        cal = ("UNDER-weighted" if b > b0 + 2 * se0 and p < 0.05 else
               "wrong way" if b < -2 * se and p < 0.05 else "")
        print(f"{c:<26} beta/pt {b:+.4f} (SE {se:.4f})  p={p:7.4f} "
              f"RR/10pt {rr10:5.2f}  {cal}")


def class_control(d: pd.DataFrame, survivors) -> None:
    d = d[d.klass.notna()].reset_index(drop=True)
    y = d.ksi.values.astype(float)
    print("\n=== DE survivors under road-class control ===")
    for c in survivors:
        x = (d[c] != 0).astype(float).values
        X0 = np.column_stack([
            np.log(d.passages.values),
            pd.get_dummies(d.city, drop_first=True).astype(float).values,
            pd.get_dummies(d.klass, drop_first=True).astype(float).values])
        X = sm.add_constant(np.column_stack([x, X0]))
        try:
            m = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
                disp=0, maxiter=500, method="bfgs")
            b, se = m.params[1], m.bse[1]
        except Exception:
            m = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
            b, se = m.params[1], m.bse[1]
        p = 2 * stats.norm.sf(abs(b / se))
        print(f"{c:<26} +class RR {np.exp(b):5.2f} "
              f"[{np.exp(b-1.96*se):4.2f}, {np.exp(b+1.96*se):4.2f}]  p={p:.4f}")


def main() -> int:
    f = load()
    results = {}
    for region, cities in REGIONS.items():
        d = f[f.city.isin(cities)].reset_index(drop=True)
        if len(d) < 50:
            continue
        results[region] = presence_block(d, region)
    de = f[f.city.isin(REGIONS["de"])].reset_index(drop=True)
    perpoint_block(de)
    surv = results["de"][results["de"].bh_pass].factor.tolist()
    if surv:
        class_control(de, surv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
