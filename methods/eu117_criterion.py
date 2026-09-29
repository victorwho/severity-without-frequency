#!/usr/bin/env python3
"""Criterion validation: does a city's index predict its injury rate?

The 26-city version could not answer this. The outcome panel and the pilot index
shared three cities, and no correlation is computable at n = 3, so the companion
paper had to argue transportability indirectly — via the SCORE's between-region
behaviour rather than the INDEX's own values.

The 117-city frame overlaps the panel far more. Counter sites carry coordinates,
so regional panels (Baden-Wurttemberg, Belgium) resolve into whichever indexed
city actually contains them, and city panels are confirmed rather than assumed.

The test: regress a city's killed-or-seriously-injured count on its published
index, with log-passages as offset, and report the rate ratio per 10 index points.
A city ranking that means anything should show injuries falling as the index rises.

Sites are attributed by point-in-polygon to the same boundaries the index uses, so
this compares like with like — unlike the paper's Sec. 3.6, which had to use the
mean risk of counter sites, a quantity Berlin's two networks disagreed about by 26
points.

Usage:
    python ccri_pilot/eu117_criterion.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eu117  # noqa: E402

VAL = os.path.join(os.path.dirname(HERE), "validation", "out")
GEO = os.path.join(VAL, "panel_v2_geo.csv")
FINAL = os.path.join(VAL, "panel_final.csv")
_SHADOW = "--shadow" in sys.argv
INDEX = os.path.join(
    os.environ.get("EU117_RESULTS_SHADOW" if _SHADOW else "EU117_RESULTS")
    or os.path.join(HERE, "results117s" if _SHADOW else "results117"),
    "index117.json")

# Panels that ARE a single indexed city, so need no point-in-polygon step.
WHOLE_CITY = {"zurich": "zurich", "basel": "basel"}


def attribute_sites():
    """Panel sites -> indexed city, by point-in-polygon on the index boundaries."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    d = pd.read_csv(GEO)
    d = d[d["lon"].notna() & d["lat"].notna()].copy()
    polys = {}
    for slug in eu117.available():
        try:
            polys[slug] = prep(eu117.load_boundary(slug))
        except Exception:
            pass
    # only test cities whose bbox could plausibly contain the panel
    bounds = {s: eu117.load_boundary(s).bounds for s in polys}

    assigned = []
    for _, r in d.iterrows():
        p = Point(r["lon"], r["lat"])
        hit = None
        for s, b in bounds.items():
            if b[0] <= r["lon"] <= b[2] and b[1] <= r["lat"] <= b[3] and polys[s].covers(p):
                hit = s
                break
        assigned.append(hit)
    d["slug"] = assigned
    return d


def main() -> int:
    if not os.path.exists(INDEX):
        print("run eu117_index.py first")
        return 1
    idx = json.load(open(INDEX, encoding="utf-8"))
    ival = {c["slug"]: c for c in idx["cities"]}

    d = attribute_sites()
    inside = d[d["slug"].notna()]
    print(f"panel sites with coordinates: {len(d):,}; "
          f"inside an indexed city: {len(inside):,} "
          f"({100*len(inside)/len(d):.0f}%)")
    print(f"regions represented: {sorted(d['city'].unique())}\n")

    g = inside.groupby("slug").agg(
        sites=("risk", "size"), passages=("passages", "sum"),
        ksi=("ksi", "sum"), all_crashes=("all_crashes", "sum"),
        site_risk=("risk", "mean")).reset_index()

    # whole-city panels that postdate the geocoded file
    fin = pd.read_csv(FINAL)
    for region, slug in WHOLE_CITY.items():
        sub = fin[fin["city"] == region]
        if not len(sub) or slug not in ival:
            continue
        g = pd.concat([g, pd.DataFrame([{
            "slug": slug, "sites": len(sub),
            "passages": sub["passages"].sum(), "ksi": sub["ksi"].sum(),
            "all_crashes": sub["all_crashes"].sum(),
            "site_risk": sub["risk"].mean()}])], ignore_index=True)

    g = g[g["slug"].isin(ival) & (g["passages"] > 0)].copy()
    g["index"] = g["slug"].map(lambda s: ival[s]["index"])
    g["tier"] = g["slug"].map(lambda s: ival[s]["tier"])
    g["label"] = g["slug"].map(lambda s: ival[s]["label"])
    g["ksi_per_10m"] = 1e7 * g["ksi"] / g["passages"]
    g = g.sort_values("index", ascending=False)

    print(f"{len(g)} indexed cities have outcome data "
          f"(the 26-city version had 3)\n")
    print(f"   {'city':<16}{'index':>7}{'tier':>6}{'sites':>7}{'passages':>12}"
          f"{'KSI':>6}{'KSI/10M':>10}")
    for _, r in g.iterrows():
        print(f"   {r['label'][:16]:<16}{r['index']:>7.1f}{int(r['tier']):>6}"
              f"{int(r['sites']):>7}{r['passages']/1e6:>11.0f}M{int(r['ksi']):>6}"
              f"{r['ksi_per_10m']:>10.2f}")

    if len(g) < 5:
        print("\ntoo few cities for a correlation")
        return 0

    rho = g["index"].corr(g["ksi_per_10m"], method="spearman")
    print(f"\n   Spearman(index, KSI per passage) = {rho:+.3f}  (n={len(g)})")
    print("   negative is the hypothesised direction: higher index = safer")

    y = g["ksi"].values.astype(float)
    X = sm.add_constant(g[["index"]].astype(float)).values
    off = np.log(g["passages"].values.astype(float))
    aux = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
    mu = np.asarray(aux.fittedvalues)
    alpha = max(1e-6, float(np.asarray(
        sm.OLS(((y - mu) ** 2 - y) / mu, mu).fit().params)[0]))
    m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
               offset=off).fit()
    b = np.asarray(m.params)[1]
    ci = np.asarray(m.conf_int())[1]
    print(f"   KSI rate ratio per +10 index points: {np.exp(10*b):.2f} "
          f"[{np.exp(10*ci[0]):.2f}, {np.exp(10*ci[1]):.2f}]  "
          f"p={np.asarray(m.pvalues)[1]:.3f}  (alpha={alpha:.2f})")

    # Country dominates raw KSI rates: definitions of "serious", reporting rates
    # and trauma care differ far more between states than cycling networks do
    # within them. Without this control the headline is a country effect.
    g["iso2"] = g["slug"].map(lambda s: ival[s]["iso2"].upper())
    cf = pd.get_dummies(g["iso2"], drop_first=True).astype(float)
    Xc = sm.add_constant(pd.concat(
        [g[["index"]].astype(float).reset_index(drop=True),
         cf.reset_index(drop=True)], axis=1)).values
    try:
        auxc = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=off).fit()
        muc = np.asarray(auxc.fittedvalues)
        ac = max(1e-6, float(np.asarray(
            sm.OLS(((y - muc) ** 2 - y) / muc, muc).fit().params)[0]))
        mc = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=ac),
                    offset=off).fit()
        bc = np.asarray(mc.params)[1]
        cic = np.asarray(mc.conf_int())[1]
        print(f"   with COUNTRY fixed effects:            {np.exp(10*bc):.2f} "
              f"[{np.exp(10*cic[0]):.2f}, {np.exp(10*cic[1]):.2f}]  "
              f"p={np.asarray(mc.pvalues)[1]:.3f}   "
              f"({g['iso2'].nunique()} countries, {len(g)} cities)")
    except Exception as exc:
        print(f"   country-adjusted model not estimable: {type(exc).__name__}")

    print(f"\n   within-country ordering (the part a country effect cannot explain):")
    for cc, sub in g.groupby("iso2"):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("index", ascending=False)
        ok = (sub["ksi_per_10m"].is_monotonic_increasing)
        arrow = "consistent" if ok else "INCONSISTENT"
        detail = ", ".join(f"{r['label']} {r['index']:.0f}/{r['ksi_per_10m']:.2f}"
                           for _, r in sub.iterrows())
        print(f"     {cc}: {arrow:<12} {detail}")

    ya = g["all_crashes"].values.astype(float)
    auxa = sm.GLM(ya, X, family=sm.families.Poisson(), offset=off).fit()
    mua = np.asarray(auxa.fittedvalues)
    aa = max(1e-6, float(np.asarray(
        sm.OLS(((ya - mua) ** 2 - ya) / mua, mua).fit().params)[0]))
    ma = sm.GLM(ya, X, family=sm.families.NegativeBinomial(alpha=aa),
                offset=off).fit()
    ba = np.asarray(ma.params)[1]
    cia = np.asarray(ma.conf_int())[1]
    print(f"\n   ALL crashes per +10 index points:      {np.exp(10*ba):.2f} "
          f"[{np.exp(10*cia[0]):.2f}, {np.exp(10*cia[1]):.2f}]  "
          f"p={np.asarray(ma.pvalues)[1]:.3f}")

    if g["tier"].nunique() > 1:
        print("\n   by tier:")
        for t, sub in g.groupby("tier"):
            print(f"     Tier {int(t)}: {len(sub)} cities, "
                  f"{1e7*sub['ksi'].sum()/sub['passages'].sum():.2f} KSI per 10M "
                  f"({', '.join(sub['label'])})")

    g.to_csv(os.path.join(os.path.dirname(INDEX), "criterion.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
