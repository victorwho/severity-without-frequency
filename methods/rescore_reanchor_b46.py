#!/usr/bin/env python3
"""Band re-anchoring on the b46v1 scale — the run reanchor_bands.py asked for.

Inputs
  out/panel_final_b46.csv       panel with b46v1 risk (rescore_panel_b46.py)
  out/hist_road_risk_data*.csv  1-pt network histograms, km + segment count,
                                TABLESAMPLE SYSTEM(2) of both generations

Findings this run produced (2026-09-04, recorded in plan §8.37):
  * Unchanged cuts on b46v1 scores: 3 inversions, 0 separations — the live
    display is mis-anchored (>100 "extreme" alone jumped 2.18% -> 5.16% of km).
  * Percentile-matching all 8 old cuts is DEGENERATE: old 65 and 80 both land
    at b46 ~80 — the re-weights collapsed the old 65-80 band (the same band the
    outcome data flagged as anomalous), so 8-band continuity cannot exist.
  * Winner: cuts [42, 80, 105, 130] -> adjusted KSI ladder
    1.00 / 1.04 / 1.27 / 1.57 / 1.94, zero inversions; grouped for claims as
    3 tiers (<42 / 42-80 / >80) with the >80 tier shaded at 105 and 130.

Usage: python validation/rescore_reanchor_b46.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reanchor_bands import evaluate  # noqa: E402

H = Path(__file__).resolve().parent
OLD_CUTS = [30, 42, 50, 58, 65, 80, 100]
CANDS = {
    "status quo 8 on b46": OLD_CUTS,
    "6 [32,41,60,80,120]": [32, 41, 60, 80, 120],
    "5 [42,80,105,130]  ": [42, 80, 105, 130],
    "5 [41,80,105,130]  ": [41, 80, 105, 130],
    "4 [41,80,130]      ": [41, 80, 130],
    "3 [41,120]         ": [41, 120],
}
FINAL = [42, 80, 105, 130]          # claims-only skeleton (zero inversions)
SHADES = [32, 42, 50, 60, 70, 80, 90, 105, 130]   # adopted display: 3 claims / 10 shades


def main() -> int:
    d = pd.read_csv(H / "out" / "panel_final_b46.csv")
    d = d[(d.passages > 0) & d.risk_b46.notna()].copy()
    d["risk"] = d.risk_b46
    d = d.reset_index(drop=True)
    print(f"panel: {len(d)} sites, {int(d.ksi.sum())} KSI (b46v1 scores)\n")
    print(f"{'scale':<22}{'bands':>6}{'inv':>5}{'sep':>5}   adjusted RR ladder")
    for name, cuts in CANDS.items():
        rr, ci, inv, sep, b, edges = evaluate(d, cuts)
        print(f"{name:<22}{len(rr):6d}{inv:5d}{sep:5d}   "
              + " -> ".join(f"{r:.2f}" for r in rr))

    rr, ci, inv, sep, b, edges = evaluate(d, FINAL)
    print(f"\nFINAL {FINAL}: per-band panel detail")
    for i in range(len(rr)):
        m = b == i
        mp = d.passages[m].sum() / 1e6
        k = int(d.ksi[m].sum())
        lo = "-inf" if np.isinf(edges[i]) else f"{edges[i]:.0f}"
        hi = "inf" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:.0f}"
        print(f"  {lo + '-' + hi:>10}  sites {int(m.sum()):5d}  KSI {k:4d}  "
              f"KSI/M {k / mp:6.3f}  adjRR {rr[i]:5.2f} [{ci[i][0]:4.2f},{ci[i][1]:5.2f}]")

    new = pd.read_csv(H / "out" / "hist_road_risk_data.csv").sort_values("bin")
    old = pd.read_csv(H / "out" / "hist_road_risk_data_old_b46v1.csv")
    tot_km, tot_n = new.km.sum(), new.n.sum()
    print("\nb46v1 network paint shares under the FINAL cuts:")
    for lo, hi, lab in ((-1, 42, "Safer <42"), (42, 80, "Typical 42-80"),
                        (80, 105, "High 80-105"), (105, 130, "Very high 105-130"),
                        (130, 999, "Extreme >130")):
        m = (new.bin >= lo) & (new.bin < hi)
        print(f"  {lab:<20} {100 * new.km[m].sum() / tot_km:6.2f}% of km   "
              f"{100 * new.n[m].sum() / tot_n:6.2f}% of segments")
    rr, ci, inv, sep, b, edges = evaluate(d, SHADES)
    print(f"\nADOPTED 10-shade display {SHADES}: {inv} within-tier inversions "
          f"(expected; claims attach to tiers only)")
    for i in range(len(rr)):
        m = b == i
        mp = d.passages[m].sum() / 1e6
        k = int(d.ksi[m].sum())
        lo = "-inf" if np.isinf(edges[i]) else f"{edges[i]:.0f}"
        hi = "inf" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:.0f}"
        print(f"  {lo + '-' + hi:>10}  sites {int(m.sum()):5d}  KSI {k:4d}  "
              f"KSI/M {k / mp:6.3f}  adjRR {rr[i]:5.2f}")

    s_old = old.km[old.bin >= 100].sum() / old.km.sum()
    print(f"\n(low-zoom sanity: new Extreme >130 = "
          f"{100 * new.km[new.bin >= 130].sum() / tot_km:.2f}% of km vs old >100 "
          f"= {100 * s_old:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
