# Severity Without Frequency

A multi-country validation of a deployed, safety-weighted cycling routing risk score against real injury outcomes.

**Read the paper: [paper.md](paper.md)**

## Summary

Cycling-safety routing products increasingly steer riders using a composite, per-segment risk score, but such scores are rarely validated against exposure-adjusted injury outcomes — and where they are, crash frequency and severity are almost always validated as one undifferentiated outcome. This paper tests whether a deployed cycling routing risk score predicts frequency and severity differently, once cyclist exposure is properly accounted for.

Using a five-country, 13-region counter-exposure panel of 2,034 sites recording approximately 3.98 billion cyclist passages, 10,399 crashes, and 1,169 killed-or-seriously-injured (KSI) outcomes:

- The score's relationship to **raw crash frequency** is weak and statistically inconsistent across estimators and cut-points, and does not survive control for road classification.
- The score's relationship to **crash severity** (KSI) is significant at every cut-point tested, stable across all 13 leave-one-region-out checks, strengthens under a later, independently re-scored model generation, and — for the first time across the generations examined — survives full road-class control, with the surviving signal concentrated at mid-block locations rather than junctions.

A validated risk score therefore predicts *how bad* a crash is likely to be if one occurs, not *how likely* a crash is to occur at all — a distinction the surrounding literature rarely tests for, and one with direct implications for how such a score should be described to the people routing on it.

## Repository contents

| Path | Contents |
|---|---|
| [`paper.md`](paper.md) | The full paper (Vancouver-style numbered citations, DOI-linked references). |
| [`references.bib`](references.bib) | BibTeX bibliography for the 38 works cited in the paper. |
| [`methods/`](methods) | The statistical analysis scripts used to produce every table in the paper: negative-binomial regression, threshold/E-value analysis, road-class control, the mid-block/junction split, the per-factor KSI audit, band-resolution testing, and the city-level criterion validation. |
| [`results/`](results) | The raw output logs those scripts produced — the exact numbers reported in the paper's tables, before any formatting. |
| [`CITATION.cff`](CITATION.cff) | Machine-readable citation metadata (GitHub's "Cite this repository" feature reads this). |

## What is deliberately not included

The routing product's scoring algorithm (its Lua implementation) and the point-value weights assigned to individual scoring factors are proprietary to the product this score serves, and are not published here or anywhere else. Everything in this repository reports the **statistical relationship** between the score and real-world outcomes — relative risks, confidence intervals, p-values, and the *sign* (protective or harmful) of individual named factors — never the underlying formula or its coefficients. This is stated explicitly in the paper's Methods section (§2.2, §2.6) and is not an omission from the reproducibility bundle; it is the boundary the bundle was built around.

For the same reason, this repository does not include the raw per-site panel data (site-level risk scores, coordinates, and outcomes). The scripts in `methods/` operate on that panel, and the `results/` logs are their output; the panel itself is withheld because, at the resolution of ~2,000 individual road segments, it would let a sufficiently motivated reader statistically recover coefficients that the aggregate figures reported in the paper do not reveal.

## Data availability

Crash and counter/passage data underlying the panel come from public police, government, and city-planner sources across Germany, Switzerland, Belgium, France, and Spain, described in the paper's Methods (§2.1). The city-level index data (§2.8, §3.11) derives from the routing product's own published multi-city risk index.

## Reproducing the analysis

Each script in `methods/` is self-contained and documented at its top with what it does and which panel/log file it reads or writes. They were run against internal panel data that is not included in this repository (see above); they are published so the statistical method — not the input data — can be inspected and checked against what the paper reports.

## Citation

See [`CITATION.cff`](CITATION.cff), or cite as:

> Rotariu, V. (2026). *Severity Without Frequency: A Multi-Country Validation of a Cycling Routing Risk Score.*

## License

[CC BY 4.0](LICENSE) — the paper, bibliography, methods scripts, and result logs in this repository may be shared and adapted for any purpose, including commercially, with attribution. The proprietary scoring algorithm and factor weights described in the paper's Methods are explicitly out of scope for this license (see "What is deliberately not included," above).

## Status

This is an AI-assisted draft (built with [opendraft](https://github.com/federicodeponte/opendraft)), produced from real internal validation data and literature verified against Crossref/OpenAlex/DataCite. It has not undergone external peer review. Read it accordingly, and see the paper's own Limitations section (§4.7) for what it does not establish.
