# Scoring Notes and Variants

This document explains what the base script measures, highlights potential inconsistencies or interpretation pitfalls, and defines two alternative variants:

- A “tighter scoring” version that is more conservative by construction
- A “modernized” version that merges legacy/synonymous ancestry labels into a single category

## What the base script measures

- Unit of analysis: persons (per‑capita), using ACS PUMS 2023 1‑Year microdata.
- Outcome: whether a person lives in a household that received SNAP at any time in the last 12 months.
- Per‑capita rate for an ancestry group A:
  - numerator = sum of person weights (PWGTP) for people of ancestry A who live in a SNAP household (FS = 1)
  - denominator = sum of person weights (PWGTP) for people of ancestry A
- Ancestry definition: uses ANC1P where available; if missing/zero, falls back to ANC2P; otherwise “Unclassified”.
- Labels: loaded from the 2023 PUMS data dictionary; exact wording reflects Census terminology.

## Important interpretation notes

- Household SNAP vs. individual recipients: The numerator includes all persons living in a SNAP recipient household, not only the enrolled individuals.
- Missing FS handling: Missing household SNAP values are treated as False (non‑SNAP) during the person–household join. This depresses rates slightly and is conservative.
- Group Quarters (GQ): By default, GQ residents appear in the person file and are usually not in SNAP households; including them in the denominator but not the numerator can also depress rates.
- Primary vs. secondary ancestry: Using ANC1P primarily and ANC2P as fallback avoids double counting, but people who report an ancestry only in ANC2P may still appear via the fallback. A strictly primary‑only definition would reduce denominators and numerators for some groups.
- Point estimates only: Results use PWGTP for point estimates; replicate weights (variance/MOEs) are not computed here.

## Tighter scoring (more conservative) — script_tighter_scoring.py

Objectives:
- Exclude Group Quarters (GQ) from both numerator and denominator; count Housing Units (HU) only.
- Use only primary ancestry (ANC1P); do not fall back to ANC2P.
- Continue treating missing FS as non‑SNAP during join.

Implementation details:
- Identify HU vs. GQ using `SERIALNO` prefix: `SERIALNO.str[4:6] == 'HU'` ⇒ HU; `'GQ'` ⇒ GQ.
- Household SNAP Series is built only from HU records; person records are filtered to HU only before aggregation.
- Ancestry code is taken strictly from ANC1P (>0); if not available, assign `-1` (“Unclassified”).

Implications:
- Excluding GQ tends to “score down,” because GQ residents (often not SNAP) are removed from denominators.
- Ignoring ANC2P avoids enlarging denominators via secondary ancestry; this can also “score down” depending on group patterns.

## Modernized categories — script_modern.py

Objective:
- Merge legacy/synonymous labels into a single, modern category for reporting clarity (e.g., “Afro American”, “Afro‑American”, “Negro”, and any direct “Black” label) while keeping specific African ancestries (e.g., Nigerian, Somali) distinct.

Implementation details:
- After attaching Census labels, apply a mapping that normalizes exact label matches:
  - {"Afro American", "Afro-American", "African American", "African-American", "Black", "Negro"} ⇒ "African American"
- Aggregate results by the normalized label: sum people and people_in_snap_hh, recompute rate.
- All other labels remain unchanged.

Notes and caveats:
- We intentionally do NOT merge specific African ancestries (e.g., “Nigerian”, “Ghanaian”, “Somali”) into “African American”. Those remain as‑is.
- We match exact labels only to avoid unintended merges (e.g., avoiding terms like “Black Sea German”).

## Reproducibility

- All variants read the same PUMS inputs; the tighter/modern scripts adjust only filtering and/or labeling logic.
- All scripts support `--anc-labels auto` to fetch and use the official 2023 PUMS data dictionary.

## Suggested usage

- Base script (balanced default): `script.py`
- Conservative (“score down”): `script_tighter_scoring.py`
- Modernized categories: `script_modern.py`

