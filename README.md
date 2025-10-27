# SNAP Per‑Capita by Ancestry (ACS PUMS 2023)

This repo contains a Python script to compute per‑capita SNAP participation by detailed ancestry using the ACS Public Use Microdata Sample (PUMS) 2023 1‑Year files.

What it does
- Joins household records (SNAP receipt in last 12 months) to person records by `SERIALNO`.
- Aggregates person weights (`PWGTP`) by detailed ancestry codes (`ANC1P`/`ANC2P`).
- Outputs a CSV and bar charts (Top‑N and All groups) with percent labels.

Requirements
- Python 3.9+ recommended
- Packages: `pandas`, `numpy`, `matplotlib`
  - Install: `pip install pandas numpy matplotlib`

Get the data (2023 1‑Year ACS PUMS)
- Download official CSV zips from the U.S. Census Bureau:
  - Households: https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hus.zip
  - Persons:    https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip

Example using curl from repo root:
- macOS/Linux
  - `curl -O https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hus.zip`
  - `curl -O https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pus.zip`
  - `unzip -j csv_hus.zip 'psam_hus*.csv' -d snap_benefits/`
  - `unzip -j csv_pus.zip 'psam_pus*.csv' -d snap_benefits/`

Run the script
- From repo root, either cd into the folder or call the script with paths:
  1) Change into the work dir
     - `cd snap_benefits`
  2) Run with auto‑labels (downloads the 2023 PUMS data dictionary to label ancestries):
     - Top‑30 plot and CSV
       - `python script.py --household psam_husa.csv psam_husb.csv --person psam_pusa.csv psam_pusb.csv --min-people 50000 --topn 30 --anc-labels auto --outdir out`
     - All groups plot and CSV (no filtering)
       - `python script.py --household psam_husa.csv psam_husb.csv --person psam_pusa.csv psam_pusb.csv --min-people 0 --topn 0 --anc-labels auto --outdir out`

Outputs
- CSV: `out/snap_per_capita_ancestry.csv` (columns: ancestry_code, ancestry_label, people, people_in_snap_hh, rate)
- Charts:
  - Top‑N: `out/snap_per_capita_topN.png`
  - All groups: `out/snap_per_capita_all.png`
  - Optional focus: `out/snap_per_capita_selected.png` (use `--focus "Afghan,Somali,..."`)

Notes
- SERIALNO in 2023 PUMS is alphanumeric (e.g., `2023GQ...`), and the script handles it as a string key.
- Missing household FS is treated as not receiving SNAP in the join. If you prefer to drop missing FS households, we can add a flag.
- You can adjust memory profile with `--chunksize` (default 500k rows).

Versioning
- This project targets the latest public data: ACS PUMS 2023 1‑Year. The 2024 1‑Year PUMS files are embargoed (not yet public) as of this writing.
