# Crest Map

Scripts to pull CCRS crash/party data from data.ca.gov and build an interactive Folium map for Angeles Crest/Forest road corridors.

## Requirements
- Python 3.10+ recommended
- Packages: `pandas`, `folium`, `numpy`
- Network access to `https://data.ca.gov` for data pulls

Optional (for isolated deps):
```sh
python -m venv .venv
source .venv/bin/activate
```

## pull_data.py
Downloads CCRS crashes + parties, filters by road keywords, joins crash/party rows, and writes a CSV at the repo root.

Basic run (2016-current):
```sh
python pull_data.py
```

Use the datastore dump endpoint for faster bulk pulls:
```sh
python pull_data.py --use-dump
```

Outputs:
- `ccrs_parties_crashes_roads_<year-range>.csv` (or appends to the last output)
- `last_run.json` for incremental runs
- Optional dump totals log: `ccrs_dump_totals_<timestamp>.json`

Incremental behavior:
- If `last_run.json` exists, the script starts from the last run year and appends to the last output file.
- Delete `last_run.json` to force a full rebuild.

## generate_map.py
Reads the CCRS CSV and builds an interactive HTML map and a timestamped JSON data file.

Run:
```sh
python generate_map.py
```

Defaults:
- Input CSV: `ccrs_parties_crashes_roads_2016-2025.csv`
- Output HTML: `test.html`
- Output JSON: `angeles_forest_crashes_data_<timestamp>.json`

Notes:
- The script expects the CCRS columns emitted by `pull_data.py`.
- Open `test.html` in a browser to view the map.

## Docker
Build:
```sh
docker build -t crest-map .
```

Run with a bind mount so outputs persist in your working directory:
```sh
docker run --rm -v "$PWD:/app" crest-map
```
