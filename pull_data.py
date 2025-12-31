import argparse
import csv
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd

CKAN_BASE = "https://data.ca.gov/api/3/action"

def ckan_get(action: str, max_retries: int = 5, backoff_s: float = 1.0, **params):
    """
    GET a CKAN action endpoint (e.g. package_show, datastore_search).
    """
    url = f"{CKAN_BASE}/{action}?{urllib.parse.urlencode(params)}"
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(url) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as e:
            if attempt >= max_retries:
                raise
            sleep_for = backoff_s * (2 ** (attempt - 1))
            print(f"CKAN request failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    if not payload.get("success"):
        raise RuntimeError(f"CKAN request failed: {payload}")
    return payload["result"]

def get_ccrs_resources():
    """
    Returns a dict: {resource_name: resource_id} for the CCRS dataset.
    """
    # The dataset slug is "ccrs" on data.ca.gov. :contentReference[oaicite:4]{index=4}
    pkg = ckan_get("package_show", id="ccrs")
    return {r["name"]: r["id"] for r in pkg["resources"]}

def fetch_resource_all(resource_id: str, limit: int = 5000, sleep_s: float = 0.0, q: str | None = None) -> pd.DataFrame:
    """
    Pages through datastore_search until all records are fetched.
    """
    offset = 0
    rows = []
    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        if q:
            params["q"] = q
        res = ckan_get("datastore_search", **params)
        batch = res["records"]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)

        # Optional politeness / throttling
        if sleep_s:
            time.sleep(sleep_s)
        # Stop if we reached the end
        if offset >= res["total"]:
            break
        
    return pd.DataFrame(rows)

def fetch_resource_dump(resource_id: str, expected_total: int | None = None, max_retries: int = 3) -> pd.DataFrame:
    url = f"https://data.ca.gov/datastore/dump/{resource_id}?bom=True"
    attempt = 0
    while True:
        attempt += 1
        df = pd.read_csv(url, low_memory=False)
        if expected_total is None or len(df) >= expected_total:
            return df
        if attempt >= max_retries:
            print(f"WARNING dump rows {len(df)} < expected {expected_total} after {attempt} attempt(s).")
            return df
        sleep_for = 1.0 * attempt
        print(f"Dump rows {len(df)} < expected {expected_total}; retrying in {sleep_for:.1f}s")
        time.sleep(sleep_for)

LAST_RUN_FILE = "last_run.json"

def load_last_run():
    if not os.path.exists(LAST_RUN_FILE):
        return None, None
    try:
        with open(LAST_RUN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("last_run_iso")
        out = data.get("output_file")
        return (dt.datetime.fromisoformat(ts) if ts else None), out
    except Exception:
        return None, None

def save_last_run(ts, output_file):
    with open(LAST_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_run_iso": ts.isoformat(), "output_file": output_file}, f)

def ensure_output_columns(output_file, desired_cols):
    if not output_file or not os.path.exists(output_file):
        return None
    with open(output_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        return None
    missing = [c for c in desired_cols if c not in header]
    if not missing:
        return header
    df_existing = pd.read_csv(output_file, dtype=str)
    for c in missing:
        df_existing[c] = ""
    new_header = header + missing
    df_existing = df_existing.reindex(columns=new_header)
    df_existing.to_csv(output_file, index=False)
    return new_header

def main():
    parser = argparse.ArgumentParser(description="Pull CCRS crash + party data.")
    parser.add_argument(
        "--use-dump",
        action="store_true",
        help="Download datastore CSV dumps instead of datastore_search API.",
    )
    args = parser.parse_args()

    current_year = dt.datetime.now().year
    start_year = 2016
    crash_q = "Angeles Crest"
    party_q = crash_q
    road_keywords = [
        "angeles forest",
        "angeles crest",
        "upper big tujunga",
        "red box road",
        "glendora mountain",
        "san gabriel canyon",
        "big tujunga canyon",
        "glendora ridge",
        "azusa canyon",
    ]
    road_pattern = "|".join(re.escape(k) for k in road_keywords)
    strict_checks = True

    resources = get_ccrs_resources()

    last_run_dt, last_output_file = load_last_run()
    incremental_year = None
    if last_run_dt:
        incremental_year = last_run_dt.year
        start_year = incremental_year
        print(f"Incremental run for year {incremental_year} (last_run={last_run_dt.isoformat()})")
    else:
        #start_year = current_year
        print(f"No last_run.json found; staring from {start_year}")

    out_file_tmp = "ccrs_parties_crashes_roads_tmp.csv"
    output_file = last_output_file
    wrote_header = bool(output_file and os.path.exists(output_file))
    header_checked = False
    current_header = None
    dedupe_key = ["CollisionId", "PartyId"]
    rows_written = 0
    year_min = None
    year_max = None
    totals_log = []

    end_year = start_year if incremental_year is not None else current_year
    for year in range(start_year, end_year + 1):
        crashes_name = f"Crashes_{year}"
        parties_name = f"Parties_{year}"

        if crashes_name not in resources or parties_name not in resources:
            print(f"Skipping {year}: missing resource(s) {crashes_name} / {parties_name}")
            continue

        print(f"Downloading {crashes_name} ...")
        if args.use_dump:
            crashes_total = ckan_get("datastore_search", resource_id=resources[crashes_name], limit=1)["total"]
            totals_log.append(
                {"year": year, "resource": crashes_name, "total": int(crashes_total), "fetched_at": dt.datetime.now().isoformat()}
            )
            crashes_df = fetch_resource_dump(resources[crashes_name], expected_total=crashes_total)
        else:
            crashes_df = fetch_resource_all(resources[crashes_name], limit=5000, sleep_s=0.05)
        crashes_df["Year"] = year

        print(f"Downloading {parties_name} ...")
        if args.use_dump:
            parties_total = ckan_get("datastore_search", resource_id=resources[parties_name], limit=1)["total"]
            totals_log.append(
                {"year": year, "resource": parties_name, "total": int(parties_total), "fetched_at": dt.datetime.now().isoformat()}
            )
            parties_df = fetch_resource_dump(resources[parties_name], expected_total=parties_total)
        else:
            parties_df = fetch_resource_all(resources[parties_name], limit=5000, sleep_s=0.05)
            if parties_df.empty and party_q:
                print(f"No parties rows for '{party_q}', retrying without q ...")
                parties_df = fetch_resource_all(resources[parties_name], limit=5000, sleep_s=0.05)
        print(f"Year {year}: crashes_raw={len(crashes_df)} parties_raw={len(parties_df)}")
        # ---- TRUNCATED JOIN: choose only the columns you actually want ----
        crash_cols = [
            "Collision Id",
            "Crash Date Time",
            "NumberInjured",
            "NumberKilled",
            "Latitude",
            "Longitude",
            "PrimaryRoad",
            "SecondaryRoad",
            "Primary Road",
            "Secondary Road",
            "Collision Type Description",
            "Primary Collision Factor Violation",
            "MotorVehicleInvolvedWithDesc",
            "MotorVehicleInvolvedWithOtherDesc",
            "Day Of Week",
            "Year",
        ]
        crash_cols_requested = list(crash_cols)

        party_cols = [
            "CollisionId",
            "PartyId",
            "PartyType",
            "IsAtFault",
            "MovementPrecCollDescription",
            "Vehicle1Year",
            "Vehicle1TypeDesc",
            "Vehicle1Make",
            "Vehicle1Model",
            "GenderCode",
        ]
        party_cols_requested = list(party_cols)

        crashes_df.columns = [c.strip() if isinstance(c, str) else c for c in crashes_df.columns]
        parties_df.columns = [c.strip() if isinstance(c, str) else c for c in parties_df.columns]

        rename_map = {
            "Collision Id": "CollisionId",
            "Primary Road": "PrimaryRoad",
            "Secondary Road": "SecondaryRoad",
        }
        crashes_df = crashes_df.rename(columns=rename_map)
        parties_df = parties_df.rename(columns={"Collision Id": "CollisionId"})
        crashes_df = crashes_df.loc[:, ~pd.Index(crashes_df.columns).duplicated()].copy()
        parties_df = parties_df.loc[:, ~pd.Index(parties_df.columns).duplicated()].copy()

        crash_cols = [rename_map.get(c, c) for c in crash_cols]
        party_cols = [{"Collision Id": "CollisionId"}.get(c, c) for c in party_cols]
        crash_cols = list(dict.fromkeys(crash_cols))
        party_cols = list(dict.fromkeys(party_cols))

        crash_cols = [c for c in crash_cols if c in crashes_df.columns]
        party_cols = [c for c in party_cols if c in parties_df.columns]
        if strict_checks:
            missing_crash = [c for c in crash_cols_requested if c not in crashes_df.columns]
            missing_party = [c for c in party_cols_requested if c not in parties_df.columns]
            if missing_crash:
                print(f"WARNING Year {year}: missing crash columns: {missing_crash}")
            if missing_party:
                print(f"WARNING Year {year}: missing party columns: {missing_party}")

        crashes_t = crashes_df[crash_cols].copy()
        crashes_t = crashes_t.loc[:, ~crashes_t.columns.duplicated()].copy()
        crashes_t = crashes_t.drop_duplicates(subset=["CollisionId"])

        road_cols = [c for c in ["PrimaryRoad", "SecondaryRoad"] if c in crashes_t.columns]
        if not road_cols:
            raise RuntimeError("No road fields found in crashes data.")

        road_mask = pd.Series(False, index=crashes_t.index)
        for c in road_cols:
            series = crashes_t[c]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            road_mask = road_mask | series.astype(str).str.contains(road_pattern, case=False, na=False)
        crashes_t = crashes_t[road_mask].copy()
        print(f"Year {year}: crashes_filtered={len(crashes_t)}")
        if strict_checks and crashes_t.empty:
            print(f"WARNING Year {year}: crashes_filtered is empty after road filter.")

        parties_t = parties_df[party_cols].copy()
        if "IsAtFault" in parties_t.columns:
            parties_t.loc[:, "IsAtFault"] = parties_t["IsAtFault"].fillna("").replace("", "False")

        if "CollisionId" not in parties_t.columns:
            raise RuntimeError(f"No collision ID column found in parties data. Columns: {list(parties_t.columns)}")
        if "CollisionId" not in crashes_t.columns:
            raise RuntimeError(f"No collision ID column found in crashes data. Columns: {list(crashes_t.columns)}")

        # Align CollisionId types before filtering (some years differ)
        crashes_t["CollisionId"] = pd.to_numeric(crashes_t["CollisionId"], errors="coerce")
        parties_t["CollisionId"] = pd.to_numeric(parties_t["CollisionId"], errors="coerce")
        crashes_t = crashes_t.dropna(subset=["CollisionId"]).copy()
        parties_t = parties_t.dropna(subset=["CollisionId"]).copy()
        crashes_t["CollisionId"] = crashes_t["CollisionId"].astype(int)
        parties_t["CollisionId"] = parties_t["CollisionId"].astype(int)

        print(f"Year {year}: parties_filtered={len(parties_t)}")
        if strict_checks:
            missing_parties = crashes_t[~crashes_t["CollisionId"].isin(parties_t["CollisionId"])]
            if not missing_parties.empty:
                sample = missing_parties[["CollisionId", "Crash Date Time", "PrimaryRoad", "SecondaryRoad"]].head(5)
                print(f"WARNING Year {year}: {len(missing_parties)} crashes have no parties rows in parties data.")
                print(sample.to_string(index=False))

        joined = crashes_t.merge(
            parties_t,
            left_on="CollisionId",
            right_on="CollisionId",
            how="left",
            suffixes=("_crash", "_party"),
        )
        joined = joined.drop_duplicates(subset=["CollisionId", "PartyId"])
        print(f"Year {year}: joined={len(joined)}")

        if not joined.empty:
            year_min = year if year_min is None else min(year_min, year)
            year_max = year if year_max is None else max(year_max, year)

        if output_file and os.path.exists(output_file) and not header_checked:
            current_header = ensure_output_columns(output_file, list(joined.columns))
            header_checked = True
            wrote_header = True
        if current_header:
            joined = joined.reindex(columns=current_header)

        if output_file and incremental_year is not None:
            joined.to_csv(output_file, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        elif output_file:
            joined.to_csv(output_file, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        else:
            joined.to_csv(out_file_tmp, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        rows_written += len(joined)

        del crashes_df, parties_df, crashes_t, parties_t, joined

    if output_file and incremental_year is not None and os.path.exists(output_file):
        print("Deduping output file...")
        df_out = pd.read_csv(output_file)
        before = len(df_out)
        df_out = df_out.drop_duplicates(subset=dedupe_key)
        after = len(df_out)
        if after != before:
            print(f"Deduped rows: {before} -> {after}")
        df_out.to_csv(output_file, index=False)

    if totals_log:
        totals_path = f"ccrs_dump_totals_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        with open(totals_path, "w", encoding="utf-8") as f:
            json.dump(totals_log, f, indent=2)
        print(f"Wrote totals log: {totals_path}")

    if wrote_header and year_min is not None and year_max is not None:
        year_range = f"{year_min}-{year_max}"
        final_name = output_file or f"ccrs_parties_crashes_roads_{year_range}.csv"
        if not output_file:
            os.replace(out_file_tmp, final_name)
        print("Wrote rows:", rows_written)
        save_last_run(dt.datetime.now(), final_name)

if __name__ == "__main__":
    main()
