import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd

CKAN_BASE = "https://data.ca.gov/api/3/action"
FOREST_BOUNDARY_CACHE = "angeles_national_forest_boundary.geojson"
FOREST_BOUNDARY_ENDPOINT = (
    "https://dpw.gis.lacounty.gov/dpw/rest/services/"
    "USFSpermitsWAB/MapServer/2/query"
)
FOREST_BOUNDARY_PARAMS = {
    "where": "1=1",
    "outFields": "NAME",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
}
CORRIDOR_ROAD_KEYWORDS = [
    "angeles crest",
    "angeles forest",
    "upper big tujunga",
    "big tujunga canyon",
    "mt wilson red box",
    "red box",
    "san gabriel canyon",
    "glendora mountain",
    "glendora ridge",
]
BOUNDARY_REQUIRED_ROAD_KEYWORDS = [
    "sr-39",
    "state route 39",
]

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
    Returns a dict: {resource_name: resource_metadata} for the CCRS dataset.
    """
    pkg = ckan_get("package_show", id="ccrs")
    return {r["name"]: r for r in pkg["resources"]}

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

def fetch_resource_dump(resource_id: str, expected_total: int | None = None, max_retries: int = 5) -> pd.DataFrame:
    url = f"https://data.ca.gov/datastore/dump/{resource_id}?bom=True"
    attempt = 0
    while True:
        attempt += 1
        fd, path = tempfile.mkstemp(prefix=f"ccrs_dump_{resource_id}_", suffix=".csv")
        os.close(fd)
        try:
            cmd = [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--retry-connrefused",
                "--connect-timeout",
                "30",
                "--max-time",
                "900",
                "-o",
                path,
                url,
            ]
            subprocess.run(cmd, check=True)
            df = pd.read_csv(path, low_memory=False)
            if expected_total is None or len(df) >= expected_total:
                return df
        except (subprocess.CalledProcessError, pd.errors.ParserError, OSError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed to download/read datastore dump for {resource_id}") from exc
            sleep_for = 2.0 * attempt
            print(f"Dump download/read failed for {resource_id} (attempt {attempt}/{max_retries}): {exc}. Retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
            continue
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if attempt >= max_retries:
            print(f"WARNING dump rows {len(df)} < expected {expected_total} after {attempt} attempt(s).")
            return df
        sleep_for = 1.0 * attempt
        print(f"Dump rows {len(df)} < expected {expected_total}; retrying in {sleep_for:.1f}s")
        time.sleep(sleep_for)


def fetch_resource_csv(resource: dict) -> pd.DataFrame:
    resource_url = resource["url"]
    resource_page_url = f"https://data.ca.gov/dataset/{resource['package_id']}/resource/{resource['id']}"
    cmd = [
        "curl",
        "-fsSL",
        "-A",
        "Mozilla/5.0",
        "-H",
        f"Referer: {resource_page_url}",
        resource_url,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("curl is required for uploaded CCRS resources but is not installed.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"curl failed for {resource['name']}: {stderr}") from exc

    return pd.read_csv(pd.io.common.BytesIO(result.stdout), low_memory=False)

def fetch_resource(resource: dict, use_dump: bool = False, limit: int = 5000, sleep_s: float = 0.0) -> tuple[pd.DataFrame, str]:
    resource_id = resource["id"]
    resource_url = resource.get("url")
    datastore_active = bool(resource.get("datastore_active"))

    if use_dump and datastore_active:
        total = ckan_get("datastore_search", resource_id=resource_id, limit=1)["total"]
        df = fetch_resource_dump(resource_id, expected_total=total)
        return df, "datastore_dump"

    if datastore_active:
        try:
            df = fetch_resource_all(resource_id, limit=limit, sleep_s=sleep_s)
            return df, "datastore_search"
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or not resource_url:
                raise
            print(f"Datastore unavailable for {resource['name']}; falling back to uploaded CSV.")

    if not resource_url:
        raise RuntimeError(f"Resource {resource['name']} has no downloadable URL.")

    df = fetch_resource_csv(resource)
    return df, "resource_csv"

def fetch_year_resources(resources: dict, crashes_name: str, parties_name: str, use_dump: bool) -> tuple[pd.DataFrame, str, pd.DataFrame, str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        crashes_future = executor.submit(
            fetch_resource,
            resources[crashes_name],
            use_dump=use_dump,
            limit=5000,
            sleep_s=0.05,
        )
        parties_future = executor.submit(
            fetch_resource,
            resources[parties_name],
            use_dump=use_dump,
            limit=5000,
            sleep_s=0.05,
        )
        crashes_df, crashes_source = crashes_future.result()
        parties_df, parties_source = parties_future.result()
    return crashes_df, crashes_source, parties_df, parties_source

def load_forest_polygons(cache_file: str = FOREST_BOUNDARY_CACHE) -> list[dict]:
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    else:
        url = f"{FOREST_BOUNDARY_ENDPOINT}?{urllib.parse.urlencode(FOREST_BOUNDARY_PARAMS)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        geojson = json.loads(raw)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(geojson, f, separators=(",", ":"))

    polygons = []
    for feature in geojson.get("features", []):
        geom = feature.get("geometry") or {}
        geom_type = geom.get("type")
        coords = geom.get("coordinates") or []
        if geom_type == "Polygon":
            polygons.append(normalize_polygon(coords))
        elif geom_type == "MultiPolygon":
            for polygon_coords in coords:
                polygons.append(normalize_polygon(polygon_coords))

    polygons = [p for p in polygons if p["rings"]]
    if not polygons:
        raise RuntimeError("No Angeles National Forest polygons loaded from boundary GeoJSON.")
    return polygons

def normalize_polygon(coords: list) -> dict:
    rings = []
    xs = []
    ys = []
    for ring in coords:
        normalized_ring = []
        for point in ring:
            if len(point) < 2:
                continue
            lon = float(point[0])
            lat = float(point[1])
            normalized_ring.append((lon, lat))
            xs.append(lon)
            ys.append(lat)
        if normalized_ring:
            rings.append(normalized_ring)
    return {
        "rings": rings,
        "bbox": (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None,
    }

def point_on_segment(lon: float, lat: float, a: tuple[float, float], b: tuple[float, float]) -> bool:
    ax, ay = a
    bx, by = b
    cross = (lat - ay) * (bx - ax) - (lon - ax) * (by - ay)
    if abs(cross) > 1e-12:
        return False
    return min(ax, bx) - 1e-12 <= lon <= max(ax, bx) + 1e-12 and min(ay, by) - 1e-12 <= lat <= max(ay, by) + 1e-12

def point_in_ring(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, point in enumerate(ring):
        if point_on_segment(lon, lat, point, ring[j]):
            return True
        xi, yi = point
        xj, yj = ring[j]
        intersects = (yi > lat) != (yj > lat)
        if intersects:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside

def point_in_polygon(lon: float, lat: float, polygon: dict) -> bool:
    bbox = polygon.get("bbox")
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            return False
    rings = polygon["rings"]
    if not rings or not point_in_ring(lon, lat, rings[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in rings[1:])

def point_in_forest(lon: float, lat: float, polygons: list[dict]) -> bool:
    return any(point_in_polygon(lon, lat, polygon) for polygon in polygons)

def build_corridor_road_pattern() -> re.Pattern:
    return re.compile("|".join(re.escape(k) for k in CORRIDOR_ROAD_KEYWORDS), re.IGNORECASE)

def build_boundary_required_road_pattern() -> re.Pattern:
    return re.compile("|".join(re.escape(k) for k in BOUNDARY_REQUIRED_ROAD_KEYWORDS), re.IGNORECASE)

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
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="Include every crash inside the Angeles National Forest boundary instead of only the Crest/Forest corridor.",
    )
    parser.add_argument(
        "--require-boundary",
        action="store_true",
        help="Drop corridor crashes whose coordinates are outside the Angeles National Forest boundary.",
    )
    args = parser.parse_args()

    current_year = dt.datetime.now().year
    start_year = 2016
    crash_q = None
    party_q = None
    strict_checks = True

    resources = get_ccrs_resources()
    corridor_road_pattern = build_corridor_road_pattern()
    boundary_required_road_pattern = build_boundary_required_road_pattern()
    forest_polygons = load_forest_polygons()
    forest_min_lon = min(p["bbox"][0] for p in forest_polygons if p["bbox"])
    forest_min_lat = min(p["bbox"][1] for p in forest_polygons if p["bbox"])
    forest_max_lon = max(p["bbox"][2] for p in forest_polygons if p["bbox"])
    forest_max_lat = max(p["bbox"][3] for p in forest_polygons if p["bbox"])

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

    end_year = current_year
    for year in range(start_year, end_year + 1):
        crashes_name = f"Crashes_{year}"
        parties_name = f"Parties_{year}"

        if crashes_name not in resources or parties_name not in resources:
            print(f"Skipping {year}: missing resource(s) {crashes_name} / {parties_name}")
            continue

        print(f"Downloading {crashes_name} and {parties_name} ...")
        crashes_df, crashes_source, parties_df, parties_source = fetch_year_resources(
            resources,
            crashes_name,
            parties_name,
            use_dump=args.use_dump,
        )
        totals_log.append(
            {"year": year, "resource": crashes_name, "total": int(len(crashes_df)), "source": crashes_source, "fetched_at": dt.datetime.now().isoformat()}
        )
        crashes_df["Year"] = year

        totals_log.append(
            {"year": year, "resource": parties_name, "total": int(len(parties_df)), "source": parties_source, "fetched_at": dt.datetime.now().isoformat()}
        )
        print(f"Year {year}: crashes_raw={len(crashes_df)} ({crashes_source}) parties_raw={len(parties_df)} ({parties_source})")
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
            missing_crash = [rename_map.get(c, c) for c in crash_cols_requested if rename_map.get(c, c) not in crashes_df.columns]
            missing_party = [c for c in party_cols_requested if c not in parties_df.columns]
            if missing_crash:
                print(f"WARNING Year {year}: missing crash columns: {missing_crash}")
            if missing_party:
                print(f"WARNING Year {year}: missing party columns: {missing_party}")

        crashes_t = crashes_df[crash_cols].copy()
        crashes_t = crashes_t.loc[:, ~crashes_t.columns.duplicated()].copy()
        crashes_t = crashes_t.drop_duplicates(subset=["CollisionId"])

        lat = pd.to_numeric(crashes_t["Latitude"], errors="coerce")
        lon = pd.to_numeric(crashes_t["Longitude"], errors="coerce")
        candidate_mask = (
            lat.between(forest_min_lat, forest_max_lat)
            & lon.between(forest_min_lon, forest_max_lon)
        )
        forest_mask = pd.Series(False, index=crashes_t.index)
        for idx in crashes_t[candidate_mask].index:
            forest_mask.at[idx] = point_in_forest(float(lon.at[idx]), float(lat.at[idx]), forest_polygons)

        filter_parts = []
        if args.boundary_only:
            crashes_t = crashes_t[forest_mask].copy()
            filter_parts.append("forest boundary")
        else:
            road_mask = pd.Series(False, index=crashes_t.index)
            boundary_required_road_mask = pd.Series(False, index=crashes_t.index)
            for road_col in ["PrimaryRoad", "SecondaryRoad"]:
                if road_col in crashes_t.columns:
                    road_mask = road_mask | crashes_t[road_col].astype(str).str.contains(corridor_road_pattern, na=False)
                    boundary_required_road_mask = boundary_required_road_mask | crashes_t[road_col].astype(str).str.contains(boundary_required_road_pattern, na=False)
            road_mask = road_mask | (boundary_required_road_mask & forest_mask)
            crashes_t = crashes_t[road_mask].copy()
            outside_boundary = int((~forest_mask.loc[crashes_t.index]).sum())
            if outside_boundary:
                print(f"WARNING Year {year}: {outside_boundary} corridor crashes have coordinates outside the forest boundary.")
            if args.require_boundary:
                crashes_t = crashes_t[forest_mask.loc[crashes_t.index]].copy()
                filter_parts.append("corridor roads + forest boundary")
            else:
                filter_parts.append("corridor primary/secondary roads")
                filter_parts.append("GPS boundary sanity check")
        filter_label = " + ".join(filter_parts)
        print(f"Year {year}: crashes_filtered={len(crashes_t)} within {filter_label}")
        if strict_checks and crashes_t.empty:
            print(f"WARNING Year {year}: crashes_filtered is empty after {filter_label} filter.")

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
