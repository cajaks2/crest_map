import csv
import pandas as pd
import folium
import numpy as np
import time, json

ts = int(time.time())

expected_cols = [
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
    "Crash Date Time",
    "NumberInjured",
    "NumberKilled",
    "Latitude",
    "Longitude",
    "PrimaryRoad",
    "SecondaryRoad",
    "Collision Type Description",
    "Primary Collision Factor Violation",
    "MotorVehicleInvolvedWithDesc",
    "MotorVehicleInvolvedWithOtherDesc",
    "Day Of Week",
    "Year",
]

def load_input(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None) or []
    if any(c in expected_cols for c in first):
        df_local = pd.read_csv(path)
    else:
        df_local = pd.read_csv(path, header=None)
        if df_local.shape[1] > len(expected_cols):
            first_row = df_local.iloc[0].tolist()
            lead = 0
            for v in first_row:
                if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
                    lead += 1
                else:
                    break
            if df_local.shape[1] - lead >= len(expected_cols):
                df_local = df_local.iloc[:, lead : lead + len(expected_cols)]
        if df_local.shape[1] != len(expected_cols):
            raise ValueError(f"Unexpected column count {df_local.shape[1]} in {path}")
        df_local.columns = expected_cols
    df_local.columns = [c.strip() if isinstance(c, str) else c for c in df_local.columns]
    return df_local

raw = load_input("ccrs_parties_crashes_roads_2016-2026.csv")
df = raw.copy()
field_names = df.columns.tolist()

# Deduplicate + clean coords, keep party details per collision
if "CollisionId" in df.columns:
    collision_col = "CollisionId"
elif "Collision Id" in df.columns:
    collision_col = "Collision Id"
else:
    candidates = [c for c in df.columns if str(c).replace(" ", "").lower() == "collisionid"]
    if candidates:
        collision_col = candidates[0]
    else:
        raise KeyError("CollisionId column not found in input CSV.")
party_cols = [
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
party_cols = [c for c in party_cols if c in df.columns]
crash_cols = [c for c in df.columns if c not in party_cols]

if party_cols:
    def _party_rows(sub):
        rows = []
        for _, r in sub.iterrows():
            rows.append({c: ("" if pd.isna(r.get(c)) else str(r.get(c))) for c in party_cols})
        return rows

    agg = {c: "first" for c in crash_cols}
    agg["__parties"] = _party_rows
    grouped = df.groupby(collision_col, dropna=False)
    crash_cols_no_id = [c for c in crash_cols if c != collision_col]
    base = grouped[crash_cols_no_id].first().reset_index()
    try:
        parties_series = grouped.apply(lambda g: _party_rows(g[party_cols]), include_groups=False)
    except TypeError:
        parties_series = grouped.apply(lambda g: _party_rows(g[party_cols]))
    parties = parties_series.reset_index(name="__parties")
    df = base.merge(parties, on=collision_col, how="left")
else:
    df = df.drop_duplicates(subset=[collision_col]).copy()

field_names = [c for c in df.columns if c != "__parties"]
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Latitude","Longitude"]).copy()

# Trim garbage coords
lat_lo, lat_hi = df["Latitude"].quantile([0.005, 0.995])
lon_lo, lon_hi = df["Longitude"].quantile([0.005, 0.995])
df = df[df["Latitude"].between(lat_lo, lat_hi) & df["Longitude"].between(lon_lo, lon_hi)].copy()

# Years (prefer explicit Year column)
if "Year" in df.columns:
    df["__year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(-1).astype(int)
else:
    dt = pd.to_datetime(df["Crash Date Time"], errors="coerce")
    df["__year"] = dt.dt.year.fillna(-1).astype(int)
df = df[df["__year"].between(1900, 2100)].copy()
years = sorted(df["__year"].unique().tolist())
has_unknown_year = False

if years:
    min_year, max_year = min(years), max(years)
    year_range = str(min_year) if min_year == max_year else f"{min_year}-{max_year}"
else:
    year_range = "Unknown"
title_text = f"Angeles Forest Crashes {year_range}"

# Viz fields
df["__ctype"] = df["Collision Type Description"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
df["__killed"] = pd.to_numeric(df["NumberKilled"], errors="coerce").fillna(0).astype(int)
df["__fatal"] = df["__killed"] > 0
df["__veh_types"] = df["__parties"].apply(
    lambda ps: [p.get("Vehicle1TypeDesc","") for p in ps if p.get("Vehicle1TypeDesc","")] if isinstance(ps, list) else []
)

# Colors
counts_all = df["__ctype"].value_counts()
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
           "#aec7e8","#ffbb78"]
TOP_N = 12
top_types = list(counts_all.head(TOP_N).index)
color_map = {t: palette[i % len(palette)] for i, t in enumerate(top_types)}
other_color = "#444444"
unknown_color = "#999999"
veh_counts = pd.Series([t for sub in df["__veh_types"] for t in sub]).value_counts()
TOP_VEH_N = 7
top_veh_types = list(veh_counts.head(TOP_VEH_N).index)
if "Bicycle" in veh_counts.index and "Bicycle" not in top_veh_types:
    top_veh_types.append("Bicycle")
top_veh_types.append("Other")

def color_for(t):
    t = "Unknown" if pd.isna(t) else str(t).strip()
    if t == "" or t.lower() == "nan":
        t = "Unknown"
    if t == "Unknown":
        return unknown_color
    return color_map.get(t, other_color)

def esc(x):
    if pd.isna(x): return ""
    s = str(x)
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace('"',"&quot;").replace("'","&#39;"))

def compute_counts(sub):
    return {"types": sub["__ctype"].value_counts().to_dict(),
            "fatal": int(sub["__fatal"].sum()),
            "total": int(len(sub))}

per_year_counts = {str(y): compute_counts(df[df["__year"] == y]) for y in years}

# Map bounds
min_lat, max_lat = float(df["Latitude"].min()), float(df["Latitude"].max())
min_lon, max_lon = float(df["Longitude"].min()), float(df["Longitude"].max())
center = [(min_lat+max_lat)/2, (min_lon+max_lon)/2]

m = folium.Map(location=center, zoom_start=10, tiles=None, prefer_canvas=True, control_scale=True)
folium.TileLayer("OpenStreetMap", control=False).add_to(m)
m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

# CSS + controls position + label styling (label not inside control, so no clipping)
embed_description = (
    "Interactive map of Angeles Forest crash reports with year, incident, and vehicle filters."
)
m.get_root().header.add_child(folium.Element(f"""
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="build-timestamp" content="{ts}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title_text}">
<meta property="og:description" content="{embed_description}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{title_text}">
<meta name="twitter:description" content="{embed_description}">
<style>
.leaflet-marker-icon.fatal-x, .leaflet-marker-shadow.fatal-x {{ pointer-events: none !important; }}
.leaflet-marker-icon.fatal-x * {{ pointer-events: none !important; user-select:none !important; }}
.leaflet-popup {{ z-index: 900; }}

/* move top-right controls down a bit (LayerControl + zoom) */
.leaflet-top.leaflet-right {{ top: 120px; }}
/* move top-left zoom controls down to avoid title overlap */
.leaflet-top.leaflet-left {{ top: 72px; }}

/* Select buttons inside layers control */
.layer-tools {{
  display:flex; gap:6px; padding:6px 6px 0 6px;
}}
.layer-tools button {{
  flex:1;
  border:1px solid #cfcfcf;
  background:#fff;
  border-radius:8px;
  padding:6px 8px;
  font-family:system-ui;
  font-size:12px;
  font-weight:700;
  cursor:pointer;
}}
.layer-tools button:active {{ background:#f2f2f2; }}

/* Layer control header */
.layer-title {{
  padding: 6px 6px 0 6px;
  font-family: system-ui;
  font-size: 12px;
  font-weight: 800;
}}

/* Map title */
#map-title {{
  position: fixed;
  top: 8px;
  left: 8px;
  z-index: 9999;
  background: rgba(255,255,255,.94);
  padding: 8px 12px;
  border-radius: 8px;
  font-family: system-ui;
  font-size: 14px;
  font-weight: 700;
  box-shadow: 0 2px 6px rgba(0,0,0,.25);
  max-width: calc(100vw - 16px);
}}

/* Fixed label above collapsed toggle */
#year-filter-label {{
  position: fixed;
  z-index: 9999;
  background: rgba(255,255,255,.96);
  border: 1px solid #cfcfcf;
  border-radius: 10px;
  padding: 4px 8px;
  font-family: system-ui;
  font-size: 12px;
  font-weight: 800;
  box-shadow: 0 2px 6px rgba(0,0,0,.18);
  white-space: nowrap;
  pointer-events: none;
}}


/* Info button stays on left below title */
#info-button {{
  position: fixed;
  left: 56px;
  right: auto;
  top: 56px;
  z-index: 500;
  background: rgba(255,255,255,.94);
  border: 1px solid #cfcfcf;
  border-radius: 999px;
  padding: 6px 10px;
  font-family: system-ui;
  font-size: 12px;
  font-weight: 700;
  box-shadow: 0 2px 6px rgba(0,0,0,.18);
  cursor: pointer;
}}

/* Popup */
.leaflet-popup-content-wrapper {{ padding: 0; }}
.leaflet-popup-content {{ margin: 0; }}
.popup-wrap {{ font-family: system-ui; font-size: 12px; width: min(64vw, 260px); }}
.popup-title {{ font-weight: 700; padding: 6px 8px; background:#f6f6f6; border-bottom: 1px solid #ddd; }}
.popup-body {{ padding: 6px 8px; }}
.kv {{ display: grid; grid-template-columns: 96px 1fr; gap: 6px; margin-bottom: 4px; }}
.k {{ font-weight: 700; color: #444; }}
.v {{ word-break: break-word; }}
.popup-wrap .btn {{
  display:inline-block;
  width:100%;
  text-align:center;
  padding:8px 10px;
  margin-top:8px;
  border-radius:8px;
  border:2px solid #7ea6e3;
  background:#F0FCFF;
  font-weight:700;
  box-shadow: 0 2px 4px rgba(0,0,0,.18);
  transition: transform 0.08s ease, box-shadow 0.08s ease;
}}
.popup-wrap .btn:hover {{ box-shadow: 0 3px 6px rgba(0,0,0,.22); }}
.popup-wrap .btn:active {{ transform: translateY(1px); box-shadow: 0 2px 3px rgba(0,0,0,.16); }}
.btn:active {{ background:#f2f2f2; }}
.full-wrap {{
  display:none;
  margin-top:8px;
  border-top:1px solid #e6e6e6;
  padding-top:8px;
}}
.full-table-wrap {{
  max-height: 26vh;
  overflow-y: auto;
  border:1px solid #ddd;
  border-radius: 6px;
}}
.full-table-wrap table {{ width:100%; border-collapse: collapse; }}
.full-table-wrap th {{ text-align:left; padding:2px 6px; white-space:normal; word-break:break-word; vertical-align:top; width: 38%; }}
.full-table-wrap td {{ padding:2px 6px; word-break:break-word; vertical-align:top; }}
@media (max-width: 420px) {{
  .popup-wrap {{ width: 64vw; }}
  .kv {{ grid-template-columns: 96px 1fr; }}
  .full-table-wrap {{ max-height: 24vh; }}
  #legend-filters {{ flex-direction: column; }}
  #incident-filter, #vehicle-filter {{ width: 100%; }}
}}
@media (max-width: 480px) {{
  .leaflet-top.leaflet-right {{ top: 150px; }}
  .leaflet-top.leaflet-left {{ top: 92px; }}
  #map-title {{
    font-size: 12px;
    padding: 6px 10px;
  }}
  #year-filter-label {{
    font-size: 11px;
    padding: 3px 6px;
    max-width: 70vw;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  #info-button {{
    top: 52px;
    font-size: 11px;
    padding: 5px 8px;
  }}
}}
</style>
"""))

title_html = f'<div id="map-title">{title_text}</div>'

# UI: title/info + legend + year label element
m.get_root().html.add_child(folium.Element(title_html + """

<div id="year-filter-label">Year filter</div>

<button id="info-button" onclick="
  var p=document.getElementById('about-panel');
  if(!p) return;
  p.style.display = (p.style.display==='none' || p.style.display==='') ? 'block' : 'none';
"
>
ⓘ Info
</button>

<div id="about-panel" onclick="if(event.target===this){this.style.display='none';}"
style="display:block;position:fixed;top:92px;left:56px;right:auto;z-index:500;
background:rgba(255,255,255,.94);padding:10px 12px 12px 12px;border-radius:8px;
font-family:system-ui;font-size:12px;max-width:360px;box-shadow:0 2px 6px rgba(0,0,0,.25);">
<button onclick="document.getElementById('about-panel').style.display='none';"
style="position:absolute;top:6px;right:6px;border:none;background:none;font-size:16px;font-weight:700;cursor:pointer;">×</button>
<b>About this map</b><br><br>
Use the year selector (top-right) to toggle years. Selector in bottum left, legend counts update automatically. You can select type of vehicle as well as type of incident. You can also add a text filter for any field, multiple queries can be done if seperated by a semicolon. EG: crest;motorcylce. Fatal is a keyword to search for fatalities.
<br><br>
<b>Fatal crashes</b> (NumberKilled &gt; 0) are shown with an <b>× overlay</b>.
</div>

  <div id="legend" class="collapsed" style="
 position: fixed;
 bottom: 12px;
 left: 12px;
 z-index: 9999;
 background: rgba(255,255,255,0.94);
 padding: 10px 10px 8px 10px;
 border-radius: 8px;
 box-shadow: 0 2px 6px rgba(0,0,0,0.25);
 font-family: system-ui;
 width: min(45vw, 360px);
 max-height: 40vh;
 overflow: auto;
">
<div id="legend-head" style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
  <div style="font-weight:700; font-size:13px;">Search/Guide</div>
  <button id="legend-collapse" style="border:1px solid #cfcfcf;background:#fff;border-radius:8px;padding:2px 6px;font-size:11px;font-weight:700;cursor:pointer;">Show</button>
</div>
  <div id="legend-sub" style="font-size:11px; color:#555; margin-bottom:8px;"></div>
  <div id="legend-search" style="display:flex;gap:6px;align-items:center;margin:6px 0 8px 0;">
    <input id="legend-search-input" type="text" placeholder="Example: thursday or motorcyle or fatal;crest." style="flex:1;border:1px solid #cfcfcf;border-radius:8px;padding:4px 8px;font-size:11px;" />
    <button id="legend-search-clear" class="mini-btn" type="button" style="min-width:auto;padding:2px 8px;">Clear</button>
  </div>
  <div id="legend-filters" style="display:flex;gap:10px;">
    <div id="vehicle-filter" style="flex:1;margin:6px 0;"></div>
    <div id="incident-filter" style="flex:1;margin:6px 0;">
      <div id="legend-body"></div>
    </div>
  </div>
  <style>
    #legend .li { display:flex; align-items:flex-start; gap:8px; margin:6px 0; }
    #legend .sw { width:12px; height:12px; border-radius:3px; margin-top:2px; flex:0 0 12px; }
    #legend .tx { font-size:12px; line-height:1.2; min-width:0; overflow-wrap:anywhere; display:flex; gap:6px; }
    #legend .tx .lbl { flex:1; min-width:0; overflow-wrap:anywhere; }
    #legend .ct { color:#666; font-size:11px; }
    #legend .section-head { display:flex; align-items:center; justify-content:space-between; gap:6px; margin:4px 0 6px 0; }
    #legend .section-actions { display:flex; gap:4px; }
    #legend .mini-btn { border:1px solid #cfcfcf; background:#fff; border-radius:8px; padding:2px 6px; font-size:10px; font-weight:700; cursor:pointer; min-width:48px; text-align:center; }
    #legend .legend-divider { height:1px; background:#e0e0e0; margin:8px 0; }
    #legend.collapsed #legend-sub,
    #legend.collapsed #legend-tools,
    #legend.collapsed #legend-search,
    #legend.collapsed #legend-filters,
    #legend.collapsed #legend-body { display:none !important; }
  </style>
</div>

"""))

# Popup fields
summary_fields = [
    ("CollisionId","Collision Id"),
    ("Report Number","Report Number"),
    ("Crash Date Time","Crash Date Time"),
    ("Day Of Week","Day Of Week"),
    ("PrimaryRoad","Primary Road"),
    ("SecondaryRoad","Secondary Road"),
    ("City Name","City"),
    ("Collision Type Description","Collision Type"),
    ("Primary Collision Factor Violation","Violation"),
    ("NumberInjured","Injured"),
    ("NumberKilled","Killed"),
    ("HitRun","Hit & Run"),
    ("IsTowAway","Tow Away"),
    ("Weather 1","Weather 1"),
    ("Road Condition 1","Road Condition 1"),
    ("LightingDescription","Lighting"),
    ("MotorVehicleInvolvedWithDesc","Involved With"),
    ("MotorVehicleInvolvedWithOtherDesc","Involved With (Other)"),
    ("MilepostMarker","Milepost Marker"),
]

def build_popup_data(row):
    fields = {c: ("" if pd.isna(row.get(c)) else str(row.get(c))) for c in field_names}
    parties = row.get("__parties") if "__parties" in row else None
    return {
        "fields": fields,
        "summary": {label: fields.get(colname, "") for colname, label in summary_fields},
        "collision_id": fields.get("CollisionId", "") or fields.get("Collision Id", ""),
        "parties": parties or []
    }

def build_search_text(row):
    fields = {c: ("" if pd.isna(row.get(c)) else str(row.get(c))) for c in field_names}
    parts = []
    for c in [
        "CollisionId",
        "PrimaryRoad",
        "SecondaryRoad",
        "City Name",
        "Collision Type Description",
        "MotorVehicleInvolvedWithDesc",
        "MotorVehicleInvolvedWithOtherDesc",
        "Day Of Week",
        "Year",
        "Primary Collision Factor Violation",
    ]:
        v = fields.get(c, "")
        if v:
            parts.append(v)
    parties = row.get("__parties") if "__parties" in row else None
    if parties:
        for p in parties:
            for k in [
                "Vehicle1TypeDesc",
                "Vehicle1Make",
                "Vehicle1Model",
                "MovementPrecCollDescription",
                "PartyType",
                "GenderCode",
            ]:
                v = p.get(k, "")
                if v:
                    parts.append(str(v))
    return " ".join(parts).strip()

# Add year layers (all shown by default)
year_group_vars = {}
for y in years:
    fg = folium.FeatureGroup(name=str(y), show=True)
    year_group_vars[str(y)] = fg.get_name()
    fg.add_to(m)

# no Unknown group when using explicit Year column

# Build marker data for JSON
marker_data = []
for _, r in df.iterrows():
    year_label = str(r["__year"])
    popup_data = build_popup_data(r)
    marker_data.append({
        "lat": float(r["Latitude"]),
        "lon": float(r["Longitude"]),
        "ctype": r["__ctype"],
        "fatal": bool(r["__fatal"]),
        "veh": r["__veh_types"],
        "year": year_label,
        "data": popup_data,
        "search": build_search_text(r).lower()
    })

# Collapsible LayerControl
folium.LayerControl(collapsed=True).add_to(m)

# Auto-size dots by zoom + data load
data_json = json.dumps(marker_data, ensure_ascii=True, separators=(",", ":"))
data_file = f"angeles_forest_crashes_data_{ts}.json"
with open(data_file, "w", encoding="utf-8") as f:
    f.write(data_json)

groups_json = json.dumps(year_group_vars)
m.get_root().html.add_child(folium.Element(f"""
<script>
(function() {{
  function isCoarsePointer() {{ return window.matchMedia && window.matchMedia("(pointer: coarse)").matches; }}
  function r(z) {{
    var coarse = isCoarsePointer();
    if (z>=15) return coarse ? 12 : 10;
    if (z>=14) return coarse ? 10 : 8;
    if (z>=13) return coarse ? 9 : 7;
    if (z>=12) return coarse ? 8 : 6;
    return coarse ? 7 : 5;
  }}
  function hitR(z) {{
    var coarse = isCoarsePointer();
    if (z>=15) return coarse ? 24 : 16;
    if (z>=13) return coarse ? 22 : 15;
    return coarse ? 20 : 14;
  }}
  var mapName="{m.get_name()}";
  var dataUrl="{data_file}";
  var groupVars={groups_json};
  function getMap(){{ return window[mapName] || window._leaflet_map || null; }}
  function getGroup(year){{ var k=groupVars[year]; return k ? window[k] : null; }}

  var allMarkers = [];
  var markerById = {{}};
  window._allMarkers = allMarkers;
  window._markerById = markerById;

  function addMarker(item){{
    var map = getMap();
    if (map && !window._leaflet_map) window._leaflet_map = map;
    var g = getGroup(item.year);
    if (!g) return;
    var ctype = item.ctype || "Unknown";
    var col = (ctype === "Unknown") ? "{unknown_color}" : ({json.dumps(color_map)}[ctype] || "{other_color}");

    var m = L.circleMarker([item.lat, item.lon], {{
      radius: 6,
      color: col,
      fill: true,
      fillColor: col,
      fillOpacity: 0.80,
      weight: 2
    }});
    var h = L.circleMarker([item.lat, item.lon], {{
      radius: hitR(getMap() && getMap().getZoom ? getMap().getZoom() : 10),
      color: "#000",
      opacity: 0,
      fill: true,
      fillColor: "#000",
      fillOpacity: 0,
      weight: 0
    }});
    var popupHtml = buildPopup(item.data);
    if (popupHtml) m.bindPopup(popupHtml, {{maxWidth: 360}});
    if (popupHtml) h.bindPopup(popupHtml, {{maxWidth: 360}});
    var tooltip = "";
    var collisionId = "";
    if (item.data && item.data.collision_id) collisionId = String(item.data.collision_id);
    if (collisionId) tooltip = collisionId + " | " + ctype;
    if (tooltip) m.bindTooltip(tooltip);
    if (tooltip) h.bindTooltip(tooltip);
    m.options._incidentType = ctype;
    m.options._collisionId = collisionId;
    h.options._incidentType = ctype;
    h.options._collisionId = collisionId;
    if (collisionId) markerById[collisionId] = m;
    h.addTo(g);
    m.addTo(g);

    var f = null;
    if (item.fatal) {{
      f = L.marker([item.lat, item.lon], {{
        icon: L.divIcon({{
          className: "fatal-x",
          iconSize: [18,18],
          iconAnchor: [9,9],
          html: "<div style='width:18px;height:18px;display:flex;align-items:center;justify-content:center;color:#000;font-weight:900;font-size:18px;line-height:18px;user-select:none;'>×</div>"
        }})
      }});
      if (popupHtml) f.bindPopup(popupHtml, {{maxWidth: 360}});
      if (tooltip) f.bindTooltip(tooltip);
      f.options._incidentType = ctype;
      f.options._collisionId = collisionId;
      f.addTo(g);
    }}
    allMarkers.push({{m:m, h:h, f:f, t: ctype, g:g, v: item.veh || [], s: item.search || "", r: m.getRadius ? m.getRadius() : null, hidden:false}});
  }}

  function escHtml(s) {{
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
                    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
  }}

  function safeId(s) {{
    return String(s || "").replace(/[^a-zA-Z0-9_-]/g, "");
  }}

  function buildPopup(data) {{
    if (!data || !data.fields) return "";
    var summary = data.summary || {{}};
    var summaryRows = [];
    Object.keys(summary).forEach(function(k){{
      var v = summary[k];
      if (v === null || v === undefined || String(v).trim() === "") return;
      summaryRows.push("<div class='kv'><div class='k'>"+escHtml(k)+"</div><div class='v'>"+escHtml(v)+"</div></div>");
    }});
    var parties = data.parties || [];
    var partyRows = [];
    if (parties.length) {{
      parties.forEach(function(p, idx){{
        var cells = [];
        Object.keys(p).forEach(function(k){{
          var v = p[k];
          if (v === null || v === undefined) v = "";
          cells.push("<tr><th>"+escHtml(k)+"</th><td>"+escHtml(v)+"</td></tr>");
        }});
        var topStyle = idx === 0 ? "" : "margin-top:6px;border-top:1px solid #e6e6e6;padding-top:6px;";
        partyRows.push("<div style='"+topStyle+"'><div style='font-weight:700;margin-bottom:4px;'>Party "+(idx+1)+"</div><table>"+cells.join("")+"</table></div>");
      }});
    }}
    if (parties.length) {{
      var types = parties.map(function(p){{ return p.Vehicle1TypeDesc || ""; }}).filter(function(v){{ return v; }});
      if (types.length) {{
        summaryRows.push("<div class='kv'><div class='k'>Vehicles</div><div class='v'>"+escHtml(types.join(', '))+"</div></div>");
      }}
    }}
    var detailsId = "full_" + (safeId(data.collision_id) || Math.floor(Math.random()*1e9));
    return "<div class='popup-wrap'>" +
      "<div class='popup-title'>Crash summary</div>" +
      "<div class='popup-body'>" +
      summaryRows.join("") +
      "<button class='btn full-toggle' data-target='"+detailsId+"'>Show full report</button>" +
      "<div id='"+detailsId+"' class='full-wrap'><div class='full-table-wrap'>" + (partyRows.length ? partyRows.join("") : "") + "</div></div>" +
      "</div></div>";
  }}

  document.addEventListener("click", function(e){{
    var btn = e.target && e.target.closest ? e.target.closest(".full-toggle[data-target]") : null;
    if (!btn) return;
    var id = btn.getAttribute("data-target");
    if (!id) return;
    var el = document.getElementById(id);
    if (!el) return;
    var shown = (el.style.display === "block");
    el.style.display = shown ? "none" : "block";
    btn.textContent = shown ? "Show full report" : "Hide full report";
    var url = new URL(window.location.href);
    var collision = url.searchParams.get("collision") || "";
    var map = getMap();
    var center = map && map.getCenter ? map.getCenter() : null;
    setUrlParam(collision, center ? center.lat : null, center ? center.lng : null, map ? map.getZoom() : null, !shown);
  }});

  function setUrlParam(id, lat, lng, zoom, full){{
    var url = new URL(window.location.href);
    if (id) {{
      url.searchParams.set("collision", id);
    }} else {{
      url.searchParams.delete("collision");
    }}
    if (typeof lat === "number" && typeof lng === "number") {{
      url.searchParams.set("lat", lat.toFixed(6));
      url.searchParams.set("lng", lng.toFixed(6));
    }}
    if (typeof zoom === "number") {{
      url.searchParams.set("zoom", String(Math.round(zoom)));
    }}
    if (full === true) {{
      url.searchParams.set("full", "1");
    }} else if (full === false) {{
      url.searchParams.delete("full");
    }}
    window.history.replaceState({{}}, "", url.toString());
  }}

  function applyUrlView(map, lat, lng, zoom){{
    if (!map || isNaN(lat) || isNaN(lng)) return;
    var z = isNaN(zoom) ? map.getZoom() : zoom;
    map.setView([lat, lng], z, {{animate: false}});
  }}

  function openFromUrl(){{
    var url = new URL(window.location.href);
    var id = url.searchParams.get("collision");
    var map = getMap();
    if (!map) return;
    if (id) {{
      var m = markerById[id];
      if (m && m.getLatLng) {{
        map.setView(m.getLatLng(), Math.max(map.getZoom(), 13), {{animate: false}});
        m.openPopup();
        var full = url.searchParams.get("full") === "1";
        if (full) {{
          setTimeout(function(){{
            var popupEl = map._popup && map._popup.getElement ? map._popup.getElement() : null;
            var btn = popupEl ? popupEl.querySelector(".full-toggle[data-target]") : null;
            if (btn && btn.textContent.indexOf("Show") !== -1) btn.click();
          }}, 50);
        }}
      }}
    }}
    var lat = parseFloat(url.searchParams.get("lat"));
    var lng = parseFloat(url.searchParams.get("lng"));
    var zoom = parseInt(url.searchParams.get("zoom"), 10);
    if (!isNaN(lat) && !isNaN(lng)) {{
      applyUrlView(map, lat, lng, zoom);
      // Re-apply once to override any late fitBounds calls
      setTimeout(function(){{ applyUrlView(map, lat, lng, zoom); }}, 300);
    }}
  }}

  var mapEventsBound = false;
  var urlReady = false;

  function bindMapEvents(){{
    var map = getMap();
    if (!map || mapEventsBound) return;
    mapEventsBound = true;
    map.on('zoomend', adjustRadius);
    map.on('moveend', function(){{
      if (!urlReady) return;
      var center = map.getCenter ? map.getCenter() : null;
      var url = new URL(window.location.href);
      var collision = url.searchParams.get("collision") || "";
      var full = url.searchParams.get("full") === "1";
      setUrlParam(collision, center ? center.lat : null, center ? center.lng : null, map.getZoom(), full ? true : false);
      if (typeof window.applyTypeFilters === "function") {{
        setTimeout(window.applyTypeFilters, 0);
      }}
    }});
    map.on('popupopen', function(e){{
      if (e && e.popup && e.popup._source && e.popup._source.options) {{
        var src = e.popup._source;
        var id = src.options._collisionId || "";
        var center = map.getCenter ? map.getCenter() : null;
        setUrlParam(id, center ? center.lat : null, center ? center.lng : null, map.getZoom());
      }}
    }});
    map.on('popupclose', function(){{
      var center = map.getCenter ? map.getCenter() : null;
      setUrlParam("", center ? center.lat : null, center ? center.lng : null, map.getZoom(), false);
    }});
    if (map.whenReady) {{
      map.whenReady(function(){{
        openFromUrl();
        urlReady = true;
      }});
    }} else {{
      openFromUrl();
      urlReady = true;
    }}
  }}

  function adjustRadius(){{
    var map = getMap();
    if (!map) return;
    var R = r(map.getZoom());
    var HR = hitR(map.getZoom());
    allMarkers.forEach(function(o){{
      if(!o.m || !o.m.setRadius) return;
      if(o.hidden) {{
        o.m.setRadius(0);
        if (o.h && o.h.setRadius) o.h.setRadius(0);
        return;
      }}
      o.r = R;
      o.m.setRadius(R);
      if (o.h && o.h.setRadius) o.h.setRadius(HR);
    }});
  }}

  function buildInBatches(items){{
    var i = 0;
    var batch = 400;
    function step(){{
      var end = Math.min(i + batch, items.length);
      for (; i < end; i++) addMarker(items[i]);
      if (i < items.length) {{
        setTimeout(step, 0);
      }} else {{
        adjustRadius();
        if (typeof window.applyTypeFilters === "function") {{
          window.applyTypeFilters();
        }} else {{
          window._pendingTypeFilter = true;
        }}
        if (!urlReady) {{
          openFromUrl();
          urlReady = true;
        }}
        bindMapEvents();
      }}
    }}
    step();
  }}

  fetch(dataUrl).then(r=>r.json()).then(function(items){{
    buildInBatches(items || []);
  }});

  // In case map is available immediately
  bindMapEvents();
}})();
</script>
"""))

# Robust UI JS: insert All/None + title exactly once, update legend (using MutationObserver)
counts_json = json.dumps(per_year_counts)
color_json = json.dumps({k: v for k, v in color_map.items()})
top_types_json = json.dumps(top_types)
top_veh_json = json.dumps(top_veh_types)

m.get_root().html.add_child(folium.Element(f"""
<script>
(function(){{
  var perYear = {counts_json};
  var colorMap = {color_json};
  var topTypes = {top_types_json};
  var topVehTypes = {top_veh_json};
  var topVehSet = (function(){{
    var s = {{}};
    topVehTypes.forEach(function(t){{ if (t !== "Other") s[t] = true; }});
    return s;
  }})();
  var otherColor = "{other_color}";
  var unknownColor = "{unknown_color}";
  var selectedCats = {{}};

  function setSearchParam(q){{
    var url = new URL(window.location.href);
    if (q) url.searchParams.set("q", q);
    else url.searchParams.delete("q");
    window.history.replaceState({{}}, "", url.toString());
  }}

  function esc(s){{
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
                    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
  }}

  function selectedYears(){{
    var years = [];
    document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function(lbl){{
      var cb = lbl.querySelector('input[type="checkbox"]');
      var nameEl = lbl.querySelector('span');
      if(!cb || !nameEl) return;
      var name = nameEl.textContent.trim();
      if(cb.checked && perYear[name]) years.push(name);
    }});
    years.sort(function(a,b){{
      var ai = parseInt(a,10), bi = parseInt(b,10);
      var aN = isNaN(ai), bN = isNaN(bi);
      if(aN && !bN) return 1;
      if(!aN && bN) return -1;
      if(aN && bN) return a.localeCompare(b);
      return ai - bi;
    }});
    return years;
  }}

  function sumCounts(yrs){{
    var types = {{}};
    var fatal = 0;
    var total = 0;
    yrs.forEach(function(y){{
      var d = perYear[y];
      if(!d) return;
      total += d.total || 0;
      fatal += d.fatal || 0;
      var t = d.types || {{}};
      for (var k in t) types[k] = (types[k]||0) + t[k];
    }});
    return {{types: types, fatal: fatal, total: total}};
  }}

  function renderLegend(){{
    var yrs = selectedYears();
    var agg = sumCounts(yrs);
    var sub = document.getElementById('legend-sub');
    var body = document.getElementById('legend-body');
    if(!sub || !body) return;

    var label = yrs.length ? yrs.join(', ') : 'None';
    sub.innerHTML = "Showing years: <b>"+esc(label)+"</b> • Total Incidents On Screen: <b id='legend-total'>"+agg.total+"</b>";

    var lines = [
      "<div class='section-head'>",
      "<div>Incident types</div>",
      "<div class='section-actions'>",
      "<button id='inc-all' class='mini-btn'>All</button>",
      "<button id='inc-none' class='mini-btn'>None</button>",
      "</div>",
      "</div>"
    ];
    var cats = [];
    topTypes.forEach(function(t){{
      var ct = agg.types[t] || 0;
      if(ct === 0) return;
      cats.push({{key: t, label: t, count: ct, color: (t === "Unknown") ? unknownColor : (colorMap[t] || otherColor)}});
    }});

    if(topTypes.indexOf("Unknown") === -1 && (agg.types["Unknown"]||0) > 0){{
      cats.push({{key: "Unknown", label: "Unknown", count: (agg.types["Unknown"]||0), color: unknownColor}});
    }}

    var otherCt = 0;
    for (var k in agg.types){{
      if (topTypes.indexOf(k) === -1 && k !== "Unknown") otherCt += agg.types[k];
    }}
    if(otherCt > 0){{
      cats.push({{key: "Other", label: "Other", count: otherCt, color: otherColor}});
    }}

    if(!cats.length){{
      lines.push("<div class='li'><span class='tx'>No incidents</span></div>");
    }} else {{
      cats.forEach(function(c){{
        if(typeof selectedCats[c.key] === "undefined") selectedCats[c.key] = true;
        var checked = selectedCats[c.key] ? "checked" : "";
        lines.push(
          "<label class='li' style='cursor:pointer;'>" +
          "<input class='legend-cat' type='checkbox' data-cat='"+esc(c.key)+"' "+checked+" style='margin-top:2px' />" +
          "<span class='sw' style='background:"+c.color+"'></span>" +
          "<span class='tx'><span class='lbl'>"+esc(c.label)+"</span><span class='ct' data-cat-count='"+esc(c.key)+"'>("+c.count+")</span></span>" +
          "</label>"
        );
      }});
    }}

    lines.push("<div class='li' style='margin-top:8px'><span class='sw' style='background:transparent;border:1px solid #333'></span><span class='tx'><b>Fatal crash</b>: X <span class='ct' id='legend-fatal'>("+agg.fatal+")</span></span></div>");
    body.innerHTML = lines.join('');

    body.querySelectorAll('.legend-cat').forEach(function(cb){{
      cb.addEventListener('change', function(){{
        var cat = cb.getAttribute('data-cat');
        selectedCats[cat] = cb.checked;
        applyTypeFilters();
      }});
    }});
    var incAll = body.querySelector('#inc-all');
    var incNone = body.querySelector('#inc-none');
    if (incAll) incAll.addEventListener('click', function(){{
      Object.keys(selectedCats).forEach(function(k){{ selectedCats[k] = true; }});
      renderLegend();
    }});
    if (incNone) incNone.addEventListener('click', function(){{
      Object.keys(selectedCats).forEach(function(k){{ selectedCats[k] = false; }});
      renderLegend();
    }});
    applyTypeFilters();
  }}

  function renderVehicleFilter(){{
    var panel = document.getElementById('vehicle-filter');
    if(!panel) return;
    if(panel.dataset.built === "1") return;
    panel.dataset.built = "1";
    var html = [
      "<div class='section-head'>",
      "<div>Vehicle types</div>",
      "<div class='section-actions'>",
      "<button id='veh-all' class='mini-btn'>All</button>",
      "<button id='veh-none' class='mini-btn'>None</button>",
      "</div>",
      "</div>"
    ];
    topVehTypes.forEach(function(t){{
      html.push(
        "<label class='li' style='cursor:pointer;'>" +
        "<input class='veh-cat' type='checkbox' data-veh='"+esc(t)+"' checked style='margin-top:2px' />" +
        "<span class='tx'><span class='lbl'>"+esc(t)+"</span><span class='ct' data-veh-count='"+esc(t)+"'>(0)</span></span>" +
        "</label>"
      );
    }});
    panel.innerHTML = html.join('');
    panel.querySelectorAll('.veh-cat').forEach(function(cb){{
      cb.addEventListener('change', function(){{ setTimeout(applyTypeFilters, 0); }});
    }});
    var vehAll = panel.querySelector('#veh-all');
    var vehNone = panel.querySelector('#veh-none');
    if (vehAll) vehAll.addEventListener('click', function(){{
      panel.querySelectorAll('.veh-cat').forEach(function(cb){{ cb.checked = true; }});
      setTimeout(applyTypeFilters, 0);
    }});
    if (vehNone) vehNone.addEventListener('click', function(){{
      panel.querySelectorAll('.veh-cat').forEach(function(cb){{ cb.checked = false; }});
      setTimeout(applyTypeFilters, 0);
    }});
  }}

  function ensureLegendControls(){{
    var legend = document.getElementById('legend');
    if(!legend) return;
    var btnAll = document.getElementById('legend-all');
    var btnNone = document.getElementById('legend-none');
    var btnCollapse = document.getElementById('legend-collapse');
    var searchInput = document.getElementById('legend-search-input');
    var searchClear = document.getElementById('legend-search-clear');

    if(btnAll && !btnAll.dataset.bound){{
      btnAll.dataset.bound = "1";
      btnAll.addEventListener('click', function(){{
        Object.keys(selectedCats).forEach(function(k){{ selectedCats[k] = true; }});
        renderLegend();
      }});
    }}
    if(btnNone && !btnNone.dataset.bound){{
      btnNone.dataset.bound = "1";
      btnNone.addEventListener('click', function(){{
        Object.keys(selectedCats).forEach(function(k){{ selectedCats[k] = false; }});
        renderLegend();
      }});
    }}
    if(btnCollapse && !btnCollapse.dataset.bound){{
      btnCollapse.dataset.bound = "1";
      btnCollapse.addEventListener('click', function(){{
        var collapsed = legend.classList.toggle('collapsed');
        btnCollapse.textContent = collapsed ? "Show" : "Hide";
      }});
    }}
    if(searchInput && !searchInput.dataset.bound){{
      searchInput.dataset.bound = "1";
      var url = new URL(window.location.href);
      var q = url.searchParams.get("q");
      if (q) searchInput.value = q;
      searchInput.addEventListener('input', function(){{
        setSearchParam(searchInput.value.trim());
        setTimeout(applyTypeFilters, 0);
      }});
    }}
    if(searchClear && !searchClear.dataset.bound){{
      searchClear.dataset.bound = "1";
      searchClear.addEventListener('click', function(){{
        if (searchInput) searchInput.value = "";
        setSearchParam("");
        setTimeout(applyTypeFilters, 0);
      }});
    }}
    renderVehicleFilter();
  }}

  function categoryForType(t){{
    if (t === "Unknown") return "Unknown";
    if (topTypes.indexOf(t) !== -1) return t;
    return "Other";
  }}

  function applyTypeFilters(){{
    var markers = window._allMarkers || [];
    var map = window._leaflet_map || null;
    var searchInput = document.getElementById('legend-search-input');
    var query = searchInput ? searchInput.value.trim().toLowerCase() : "";
    var terms = query ? query.split(";").map(function(t){{ return t.trim(); }}).filter(function(t){{ return t; }}) : [];
    var requireFatal = false;
    if (terms.length) {{
      var filtered = [];
      terms.forEach(function(t){{
        if (t === "fatal") requireFatal = true;
        else filtered.push(t);
      }});
      terms = filtered;
    }}
    function showLayer(layer, group){{
      if (!layer) return;
      if (group && group.addLayer) {{ group.addLayer(layer); return; }}
      if (map && map.addLayer) map.addLayer(layer);
    }}
    function hideLayer(layer, group){{
      if (!layer) return;
      if (group && group.removeLayer) {{ group.removeLayer(layer); return; }}
      if (map && map.removeLayer) map.removeLayer(layer);
    }}
    var allowedVeh = null;
    var vehChecks = document.querySelectorAll('.veh-cat');
    if (vehChecks.length) {{
      allowedVeh = {{}};
      vehChecks.forEach(function(cb){{ if (cb.checked) allowedVeh[cb.getAttribute('data-veh')] = true; }});
    }}
    var countsByCat = {{}};
    var visibleCount = 0;
    var visibleFatal = 0;

    var bounds = (map && map.getBounds) ? map.getBounds() : null;
    markers.forEach(function(o){{
      var cat = categoryForType(o.t || "Unknown");
      var show = !!selectedCats[cat];
      if (map && o.g && map.hasLayer && !map.hasLayer(o.g)) {{
        show = false;
      }}
      if (show && bounds && o.m && o.m.getLatLng && !bounds.contains(o.m.getLatLng())) {{
        show = false;
      }}
      if (show && terms.length) {{
        var s = o.s || "";
        for (var i = 0; i < terms.length; i++) {{
          if (s.indexOf(terms[i]) === -1) {{ show = false; break; }}
        }}
      }}
      if (show && requireFatal && !o.f) {{
        show = false;
      }}
      if (show && allowedVeh) {{
        var v = o.v || [];
        var ok = false;
        if (allowedVeh["Other"]) {{
          ok = v.some(function(t){{ return !topVehSet[t]; }});
        }}
        if (!ok) {{
          ok = v.some(function(t){{ return !!allowedVeh[t]; }});
        }}
        show = ok;
      }}
      var g = o.g || null;
      o.hidden = !show;
      if (g && g.hasLayer) {{
        if (show) {{
          if (o.h && !g.hasLayer(o.h)) showLayer(o.h, g);
          if (!g.hasLayer(o.m)) showLayer(o.m, g);
          if (o.f && !g.hasLayer(o.f)) showLayer(o.f, g);
        }} else {{
          if (o.h && g.hasLayer(o.h)) hideLayer(o.h, g);
          if (g.hasLayer(o.m)) hideLayer(o.m, g);
          if (o.f && g.hasLayer(o.f)) hideLayer(o.f, g);
        }}
      }} else {{
        if (show) {{
          showLayer(o.h, g);
          showLayer(o.m, g);
          showLayer(o.f, g);
        }} else {{
          hideLayer(o.h, g);
          hideLayer(o.m, g);
          hideLayer(o.f, g);
        }}
      }}
      if (o.m && o.m.setStyle) {{
        o.m.setStyle({{opacity: show ? 1 : 0, fillOpacity: show ? 0.8 : 0}});
      }}
      if (o.m && o.m.setRadius) {{
        if (show) {{
          var r = (o.r !== null && typeof o.r !== "undefined") ? o.r : 6;
          o.m.setRadius(r);
          if (o.h && o.h.setRadius) o.h.setRadius(hitR(map && map.getZoom ? map.getZoom() : 10));
        }} else {{
          o.m.setRadius(0);
          if (o.h && o.h.setRadius) o.h.setRadius(0);
        }}
      }}
      if (o.f && o.f.setOpacity) {{
        o.f.setOpacity(show ? 1 : 0);
      }}
      if (show) {{
        visibleCount += 1;
        countsByCat[cat] = (countsByCat[cat] || 0) + 1;
        if (o.f) visibleFatal += 1;
      }}
    }});

    var totalEl = document.getElementById('legend-total');
    if (totalEl) totalEl.textContent = String(visibleCount);
    var fatalEl = document.getElementById('legend-fatal');
    if (fatalEl) fatalEl.textContent = "(" + visibleFatal + ")";
    document.querySelectorAll('[data-cat-count]').forEach(function(el){{
      el.textContent = "(0)";
    }});
    Object.keys(countsByCat).forEach(function(k){{
      var el = document.querySelector("[data-cat-count='"+k.replace(/'/g, "\\'")+"']");
      if (el) el.textContent = "(" + countsByCat[k] + ")";
    }});
    if (allowedVeh) {{
      var vehCounts = {{}};
      markers.forEach(function(o){{
        if (o.hidden) return;
        (o.v || []).forEach(function(t){{
          var key = topVehSet[t] ? t : "Other";
          vehCounts[key] = (vehCounts[key] || 0) + 1;
        }});
      }});
      document.querySelectorAll('[data-veh-count]').forEach(function(el){{
        el.textContent = "(0)";
      }});
      Object.keys(vehCounts).forEach(function(k){{
        var el = document.querySelector("[data-veh-count='"+k.replace(/'/g, "\\'")+"']");
        if (el) el.textContent = "(" + vehCounts[k] + ")";
      }});
    }}
  }}
  window.applyTypeFilters = applyTypeFilters;
  if (window._pendingTypeFilter) {{
    window._pendingTypeFilter = false;
    applyTypeFilters();
  }}

  function setAll(checked){{
    // use click to let Leaflet do layer adds/removes
    document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]').forEach(function(cb){{
      if(cb.checked !== checked){{
        cb.click();
      }}
    }});
    setTimeout(renderLegend, 50);
  }}

  function ensureButtonsOnce(){{
    var control = document.querySelector('.leaflet-control-layers');
    var overlays = document.querySelector('.leaflet-control-layers-overlays');
    var list = control ? control.querySelector('.leaflet-control-layers-list') : null;
    if(!control || !overlays || !list) return false;

    // Hard guard: remove any duplicates if they exist
    var toolsExisting = list.querySelectorAll(':scope > .layer-tools');
    if(toolsExisting.length > 1){{
      for(var i=1;i<toolsExisting.length;i++) toolsExisting[i].remove();
    }}

    var titleExisting = list.querySelectorAll(':scope > .layer-title');
    if(titleExisting.length > 1){{
      for(var j=1;j<titleExisting.length;j++) titleExisting[j].remove();
    }}

    if(list.querySelector(':scope > .layer-tools') && list.querySelector(':scope > .layer-title')) return true;

    if(!list.querySelector(':scope > .layer-title')){{
      var title = document.createElement('div');
      title.className = 'layer-title';
      title.textContent = 'Year filter';
      list.insertBefore(title, list.firstChild);
    }}

    if(!list.querySelector(':scope > .layer-tools')){{
      var tools = document.createElement('div');
      tools.className = 'layer-tools';
      tools.innerHTML = "<button type='button' id='btn-all'>All</button><button type='button' id='btn-none'>None</button>";
      var ref = list.querySelector(':scope > .layer-title');
      list.insertBefore(tools, ref ? ref.nextSibling : list.firstChild);
      tools.querySelector('#btn-all').addEventListener('click', function(e){{ e.preventDefault(); setAll(true); }});
      tools.querySelector('#btn-none').addEventListener('click', function(e){{ e.preventDefault(); setAll(false); }});
    }}

    // Hook legend updates
    control.querySelectorAll('.leaflet-control-layers-overlays input[type=\"checkbox\"]').forEach(function(cb){{
      cb.addEventListener('change', function(){{ setTimeout(renderLegend, 0); }});
    }});

    renderLegend();
    return true;
  }}

  function alignLabelAboveToggle(){{
    var label = document.getElementById('year-filter-label');
    var toggle = document.querySelector('.leaflet-control-layers-toggle');
    if(!label || !toggle) return;

    label.style.display = 'block';
    label.style.left = '0px';
    label.style.top = '0px';

    var r = toggle.getBoundingClientRect();
    var w = label.offsetWidth;
    var h = label.offsetHeight;
    var left = r.left + (r.width / 2) - (w / 2);
    var top = r.top - h - 8;

    // keep on-screen
    var minLeft = 8;
    var maxLeft = window.innerWidth - w - 8;
    left = Math.max(minLeft, Math.min(left, maxLeft));
    top = Math.max(8, top);

    label.style.left = left + 'px';
    label.style.top = top + 'px';
  }}

  function boot(){{
    // MutationObserver to handle Leaflet rebuilding DOM on open/close
    var obs = new MutationObserver(function(){{
      ensureButtonsOnce();
      alignLabelAboveToggle();
      ensureLegendControls();
    }});
    obs.observe(document.body, {{childList:true, subtree:true}});

    // initial
    var tries = 0;
    var t = setInterval(function(){{
      tries++;
      ensureButtonsOnce();
      alignLabelAboveToggle();
      ensureLegendControls();
      renderLegend();
      if(tries > 20) clearInterval(t);
    }}, 200);

    window.addEventListener('resize', function(){{ setTimeout(alignLabelAboveToggle, 0); }});
    window.addEventListener('scroll', function(){{ setTimeout(alignLabelAboveToggle, 0); }}, true);
  }}

  boot();
}})();
</script>
"""))

#out = f"angeles_forest_crashes_year_filter_label_fixed_no_dupes_{ts}.html"
out = f"test.html"

m.save(out)

out
