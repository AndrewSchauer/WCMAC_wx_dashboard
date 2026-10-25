"""
Synoptic_Plots.py — weather-station (RAWS) fetch and plot engine
================================================================

Fetches recent (48-hour) timeseries data for weather stations and produces
matplotlib figures in the same style as Windrose_Plots.py
(temperature + dew point + RH, wind roses + bar charts, precip/snow).

Two data sources, chosen per station in DEFAULT_SYNOPTIC_STATIONS:
  * IEM (Iowa Environmental Mesonet) — free, no key; RAWS via NWS HADS
      https://mesonet.agron.iastate.edu/api/
  * Synoptic Data API — needs a token with access to the stations
      https://docs.synopticdata.com/services/time-series
"""

import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import requests

def make_wind_rose(ax, directions, magnitudes, title, speed_bins, colors):
    import Windrose_Plots as _wr
    return _wr.make_wind_rose(ax, directions, magnitudes, title, speed_bins, colors)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNOPTIC_TOKEN = "66a907d0388c43108d3909592b5ded79"
SYNOPTIC_BASE  = "https://api.synopticdata.com/v2"
BG             = '#0f0f1a'
CMAP_NAME      = 'plasma'

# Iowa Environmental Mesonet (IEM) — free, no API key. Serves the NWS HADS
# feed of RAWS GOES transmissions, keyed by the same NWS IDs Synoptic uses.
# Docs: https://mesonet.agron.iastate.edu/api/
IEM_OBHISTORY = "https://mesonet.agron.iastate.edu/api/1/obhistory.json"

# Default weather stations (WCMAC RAWS — from stations.xlsx).
#   source: "iem"      → Iowa Environmental Mesonet (no key needed)
#           "synoptic" → Synoptic API (needs a token with access to MT RAWS)
#           "auto"     → try Synoptic first, fall back to IEM
#   network: IEM network code (Montana DCP/HADS stations = "MT_DCP")
#   wims_id: NFDRS/WIMS station number, for reference (FEMS uses these)
#   lat/lon/elevation (ft): from IEM station metadata, shown in the UI
DEFAULT_SYNOPTIC_STATIONS = [
    {"name": "Point 6",      "stid": "PNTM8", "source": "iem", "network": "MT_DCP",
     "lat": 47.0411, "lon": -113.98623, "elevation": 7856,
     "wims_id": None},
    {"name": "Seeley Lake",  "stid": "SEEM8", "source": "iem", "network": "MT_DCP",
     "lat": 47.1827, "lon": -113.4472, "elevation": 4249,
     "wims_id": "241508"},
    {"name": "Pistol Creek", "stid": "PSTM8", "source": "iem", "network": "MT_DCP",
     "lat": 47.2167, "lon": -114.0167, "elevation": 5509,
     "wims_id": "241405"},
    {"name": "Smith Creek",  "stid": "SMTM8", "source": "iem", "network": "MT_DCP",
     "lat": 46.43944, "lon": -114.22722, "elevation": 5750,
     "wims_id": "242912"},
]

# Local time zone used for RAWS fetched from IEM
IEM_DEFAULT_TZ = "America/Denver"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WCMACWeatherDashboard/1.0)",
    "Accept":     "application/json",
}

SPEED_BINS = [0, 3, 6, 9, 12, 15]
GUST_BINS  = [0, 5, 10, 15, 20, 25]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt_utc(iso):
    """Parse a Synoptic ISO timestamp as a UTC-aware datetime."""
    s = iso.strip()
    for sep in ('T', ' '):
        if sep in s:
            date_part, rest = s.split(sep, 1)
            time_part = rest[:8]
            dt = datetime.datetime.strptime(date_part + ' ' + time_part,
                                            '%Y-%m-%d %H:%M:%S')
            return dt.replace(tzinfo=datetime.timezone.utc)
    dt = datetime.datetime.strptime(s[:16], '%Y-%m-%d %H:%M')
    return dt.replace(tzinfo=datetime.timezone.utc)


def _get_tz(tz_name):
    """
    Return a tzinfo for tz_name (e.g. 'America/Anchorage').
    Tries zoneinfo (Python 3.9+) then pytz; falls back to UTC.
    """
    if not tz_name:
        return datetime.timezone.utc
    try:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz_name)
        except ImportError:
            import pytz
            return pytz.timezone(tz_name)
    except Exception:
        return datetime.timezone.utc


def _safe_float(v):
    try:    return float(v)
    except: return None


# Unit conversion factors to our display units
# Temperature → °F, Wind speed/gust → mph, Precip/Snow → inches
_TEMP_CONVERSIONS = {
    'celsius':    lambda v: v * 9/5 + 32,
    'c':          lambda v: v * 9/5 + 32,
    'kelvin':     lambda v: (v - 273.15) * 9/5 + 32,
    'k':          lambda v: (v - 273.15) * 9/5 + 32,
}
_SPEED_CONVERSIONS = {
    'm/s':        lambda v: v * 2.23694,
    'mps':        lambda v: v * 2.23694,
    'meters/second': lambda v: v * 2.23694,
    'km/h':       lambda v: v * 0.621371,
    'kph':        lambda v: v * 0.621371,
    'kmh':        lambda v: v * 0.621371,
    'knots':      lambda v: v * 1.15078,
    'kt':         lambda v: v * 1.15078,
    'kts':        lambda v: v * 1.15078,
}
_PRECIP_CONVERSIONS = {
    'mm':         lambda v: v * 0.0393701,
    'millimeters':lambda v: v * 0.0393701,
    'cm':         lambda v: v * 0.393701,
    'centimeters':lambda v: v * 0.393701,
    'm':          lambda v: v * 39.3701,
    'meters':     lambda v: v * 39.3701,
}

def _convert(value, unit_str, conversion_map):
    """Apply a unit conversion if unit_str is in the map; otherwise pass through."""
    if value is None:
        return None
    key = unit_str.strip().lower() if unit_str else ''
    fn = conversion_map.get(key)
    return fn(value) if fn else value


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_synoptic_station(stid, recent_minutes=2880, hours=None):
    """
    Fetch timeseries data from Synoptic API for one station.
    Returns dict with keys: meta, records, has_*, stats,
                            table_headers, table_rows.
    """
    if hours:
        recent_minutes = int(hours) * 60
    url    = f"{SYNOPTIC_BASE}/stations/timeseries"
    params = {"stid": stid, "recent": recent_minutes,
              "token": SYNOPTIC_TOKEN,
              "units": "english",       # request imperial
              "windspeedunit": "mph",   # explicitly request mph (API may return knots otherwise)
              }
    r = requests.get(url, params=params, headers=HEADERS, timeout=45)
    r.raise_for_status()
    data = r.json()

    summary = data.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE", 1) != 1:
        raise ValueError(f"Synoptic error: {summary.get('RESPONSE_MESSAGE','unknown')}")

    stations = data.get("STATION", [])
    if not stations:
        raise ValueError(f"No data returned for STID={stid}")

    st    = stations[0]
    obs   = st.get("OBSERVATIONS", {})
    units = data.get("UNITS", {})  # top-level in Synoptic API response

    meta = {
        "name":      st.get("NAME", stid),
        "stid":      st.get("STID", stid),
        "elevation": _safe_float(st.get("ELEVATION")),
        "lat":       _safe_float(st.get("LATITUDE")),
        "lon":       _safe_float(st.get("LONGITUDE")),
        "state":     st.get("STATE", ""),
        "timezone":  st.get("TIMEZONE", ""),
    }

    datetimes = obs.get("date_time", [])
    n = len(datetimes)

    # Debug: print all observation keys so we can see what the station reports
    print(f"[Synoptic {stid}] observation keys:")
    for k in sorted(obs.keys()):
        if k == "date_time": continue
        non_none = sum(1 for v in obs[k] if v is not None)
        print(f"  {k:50s}  n={len(obs[k])}  non-null={non_none}")

    def _best_set(base_name, extra_aliases=None):
        """
        Find the best-populated sensor set for a variable.
        Scans set_1 through set_5 for base_name and any extra_aliases,
        returning (obs_key, values) for whichever has the most non-None obs.
        Falls back to set_1 (empty list) if nothing found.
        """
        best_key, best_vals, best_count = f"{base_name}_set_1", [None]*n, 0
        for name_candidate in [base_name] + (extra_aliases or []):
            for i in range(1, 6):
                key  = f"{name_candidate}_set_{i}"
                vals = obs.get(key, [])
                if len(vals) != n:
                    continue
                count = sum(1 for v in vals if v is not None)
                if count > best_count:
                    best_key, best_vals, best_count = key, vals, count
        return best_key, best_vals

    temp_key,  temp_vals = _best_set("air_temp")
    dew_key,   dew_vals  = _best_set("dew_point_temperature")
    rh_key,    rh_vals   = _best_set("relative_humidity")
    wspd_key,  wspd_vals = _best_set("wind_speed")
    wgst_key,  wgst_vals = _best_set("wind_gust")
    wdir_key,  wdir_vals = _best_set("wind_direction")
    # Precip: try several Synoptic variable names — stations vary in what they report
    # Stations that report hourly precip (not a running accumulator) via
    # precip_accum_one_hour_set_1.  For these stations we want to plot bars
    # (hourly amounts) + a cumulative line overlay, and sum for the 24-h stat.
    HOURLY_PRECIP_STIDS = {"AYTA2",
                           "AYMA2",
                           "WRTA2"}

    prec_key,  prec_vals = _best_set("precip_accum",
                                     extra_aliases=["precipitation",
                                                    "precip_accum_one_minute",
                                                    "precip_accum_ten_minute",
                                                    "precip_accum_ten_minute_set_1",
                                                    "precip_accum_one_hour",
                                                    "precip_accum_since_local_midnight"])

    # For hourly-precip stations, also grab the raw hourly field explicitly
    # so we can distinguish hourly amounts from accumulated totals.
    hourly_prec_key, hourly_prec_vals = None, [None] * n
    if stid.upper() in HOURLY_PRECIP_STIDS:
        # Try the specific one-hour field first; fall back to whatever _best_set found
        for candidate in ("precip_accum_one_hour",):
            for si in range(1, 6):
                k = f"{candidate}_set_{si}"
                v = obs.get(k, [])
                if len(v) == n and sum(1 for x in v if x is not None) > 0:
                    hourly_prec_key  = k
                    hourly_prec_vals = v
                    break
            if hourly_prec_key:
                break
        # If the hourly field IS what _best_set already found, keep it consistent
        if hourly_prec_key is None and "precip_accum_one_hour" in prec_key:
            hourly_prec_key  = prec_key
            hourly_prec_vals = prec_vals
    snow_key,  snow_vals = _best_set("snow_depth",
                                     extra_aliases=["snow_smoothed",
                                                    "snow_water_equiv"])

    print(f"[Synoptic {stid}] selected: prec_key={prec_key}  snow_key={snow_key}  hourly_prec_key={hourly_prec_key}")


    local_tz = _get_tz(meta["timezone"])

    # Read unit strings for whichever sensor set was selected
    u_temp  = units.get(temp_key,  "")
    u_dew   = units.get(dew_key,   "")
    u_wspd  = units.get(wspd_key,  "")
    u_wgst  = units.get(wgst_key,  "")
    u_prec  = units.get(prec_key,  "")
    u_snow  = units.get(snow_key,  "")
    u_hourly_prec = units.get(hourly_prec_key, u_prec) if hourly_prec_key else u_prec

    # If the API didn't report units, assume wind is in knots (observed default)
    # and temperature in °F (english units). This is a known Synoptic quirk.
    if not u_wspd: u_wspd = 'knots'
    if not u_wgst: u_wgst = 'knots'

    # Log which sensor sets and units were selected
    print(f"[Synoptic {stid}] sensor sets: temp={temp_key}, wspd={wspd_key}, wgst={wgst_key}")
    print(f"[Synoptic {stid}] units: temp={u_temp!r} wspd={u_wspd!r} wgst={u_wgst!r}")

    records = []
    for i, dt_str in enumerate(datetimes):
        try:   dt_utc = _parse_dt_utc(dt_str)
        except: continue
        dt_local = dt_utc.astimezone(local_tz)
        records.append({
            "dt":           dt_local,
            "time_label":   dt_local.strftime("%H:%M"),
            "temp_air":     _convert(_safe_float(temp_vals[i]),  u_temp, _TEMP_CONVERSIONS),
            "dew_point":    _convert(_safe_float(dew_vals[i]),   u_dew,  _TEMP_CONVERSIONS),
            "rh":           _safe_float(rh_vals[i]),   # RH is always %
            "wind_spd":     _convert(_safe_float(wspd_vals[i]),  u_wspd, _SPEED_CONVERSIONS),
            "wind_gust":    _convert(_safe_float(wgst_vals[i]),  u_wgst, _SPEED_CONVERSIONS),
            "wind_dir":     _safe_float(wdir_vals[i]),  # degrees, no conversion
            "precip_accum": _convert(_safe_float(prec_vals[i]),  u_prec, _PRECIP_CONVERSIONS),
            "precip_hourly": _convert(_safe_float(hourly_prec_vals[i]), u_hourly_prec, _PRECIP_CONVERSIONS) if hourly_prec_key else None,
            "snow_depth":   _convert(_safe_float(snow_vals[i]),  u_snow, _PRECIP_CONVERSIONS),
        })
    records.sort(key=lambda r: r["dt"])

    return _package_records(stid, meta, records, hourly_prec_key is not None)


def _package_records(stid, meta, records, has_hourly_field):
    """
    Shared post-processing for any source (Synoptic, IEM): hourly
    aggregation, capability flags, display table and 24h stats.
    `records` must be dicts with keys dt (tz-aware), time_label, temp_air,
    dew_point, rh, wind_spd, wind_gust, wind_dir, precip_accum,
    precip_hourly, snow_depth (display units: °F, mph, in).
    """
    # ── Hourly aggregation ────────────────────────────────────────────────────
    # If the station reports sub-hourly (e.g. every 5-10 min), aggregate to
    # hourly buckets: avg for temp/dew/rh/wind_spd/wind_dir, max for gust/precip/snow.
    def _hourly_aggregate(recs):
        if len(recs) < 4:
            return recs  # too few records to bother
        # Check median gap — if > 30 min, data is already hourly or coarser
        import statistics
        gaps = [(recs[i+1]["dt"] - recs[i]["dt"]).total_seconds() / 60
                for i in range(len(recs) - 1)]
        if not gaps or statistics.median(gaps) > 30:
            return recs
        # Bucket by truncated hour
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in recs:
            hour_key = r["dt"].replace(minute=0, second=0, microsecond=0)
            buckets[hour_key].append(r)
        def _avg(vals):
            v = [x for x in vals if x is not None]
            return sum(v) / len(v) if v else None
        def _circ_avg(degs):
            import math
            d = [x for x in degs if x is not None]
            if not d: return None
            s = sum(math.sin(math.radians(x)) for x in d)
            c = sum(math.cos(math.radians(x)) for x in d)
            return math.degrees(math.atan2(s, c)) % 360
        def _max(vals):
            v = [x for x in vals if x is not None]
            return max(v) if v else None
        def _sum(vals):
            v = [x for x in vals if x is not None]
            return sum(v) if v else None
        aggregated = []
        for hour_key in sorted(buckets):
            grp = buckets[hour_key]
            aggregated.append({
                "dt":           hour_key,
                "time_label":   hour_key.strftime("%H:%M"),
                "temp_air":     _avg([r["temp_air"]     for r in grp]),
                "dew_point":    _avg([r["dew_point"]    for r in grp]),
                "rh":           _avg([r["rh"]           for r in grp]),
                "wind_spd":     _avg([r["wind_spd"]     for r in grp]),
                "wind_gust":    _max([r["wind_gust"]    for r in grp]),
                "wind_dir":     _circ_avg([r["wind_dir"] for r in grp]),
                "precip_accum": _max([r["precip_accum"] for r in grp]),
                "precip_hourly": _sum([r["precip_hourly"] for r in grp]),
                "snow_depth":   _max([r["snow_depth"]   for r in grp]),
            })
        print(f"[Synoptic {stid}] aggregated {len(recs)} sub-hourly records → {len(aggregated)} hourly")
        return aggregated

    records = _hourly_aggregate(records)

    has_temp   = any(r["temp_air"]     is not None for r in records)
    has_wind   = any(r["wind_spd"]     is not None for r in records)
    has_precip = any(r["precip_accum"] is not None for r in records)
    has_hourly_precip = (has_hourly_field and
                         any(r["precip_hourly"] is not None for r in records))
    has_snow   = any(r["snow_depth"]   is not None for r in records)
    has_dew    = any(r["dew_point"]    is not None for r in records)
    has_rh     = any(r["rh"]           is not None for r in records)

    # ── Table ─────────────────────────────────────────────────────────────────
    table_headers = ["Date/Time"]
    if has_temp:   table_headers += ["Temp (°F)", "Dew Pt (°F)", "RH (%)"]
    if has_wind:   table_headers += ["Wind Spd (mph)", "Wind Gust (mph)", "Wind Dir (°)"]
    if has_hourly_precip:
        table_headers += ["Precip (in/hr)"]
    elif has_precip:
        table_headers += ["Precip Accum (in)"]
    if has_snow:   table_headers += ["Snow Depth (in)"]

    table_rows = []
    for rec in reversed(records):
        row = [rec["dt"].strftime("%b %d %H:%M %Z")]
        if has_temp:
            row += [f"{rec['temp_air']:.1f}"  if rec["temp_air"]  is not None else "",
                    f"{rec['dew_point']:.1f}"  if rec["dew_point"] is not None else "",
                    f"{rec['rh']:.0f}"         if rec["rh"]        is not None else ""]
        if has_wind:
            row += [f"{rec['wind_spd']:.1f}"  if rec["wind_spd"]  is not None else "",
                    f"{rec['wind_gust']:.1f}"  if rec["wind_gust"] is not None else "",
                    f"{rec['wind_dir']:.0f}"   if rec["wind_dir"]  is not None else ""]
        if has_hourly_precip:
            row += [f"{rec['precip_hourly']:.2f}" if rec["precip_hourly"] is not None else ""]
        elif has_precip:
            row += [f"{rec['precip_accum']:.2f}" if rec["precip_accum"] is not None else ""]
        if has_snow:
            row += [f"{rec['snow_depth']:.1f}" if rec["snow_depth"] is not None else ""]
        table_rows.append(row)

    # ── Stats (24h slice only) ─────────────────────────────────────────────────
    import datetime as _dt
    _cutoff_24h = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=24)
    records_24h = [r for r in records if r["dt"] >= _cutoff_24h]
    stats_src   = records_24h if records_24h else records  # fallback to all if 24h is empty
    stats = []
    temps = [r["temp_air"]  for r in stats_src if r["temp_air"]  is not None]
    winds = [r["wind_spd"]  for r in stats_src if r["wind_spd"]  is not None]
    gusts = [r["wind_gust"] for r in stats_src if r["wind_gust"] is not None]
    wdirs = [r["wind_dir"]  for r in stats_src if r["wind_dir"]  is not None]
    if temps:
        avg_temp = sum(temps) / len(temps)
        stats += [["Max Temp", f"{max(temps):.1f} °F"],
                  ["Min Temp", f"{min(temps):.1f} °F"],
                  ["Avg Temp", f"{avg_temp:.1f} °F"]]
    if wdirs:
        import math
        sin_sum = sum(math.sin(math.radians(d)) for d in wdirs)
        cos_sum = sum(math.cos(math.radians(d)) for d in wdirs)
        avg_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        cardinals = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        card = cardinals[int((avg_deg + 11.25) / 22.5) % 16]
        stats += [["Wind Dir", card]]
    if winds:
        stats += [["Avg Wind", f"{sum(winds)/len(winds):.1f} mph"]]
    if gusts:
        stats += [["Max Gust", f"{max(gusts):.1f} mph"]]
    # Precip: for hourly-recording stations sum the hourly amounts over 24h;
    # for true accumulators show the max accumulated value in the window.
    if has_hourly_precip:
        hourly_amounts = [r["precip_hourly"] for r in stats_src if r["precip_hourly"] is not None]
        if hourly_amounts:
            stats += [["Precip (24h sum)", f"{sum(hourly_amounts):.2f} in"]]
    elif has_precip:
        precips = [r["precip_accum"] for r in stats_src if r["precip_accum"] is not None]
        if precips:
            stats += [["Precip Accum", f"{max(precips):.2f} in"]]
    # Snow depth: show latest reading and 24h change
    snows = [(r["dt"], r["snow_depth"]) for r in stats_src if r["snow_depth"] is not None]
    if snows:
        snows.sort(key=lambda t: t[0])
        stats += [["Snow Depth", f"{snows[-1][1]:.1f} in"]]
        if len(snows) >= 2:
            delta = snows[-1][1] - snows[0][1]
            stats += [["24h Snow Δ", f"{delta:+.1f} in"]]
    if records:
        stats += [["Obs (24h)", str(len(stats_src))]]

    return {
        "meta":               meta,
        "records":            records,
        "has_temp":           has_temp,
        "has_wind":           has_wind,
        "has_precip":         has_precip,
        "has_hourly_precip":  has_hourly_precip,
        "has_snow":           has_snow,
        "has_dew":            has_dew,
        "has_rh":             has_rh,
        "stats":              stats,
        "table_headers":      table_headers,
        "table_rows":         table_rows,
    }


# ---------------------------------------------------------------------------
# IEM (Iowa Environmental Mesonet) fetch — RAWS via NWS HADS, no API key
# ---------------------------------------------------------------------------
#
# HADS/SHEF fields IEM returns for Montana RAWS (English units):
#   TAIRGZZ  air temperature (°F)          XRIRGZZ  relative humidity (%)
#   USIRGZZ  wind speed (mph)              UDIRGZZ  wind direction (deg)
#   UPHRGZZ  peak wind / gust, hourly (mph) URHRGZZ  peak wind direction (deg)
#   PPHRGZZ  precip, past hour (in)        PCIRGZZ  precip accumulator (in)
#   MMIRGZZ  10-hr fuel moisture (%)       MTIRGZZ  fuel temperature (°F)
# RAWS don't measure snow depth, and their unheated gauges under-catch snow.

def _dewpoint_f(temp_f, rh):
    """Dew point (°F) from air temp (°F) and RH (%) — Magnus formula."""
    import math
    if temp_f is None or rh is None or rh <= 0:
        return None
    tc = (temp_f - 32.0) * 5.0 / 9.0
    g = math.log(min(rh, 100.0) / 100.0) + 17.625 * tc / (243.04 + tc)
    td = 243.04 * g / (17.625 - g)
    return td * 9.0 / 5.0 + 32.0


def _station_config(stid):
    """Return the DEFAULT_SYNOPTIC_STATIONS entry for stid (or {})."""
    sid = (stid or "").strip().upper()
    for st in DEFAULT_SYNOPTIC_STATIONS:
        if st.get("stid", "").upper() == sid:
            return st
    return {}


def fetch_iem_station(stid, network="MT_DCP", hours=48, tz_name=None, name=None):
    """
    Fetch the last `hours` of hourly obs for one station from the IEM
    obhistory JSON service. Returns the same dict as fetch_synoptic_station,
    so build_synoptic_plots() and the summary charts work unchanged.
    """
    tz_name  = tz_name or IEM_DEFAULT_TZ
    local_tz = _get_tz(tz_name)
    now_utc  = datetime.datetime.now(datetime.timezone.utc)
    cutoff   = now_utc - datetime.timedelta(hours=hours)

    # obhistory returns one local calendar day per call — request enough days
    # to cover the window (plus one extra in case of UTC/local date edges).
    now_local = now_utc.astimezone(local_tz)
    n_days    = hours // 24 + 2
    dates     = [(now_local - datetime.timedelta(days=d)).date() for d in range(n_days)]

    def _get_day(d):
        r = requests.get(IEM_OBHISTORY,
                         params={"station": stid, "network": network,
                                 "date": d.isoformat(), "full": "true"},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        payload = r.json()
        return payload if isinstance(payload, list) else payload.get("data", [])

    rows, errors = {}, []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {d: pool.submit(_get_day, d) for d in sorted(set(dates))}
        for d, fut in futures.items():
            try:
                data = fut.result()
            except Exception as exc:
                errors.append(f"{d}: {exc}")
                continue
            for rec in data or []:
                key = rec.get("utc_valid")
                if key:
                    rows[key] = rec

    if not rows:
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise ValueError(f"No IEM data returned for {stid} in network {network}{detail}")

    def _first(rec, *keys):
        for k in keys:
            v = _safe_float(rec.get(k))
            if v is not None:
                return v
        return None

    records = []
    for key, rec in rows.items():
        try:
            dt_utc = datetime.datetime.fromisoformat(
                key.replace("Z", "")).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        if dt_utc < cutoff:
            continue
        dt_local = dt_utc.astimezone(local_tz)
        temp = _first(rec, "TAIRGZZ", "tmpf")
        rh   = _first(rec, "XRIRGZZ", "relh")
        spd  = _first(rec, "USIRGZZ")
        if spd is None:
            kt = _first(rec, "sknt")
            spd = kt * 1.15078 if kt is not None else None
        gust = _first(rec, "UPHRGZZ", "UGIRGZZ")
        if gust is None:
            kt = _first(rec, "gust")
            gust = kt * 1.15078 if kt is not None else None
        dew = _first(rec, "dwpf")
        if dew is None:
            dew = _dewpoint_f(temp, rh)
        records.append({
            "dt":            dt_local,
            "time_label":    dt_local.strftime("%H:%M"),
            "temp_air":      temp,
            "dew_point":     dew,
            "rh":            rh,
            "wind_spd":      spd,
            "wind_gust":     gust,
            "wind_dir":      _first(rec, "UDIRGZZ", "drct"),
            "precip_accum":  _first(rec, "PCIRGZZ"),
            "precip_hourly": _first(rec, "PPHRGZZ", "phour", "p01i"),
            "snow_depth":    _first(rec, "SDIRGZZ", "snowdepth"),
        })
    records.sort(key=lambda r: r["dt"])

    # RAWS also send a once-a-day report (daily min/max temp, ~06:00 local)
    # that arrives as its own row with no RH/wind/etc. Left alone it shows up
    # as a gap in the plots, so merge every row in the same clock hour into one,
    # keeping the regular hourly ob's time and filling only missing values.
    merged = {}
    for rec in records:
        key = rec["dt"].replace(minute=0, second=0, microsecond=0)
        filled = sum(v is not None for k, v in rec.items() if k not in ("dt", "time_label"))
        cur = merged.get(key)
        if cur is None:
            merged[key] = (filled, dict(rec))
            continue
        cur_filled, cur_rec = cur
        base, extra = (rec, cur_rec) if filled > cur_filled else (cur_rec, rec)
        out = dict(base)
        for k, v in extra.items():
            if out.get(k) is None and v is not None:
                out[k] = v
        merged[key] = (max(filled, cur_filled), out)
    records = [r for _, r in sorted(merged.values(), key=lambda t: t[1]["dt"])]
    # Drop rows that still have no temperature, RH or wind at all
    records = [r for r in records
               if any(r[k] is not None for k in ("temp_air", "rh", "wind_spd"))]

    if not records:
        raise ValueError(f"IEM returned no observations for {stid} in the last {hours} h")

    cfg  = _station_config(stid)
    meta = {
        "name":      name or cfg.get("name", stid),
        "stid":      stid,
        "elevation": cfg.get("elevation"),
        "lat":       cfg.get("lat"),
        "lon":       cfg.get("lon"),
        "state":     "MT" if network.startswith("MT") else network[:2],
        "timezone":  tz_name,
        "source":    "IEM (NWS HADS)",
    }
    print(f"[IEM {stid}] {len(records)} records "
          f"{records[0]['dt']:%m-%d %H:%M} → {records[-1]['dt']:%m-%d %H:%M %Z}")
    has_hourly = any(r["precip_hourly"] is not None for r in records)
    return _package_records(stid, meta, records, has_hourly)


# NWS station-ID suffix → IEM DCP network, for stations added by hand
_NWSLI_STATE_NETWORK = {
    "M8": "MT_DCP", "I1": "ID_DCP", "W4": "WY_DCP", "W1": "WA_DCP",
    "O3": "OR_DCP", "U1": "UT_DCP", "A2": "AK_DCP",
}


def _infer_iem_network(stid):
    return _NWSLI_STATE_NETWORK.get((stid or "").strip().upper()[-2:], "MT_DCP")


def fetch_station(stid, source=None, network=None, name=None, hours=48):
    """
    Fetch a weather station from the right source.
    If source/network aren't given, they come from DEFAULT_SYNOPTIC_STATIONS.
    Stations not in that list (added in the sidebar) use "auto": Synoptic
    first, then IEM, with the IEM network guessed from the ID (…M8 → MT_DCP).
    """
    cfg     = _station_config(stid)
    source  = (source or cfg.get("source") or "auto").lower()
    network = network or cfg.get("network") or _infer_iem_network(stid)

    if source == "iem":
        return fetch_iem_station(stid, network=network, name=name, hours=hours)
    if source == "auto":
        try:
            return fetch_synoptic_station(stid, hours=hours)
        except Exception as exc:
            print(f"[{stid}] Synoptic failed ({exc}); falling back to IEM")
            return fetch_iem_station(stid, network=network, name=name, hours=hours)
    return fetch_synoptic_station(stid, hours=hours)


# ---------------------------------------------------------------------------
# Station search
# ---------------------------------------------------------------------------

def search_synoptic_stations(query, state=None, limit=20):
    """
    Search Synoptic Mesonet for stations matching a name or STID fragment.
    Returns list of dicts: {name, stid, elevation, lat, lon, state}.

    Tries STID exact-match first, then falls back to full-text name search.
    Pass state='AK' to restrict to Alaska.
    """
    base_params = {"token": SYNOPTIC_TOKEN, "status": "active"}
    if state:
        base_params["state"] = state

    results = []
    seen    = set()

    def _collect(resp_json):
        for st in resp_json.get("STATION", []):
            sid = st.get("STID", "")
            if sid and sid not in seen:
                seen.add(sid)
                results.append({
                    "name":      st.get("NAME", ""),
                    "stid":      sid,
                    "elevation": _safe_float(st.get("ELEVATION")),
                    "lat":       _safe_float(st.get("LATITUDE")),
                    "lon":       _safe_float(st.get("LONGITUDE")),
                    "state":     st.get("STATE", ""),
                })

    q = query.strip()

    # 1) STID exact / prefix match
    if q:
        try:
            r = requests.get(
                f"{SYNOPTIC_BASE}/stations/metadata",
                params={**base_params, "stid": q.upper()},
                headers=HEADERS, timeout=30)
            r.raise_for_status()
            _collect(r.json())
        except Exception:
            pass

    # 2) Name / text search
    if len(results) < limit and q:
        try:
            r2 = requests.get(
                f"{SYNOPTIC_BASE}/stations/metadata",
                params={**base_params, "q": q},
                headers=HEADERS, timeout=30)
            r2.raise_for_status()
            _collect(r2.json())
        except Exception:
            pass

    return results[:limit]


# ---------------------------------------------------------------------------
# Plot helpers  (WX-style, matching Windrose_Plots.py aesthetics)
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='lightgray', labelsize=16)
    ax.spines[:].set_color('#444466')
    ax.yaxis.label.set_color('lightgray')
    ax.xaxis.label.set_color('lightgray')


def time_axis(dts, long_after_hours=60):
    """
    Work out x-axis ticks, tick labels and vertical guide lines for a list of
    datetimes (one per plotted point, oldest first).
      * up to ~2.5 days: ~12 HH:MM ticks, guide lines every 6 hours
      * longer (7-day view): ticks at midnight ("Sat 10/24") and noon
        ("12:00"), guide lines once a day at midnight
    Returns {"ticks": [idx], "labels": [str], "vlines": [idx], "long": bool}.
    """
    n = len(dts)
    if n == 0:
        return {"ticks": [], "labels": [], "vlines": [], "long": False}
    span_h = (dts[-1] - dts[0]).total_seconds() / 3600.0
    is_long = span_h > long_after_hours
    # index of the first point in each new clock hour
    first_in_hour = [i for i in range(n)
                     if i == 0 or (dts[i].hour, dts[i].date()) != (dts[i-1].hour, dts[i-1].date())]
    if not is_long:
        step   = max(1, n // 12)
        ticks  = list(range(0, n, step))
        labels = [dts[i].strftime('%H:%M') for i in ticks]
        vlines = [i for i in first_in_hour if dts[i].hour in (0, 6, 12, 18)]
    else:
        ticks  = [i for i in first_in_hour if dts[i].hour in (0, 12)]
        labels = [dts[i].strftime('%a %m/%d') if dts[i].hour == 0 else '12:00'
                  for i in ticks]
        vlines = [i for i in first_in_hour if dts[i].hour == 0]
    return {"ticks": ticks, "labels": labels, "vlines": vlines, "long": is_long}


def window_label(hours):
    """'48-Hour' / '7-Day' style label for a plot window length in hours."""
    hours = int(hours)
    return f"{hours // 24}-Day" if hours >= 72 and hours % 24 == 0 else f"{hours}-Hour"


def _vlines_6hr(ax, tax):
    """Vertical guide lines (every 6 h, or daily on long windows)."""
    for i in tax["vlines"]:
        ax.axvline(x=i, color='#7777bb', linewidth=1.3,
                   linestyle='-', alpha=0.95, zorder=8)


def _gridlines(ax, y_min, y_max, minor=2, major=10):
    if minor > 0:
        lo = int(np.floor(y_min / minor)) * minor
        hi = int(np.ceil(y_max  / minor)) * minor + minor
        for y in range(lo, hi, minor):
            ax.axhline(y, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
    if major > 0:
        lo2 = int(np.floor(y_min / major)) * major
        hi2 = int(np.ceil(y_max  / major)) * major + major
        for y in range(lo2, hi2, major):
            ax.axhline(y, color='white', linewidth=1.1, linestyle='-', zorder=1)
        # Snap ticks to major/minor grid for temp (minor=1, major=5)
        if major == 5 and minor == 1:
            from matplotlib.ticker import MultipleLocator
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))


def _apply_xaxis(ax, tax, n):
    ax.set_xticks(tax["ticks"])
    ax.set_xticklabels(tax["labels"], fontsize=13.0, color='lightgray',
                       rotation=30, ha='right')
    ax.set_xlim(-0.5, n - 0.5)
    _vlines_6hr(ax, tax)


def _speed_colors(vals, ylim):
    cmap = matplotlib.colormaps.get_cmap(CMAP_NAME)
    norm = np.clip(np.asarray(vals, dtype=float) / max(float(ylim), 1), 0, 1)
    return [cmap(0.15 + 0.7 * v) for v in norm]


def _tiered_ylim(vals):
    peak = float(max(vals)) if len(vals) else 0
    if   peak > 100: return 150
    elif peak > 50:  return 100
    else:            return max(50, int(peak * 1.25) + 5)


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def build_synoptic_plots(name, station_data):
    """
    Build WX-style matplotlib figures for a Synoptic station.
    Figures match Windrose_Plots.py aesthetics exactly.
    Returns dict: temp, wind, precip (Figures or None), stats, date_range.
    """
    records = station_data["records"]
    if not records:
        return {"temp": None, "wind": None, "precip": None,
                "stats": [], "date_range": ""}

    matplotlib.rcParams['font.family'] = 'DejaVu Sans Mono'

    times = [r["time_label"] for r in records]
    dts   = [r["dt"]         for r in records]
    x     = np.arange(len(records))
    tax   = time_axis(dts)           # ticks / labels / guide lines

    date_range = ""
    if dts:
        date_range = (f"{dts[0].strftime('%b %d %H:%M')} \u2013 "
                      f"{dts[-1].strftime('%b %d %H:%M %Z')}")

    plots = {"stats": station_data["stats"], "date_range": date_range}

    # ── TEMPERATURE (+ dew point overlay, RH subplot) ───────────────────────
    if station_data["has_temp"]:
        temps   = [r["temp_air"]  for r in records]
        dews    = [r["dew_point"] for r in records]
        rhs     = [r["rh"]        for r in records]
        has_dew = station_data.get("has_dew", any(v is not None for v in dews))
        has_rh  = station_data.get("has_rh",  any(v is not None for v in rhs))

        n_rows  = 2 if has_rh else 1
        h_ratio = [6, 3] if has_rh else [6]

        fig, axes = plt.subplots(
            n_rows, 1, figsize=(15.0, sum(h_ratio)),
            facecolor=BG, layout='constrained', squeeze=False,
            gridspec_kw={'height_ratios': h_ratio} if n_rows > 1 else {})
        ax_t = axes[0][0]
        _style_ax(ax_t)

        t_clean = np.array([v if v is not None else np.nan for v in temps])
        d_clean = np.array([v if v is not None else np.nan for v in dews]) if has_dew else None

        # Compute y range (include dew point)
        valid_t = t_clean[~np.isnan(t_clean)]
        if len(valid_t):
            t_mn, t_mx = float(valid_t.min()), float(valid_t.max())
            span = max(t_mx - t_mn, 1.0)
            pad  = max(span * 0.12, 5)
            y_lo, y_hi = t_mn - pad, t_mx + pad
            if has_dew and d_clean is not None:
                valid_d = d_clean[~np.isnan(d_clean)]
                if len(valid_d):
                    y_lo = min(y_lo, float(valid_d.min()) - pad * 0.4)
        else:
            y_lo, y_hi, span = -20, 50, 70

        _gridlines(ax_t, y_lo, y_hi, minor=1, major=5)
        _vlines_6hr(ax_t, tax)

        ax_t.fill_between(x, y_lo, t_clean, alpha=0.15, color='#ff9944', zorder=2)
        ax_t.plot(x, t_clean, color='#ffaa55', linewidth=2.5,
                  label='Temp (°F)', zorder=3)
        if has_dew and d_clean is not None:
            ax_t.plot(x, d_clean, color='#44aaff', linewidth=1.8,
                      linestyle='--', label='Dew Pt (°F)', zorder=4)

        if y_lo <= 32 <= y_hi:
            ax_t.axhline(32, color='#6688aa', linestyle='--',
                         linewidth=1.0, alpha=0.9, zorder=2)
            ax_t.text(x[-1] if len(x) else 0, 32 + span * 0.02,
                      '32°F', color='#6688aa', fontsize=14.0, ha='right')

        ax_t.set_ylim(y_lo, y_hi)
        ax_t.autoscale(False)
        ax_t.set_title('Temperature (°F)', color='white', fontsize=22.0,
                       fontweight='bold', pad=8)
        ax_t.set_ylabel('°F', color='lightgray', fontsize=18.0)
        ax_t.legend(loc='upper left', fontsize=14.0, facecolor='#16213e',
                    edgecolor='#666688', labelcolor='white')
        ax_t.grid(False)
        _apply_xaxis(ax_t, tax, len(x))

        if has_rh:
            ax_r = axes[1][0]
            _style_ax(ax_r)
            rh_clean = np.array([v if v is not None else np.nan for v in rhs])
            _gridlines(ax_r, 0, 105, minor=0, major=25)
            _vlines_6hr(ax_r, tax)
            ax_r.fill_between(x, rh_clean, alpha=0.35, color='#00e676', zorder=2)
            ax_r.plot(x, rh_clean, color='#00e676', linewidth=1.8, zorder=3)
            ax_r.set_ylim(0, 105)
            ax_r.autoscale(False)
            ax_r.set_title('Relative Humidity (%)', color='white', fontsize=20.0,
                           fontweight='bold', pad=6)
            ax_r.set_ylabel('RH %', color='lightgray', fontsize=18.0)
            ax_r.grid(False)
            _apply_xaxis(ax_r, tax, len(x))

        fig.suptitle(f'{name} \u2014 Temperature', fontsize=28.0,
                     fontweight='bold', color='white')
        plots['temp'] = fig

    # ── WIND: 2×2 WX-style (roses top-left/bottom-left, bars top-right/bot-right) ──
    if station_data["has_wind"]:
        wspds    = [r["wind_spd"]  for r in records]
        wgsts    = [r["wind_gust"] for r in records]
        wdirs    = [r["wind_dir"]  for r in records]
        has_gust = any(v is not None for v in wgsts)
        has_dir  = any(v is not None for v in wdirs)

        spd_arr = np.array([v if v is not None else 0.0 for v in wspds])
        gst_arr = np.array([v if v is not None else 0.0 for v in wgsts]) if has_gust else None

        cmap      = matplotlib.colormaps.get_cmap(CMAP_NAME)
        sp_colors = [cmap(0.15 + 0.7 * i / max(len(SPEED_BINS) - 1, 1))
                     for i in range(len(SPEED_BINS))]
        g_colors  = [cmap(0.15 + 0.7 * i / max(len(GUST_BINS) - 1, 1))
                     for i in range(len(GUST_BINS))]

        if has_dir:
            fig = plt.figure(figsize=(15.0, 10), facecolor=BG)
            gs  = gridspec.GridSpec(2, 3, figure=fig,
                                    hspace=0.45, wspace=0.30,
                                    left=0.05, right=0.97,
                                    top=0.90, bottom=0.08,
                                    width_ratios=[3, 5, 1.2])
            ax_spd_rose  = fig.add_subplot(gs[0, 0], projection='polar')
            ax_spd_bar   = fig.add_subplot(gs[0, 1])
            ax_leg       = fig.add_subplot(gs[:, 2])   # legend spans both rows
            ax_gust_rose = fig.add_subplot(gs[1, 0], projection='polar')
            ax_gust_bar  = fig.add_subplot(gs[1, 1])
            ax_spd_rose.set_facecolor('#1a1a2e')
            ax_gust_rose.set_facecolor('#1a1a2e')
        else:
            fig = plt.figure(figsize=(15.0, 6 if has_gust else 6), facecolor=BG)
            gs  = gridspec.GridSpec(2 if has_gust else 1, 1, figure=fig,
                                    hspace=0.45,
                                    left=0.08, right=0.97,
                                    top=0.88, bottom=0.10)
            ax_spd_bar   = fig.add_subplot(gs[0])
            ax_gust_bar  = fig.add_subplot(gs[1]) if has_gust else None
            ax_spd_rose  = ax_gust_rose = None

        def _wind_bar(ax, vals, label, ylim):
            _style_ax(ax)
            bar_colors = _speed_colors(vals, ylim)
            for y in range(0, ylim + 1, 5):
                ax.axhline(y, color='white', linewidth=1.1, linestyle='-', zorder=1)
            _vlines_6hr(ax, tax)
            ax.bar(x, vals, color=bar_colors, width=0.8, zorder=2, edgecolor='none')
            ax.plot(x, vals, color='white', linewidth=0.8, alpha=0.4, zorder=3)
            # Wind barbs
            if has_dir:
                dir_arr   = np.array([v if v is not None else 0.0 for v in wdirs])
                barb_step = max(1, len(x) // 30)
                for bi in range(0, len(x), barb_step):
                    spd = float(vals[bi])
                    if spd < 1: continue
                    rad = np.deg2rad(float(dir_arr[bi]))
                    u = -np.sin(rad); vb = -np.cos(rad)
                    ax.barbs(x[bi], min(spd, ylim * 0.95), u * spd, vb * spd,
                             length=6, linewidth=0.8,
                             barb_increments=dict(half=5, full=10, flag=50),
                             color='white', alpha=0.75, zorder=5)
            if len(vals):
                pi = int(np.argmax(vals))
                ax.annotate(f'{vals[pi]:.0f}',
                            xy=(x[pi], vals[pi]), xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', fontsize=14.0, color='white', fontweight='bold')
            ax.set_title(label, color='white', fontsize=22.0, fontweight='bold', pad=8)
            ax.set_ylabel('mph', color='lightgray', fontsize=18.0)
            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.set_ylim(0, ylim)
            ax.grid(False)
            _apply_xaxis(ax, tax, len(x))

        spd_ylim = _tiered_ylim(spd_arr)
        _wind_bar(ax_spd_bar, spd_arr, 'Avg Wind Speed (mph)', spd_ylim)

        if has_gust and gst_arr is not None and ax_gust_bar is not None:
            gst_ylim = _tiered_ylim(gst_arr)
            _wind_bar(ax_gust_bar, gst_arr, 'Max Gust (mph)', gst_ylim)

        if has_dir and ax_spd_rose is not None:
            spd_pairs = [(d, s) for d, s in zip(wdirs, wspds)
                         if d is not None and s is not None]
            handles, leg_labels = [], []
            if spd_pairs:
                dirs_s, spds_s = zip(*spd_pairs)
                handles, leg_labels = make_wind_rose(ax_spd_rose, list(dirs_s), list(spds_s),
                               'Wind Speed', SPEED_BINS, sp_colors)
            if has_gust and ax_gust_rose is not None:
                gst_pairs = [(d, g) for d, g in zip(wdirs, wgsts)
                             if d is not None and g is not None]
                if gst_pairs:
                    dirs_g, gsts_g = zip(*gst_pairs)
                    make_wind_rose(ax_gust_rose, list(dirs_g), list(gsts_g),
                                   'Gust', GUST_BINS, g_colors)
            # Draw shared legend in side column
            if handles:
                import Windrose_Plots as _wr2
                _wr2.draw_rose_legend(ax_leg, handles, leg_labels, SPEED_BINS)
            else:
                ax_leg.set_axis_off()

        fig.suptitle(f'{name} \u2014 Wind', fontsize=28.0,
                     fontweight='bold', color='white')
        plots['wind'] = fig

    # ── PRECIP / SNOW ─────────────────────────────────────────────────────────
    has_hourly_precip = station_data.get("has_hourly_precip", False)
    if station_data["has_precip"] or has_hourly_precip or station_data["has_snow"]:
        has_p = station_data["has_precip"] or has_hourly_precip
        has_s = station_data["has_snow"]

        if has_hourly_precip:
            # Use hourly amounts for the precip panel
            p_raw = np.array([r["precip_hourly"] if r["precip_hourly"] is not None else np.nan
                               for r in records])
        else:
            p_raw = np.array([r["precip_accum"] if r["precip_accum"] is not None else np.nan
                              for r in records])

        s_vals = np.array([v if v is not None else np.nan
                           for v in [r["snow_depth"] for r in records]])

        p_ok = has_p and not np.all(np.isnan(p_raw))
        s_ok = has_s and not np.all(np.isnan(s_vals))

        n_rows = (1 if p_ok else 0) + (1 if s_ok else 0)
        fig, axes_raw = plt.subplots(n_rows, 1, figsize=(15.0, 6 * n_rows),
                                     facecolor=BG, layout='constrained',
                                     squeeze=False)
        ax_idx = 0

        # ── Precip panel ──────────────────────────────────────────────────────
        if p_ok:
            ax_p = axes_raw[ax_idx][0]; ax_idx += 1
            _style_ax(ax_p)
            _vlines_6hr(ax_p, tax)

            if has_hourly_precip:
                # ── Hourly bar chart + cumulative line overlay ─────────────
                hourly = np.where(np.isnan(p_raw), 0.0, p_raw)
                cumul  = np.nancumsum(np.where(np.isnan(p_raw), 0.0, p_raw))

                bar_max  = float(np.nanmax(p_raw)) if not np.all(np.isnan(p_raw)) else 0.0
                cum_max  = float(cumul[-1]) if len(cumul) else 0.0
                p_hi_bar = max(bar_max * 1.25, 0.05)   # primary (bar) axis ceiling

                # Gridlines on bar axis
                import math as _math
                minor_inc = 0.01 if p_hi_bar < 0.2 else (0.05 if p_hi_bar < 1.0 else 0.1)
                major_inc = minor_inc * 5
                for yv in np.arange(0, p_hi_bar + minor_inc, minor_inc):
                    ax_p.axhline(yv, color='#bbbbcc', linewidth=0.5, linestyle='--', zorder=1)
                for yv in np.arange(0, p_hi_bar + major_inc, major_inc):
                    ax_p.axhline(yv, color='white', linewidth=1.0, linestyle='-', zorder=1)

                # Bars (hourly amounts)
                ax_p.bar(x, hourly, color='#5599ff', alpha=0.75, width=0.8,
                         zorder=2, edgecolor='none', label='Hourly Precip (in)')
                ax_p.set_ylim(0, p_hi_bar)
                ax_p.set_ylabel('Precip (in/hr)', color='#5599ff', fontsize=16.0)
                ax_p.yaxis.label.set_color('#5599ff')
                ax_p.tick_params(axis='y', colors='#5599ff', labelsize=16)

                # Cumulative line on twin axis
                ax_cum = ax_p.twinx()
                ax_cum.set_facecolor('none')
                ax_cum.tick_params(colors='#ffdd55', labelsize=14)
                ax_cum.spines['right'].set_color('#ffdd55')
                ax_cum.spines['left'].set_color('#444466')
                ax_cum.spines['top'].set_color('#444466')
                ax_cum.spines['bottom'].set_color('#444466')
                ax_cum.plot(x, cumul, color='#ffdd55', linewidth=2.5,
                            zorder=5, label='Cumulative (in)')
                cum_hi = max(cum_max * 1.15, 0.05)
                ax_cum.set_ylim(0, cum_hi)
                ax_cum.set_ylabel('Cumulative Precip (in)', color='#ffdd55', fontsize=16.0)
                ax_cum.yaxis.label.set_color('#ffdd55')

                # Combined legend
                from matplotlib.lines  import Line2D
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor='#5599ff', alpha=0.75, label='Hourly Precip (in)'),
                    Line2D([0], [0], color='#ffdd55', linewidth=2.5, label='Cumulative (in)'),
                ]
                ax_p.legend(handles=legend_handles, loc='upper left', fontsize=13.0,
                            facecolor='#16213e', edgecolor='#666688', labelcolor='white')
                ax_p.set_title('Hourly Precip & Cumulative Total', color='white',
                               fontsize=22.0, fontweight='bold', pad=8)
            else:
                # ── Original accumulated-precip line plot ─────────────────
                pv   = np.where(np.isnan(p_raw), 0.0, p_raw)
                p_lo = max(0.0, float(np.nanmin(p_raw)) - 1)
                p_hi  = float(np.nanmax(p_raw)) + 1
                for yv in np.arange(p_lo, p_hi + 0.1, 0.1):
                    ax_p.axhline(yv, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
                for yv in np.arange(p_lo, p_hi + 0.5, 0.5):
                    ax_p.axhline(yv, color='white', linewidth=1.1, linestyle='-', zorder=1)
                ax_p.fill_between(x, p_lo, pv, alpha=0.20, color='#5599ff', zorder=2)
                ax_p.plot(x, np.where(np.isnan(p_raw), np.nan, p_raw),
                          color='#5599ff', linewidth=2.5, zorder=3)
                ax_p.set_ylim(p_lo, p_hi); ax_p.autoscale(False)
                ax_p.set_title('Precip Accum', color='white', fontsize=22.0,
                               fontweight='bold', pad=8)
                ax_p.set_ylabel('Precip Accum (in)', color='#5599ff', fontsize=16.0)
                ax_p.yaxis.label.set_color('#5599ff')
                ax_p.tick_params(axis='y', colors='#5599ff', labelsize=16)

            ax_p.grid(False)
            _apply_xaxis(ax_p, tax, len(x))

        # ── Snow depth panel ──────────────────────────────────────────────────
        if s_ok:
            ax_s = axes_raw[ax_idx][0]; ax_idx += 1
            s_valid = s_vals[~np.isnan(s_vals)]
            s_mn = float(s_valid.min()); s_mx = float(s_valid.max())
            s_pad = max((s_mx - s_mn) * 0.12, 5)
            s_lo  = s_mn - s_pad; s_hi = s_mx + s_pad
            _style_ax(ax_s)
            _gridlines(ax_s, s_lo, s_hi, minor=1, major=5)
            _vlines_6hr(ax_s, tax)
            safe_s = np.where(np.isnan(s_vals), s_lo, s_vals)
            ax_s.fill_between(x, s_lo, safe_s, alpha=0.20, color='#7799ff', zorder=2)
            ax_s.plot(x, s_vals, color='#8899ff', linewidth=2.5, zorder=3)
            ax_s.set_ylim(s_lo, s_hi); ax_s.autoscale(False)
            ax_s.set_title('Snow Depth', color='white', fontsize=22.0,
                           fontweight='bold', pad=8)
            ax_s.set_ylabel('Snow Depth (in)', color='#7799ff', fontsize=16.0)
            ax_s.yaxis.label.set_color('#7799ff')
            ax_s.tick_params(axis='y', colors='#7799ff', labelsize=16)
            ax_s.grid(False)
            _apply_xaxis(ax_s, tax, len(x))

        fig.suptitle(f'{name} \u2014 Precipitation & Snow', fontsize=28.0,
                     fontweight='bold', color='white')
        plots['precip'] = fig

    return plots
