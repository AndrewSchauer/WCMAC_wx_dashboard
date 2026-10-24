"""
Snotel_Plots.py — NRCS SNOTEL / SnoLite data fetch and plot engine
===================================================================

Fetches data from the NRCS AWDB REST API (no API key required) and
produces matplotlib figures for snow, precip, temperature, and wind
(SnoLite stations).

Data is fetched for the current water year (Oct 1 → today) at DAILY
resolution so charts show the full seasonal picture.  Hourly data is
fetched for the most recent 7 days for the wind rose / bar charts on
SnoLite stations that have wind sensors.

NRCS AWDB REST API docs:
  https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
"""

import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

# Wind rose renderer lives in Windrose_Plots; import lazily to avoid circular deps
def make_wind_rose(ax, directions, magnitudes, title, speed_bins, colors):
    import Windrose_Plots as _wr
    return _wr.make_wind_rose(ax, directions, magnitudes, title, speed_bins, colors)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

BG        = '#0f0f1a'
CMAP_NAME = 'plasma'

# Default SNOTEL stations (WCMAC forecast area — from stations.xlsx)
# triplet format: "SITENUM:STATE:NETWORK"
# SNTL = full SNOTEL,  SNTLT = SnoLite
# has_wind=True only for SnoLite stations that report wind speed + direction

DEFAULT_SNOTEL_STATIONS = [
    {"name": "North Fork Jocko Creek", "triplet": "667:MT:SNTL", "has_wind": False},
    {"name": "Stuart Mountain",        "triplet": "901:MT:SNTL", "has_wind": False},
    {"name": "Lolo Pass",              "triplet": "588:ID:SNTL", "has_wind": False},
    {"name": "Twin Lakes",             "triplet": "836:MT:SNTL", "has_wind": False},
    {"name": "Saddle Mountain",        "triplet": "727:MT:SNTL", "has_wind": False},
]

# States searched by the SNOTEL station-search box (comma-separated AWDB codes)
SNOTEL_SEARCH_STATES = "MT,ID"

# All SNOTEL hourly timestamps are converted to this zone for plots, tables
# and the 24h/48h cutoffs, so they line up with the RAWS data.
DISPLAY_TZ = "America/Denver"

# NRCS records most SNOTEL hourly data in PST (dataTimeZone = -8) regardless
# of where the station is. Used when station metadata has no dataTimeZone.
DEFAULT_DATA_TZ_OFFSET = -8.0

STATION_URL_TEMPLATE = "https://wcc.sc.egov.usda.gov/nwcc/site?sitenum={sitenum}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WCMACWeatherDashboard/1.0)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def water_year_start():
    """Return Oct 1 of the current water year as a date string YYYY-MM-DD."""
    today = datetime.date.today()
    year  = today.year if today.month >= 10 else today.year - 1
    return datetime.date(year, 10, 1).isoformat()


def _display_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(DISPLAY_TZ)
    except Exception:
        # Fallback: fixed MST if zoneinfo / tzdata is unavailable
        return datetime.timezone(datetime.timedelta(hours=-7))


def local_now():
    """Current naive datetime in DISPLAY_TZ (Mountain time, DST-aware).
    All SNOTEL hourly timestamps are converted to this zone, so every
    24h/48h cutoff is computed against this clock."""
    return datetime.datetime.now(_display_tz()).replace(tzinfo=None)


# Backwards-compatible alias (older code called this alaska_now)
alaska_now = local_now  # noqa


def station_now(tz_offset_hours):
    """Current naive datetime in a station's AWDB data time zone (fixed offset)."""
    return (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(hours=tz_offset_hours)).replace(tzinfo=None)


def today_str():
    """Current date as YYYY-MM-DD (for daily data end date)."""
    return local_now().date().isoformat()


def now_str(tz_offset_hours=DEFAULT_DATA_TZ_OFFSET):
    """Current datetime as YYYY-MM-DD HH:MM in the station's data time zone,
    used as the AWDB hourly endDate so no recent hours are cut off."""
    return station_now(tz_offset_hours).strftime('%Y-%m-%d %H:%M')


def days_ago(n):
    return (local_now().date() - datetime.timedelta(days=n)).isoformat()


def _to_display_tz(date_str, tz_offset_hours):
    """Convert an AWDB hourly timestamp (naive, station data time zone)
    to a naive 'YYYY-MM-DD HH:MM' string in DISPLAY_TZ."""
    try:
        s = date_str.strip().replace('T', ' ')[:16]
        dt = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M')
        src = datetime.timezone(datetime.timedelta(hours=tz_offset_hours))
        return dt.replace(tzinfo=src).astimezone(_display_tz()).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return date_str


def _data_tz_offset(meta):
    """Station data time-zone offset (hours) from AWDB metadata, default PST."""
    try:
        v = meta.get("dataTimeZone")
        return float(v) if v is not None else DEFAULT_DATA_TZ_OFFSET
    except (TypeError, ValueError):
        return DEFAULT_DATA_TZ_OFFSET


# ---------------------------------------------------------------------------
# AWDB API helpers
# ---------------------------------------------------------------------------

def _get(endpoint, params, timeout=45):
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        raise ImportError(
            "The 'requests' package is required for SNOTEL data. "
            "Install it with:  pip install requests"
        )
    url = f"{AWDB_BASE}/{endpoint}"
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    try:
        r = session.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"AWDB API timed out after {timeout}s — NRCS server may be slow or unreachable")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"AWDB API connection failed — server may be unreachable from this network: {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"AWDB API HTTP error: {e}")


def get_station_metadata(triplet):
    """Return metadata dict for a single station triplet.
    Returns a stub dict if the NRCS AWDB server is unreachable."""
    try:
        data = _get("stations", {"stationTriplets": triplet}, timeout=25)
        return data[0] if data else {}
    except Exception as exc:
        print(f"[SNOTEL metadata] Could not fetch metadata for {triplet}: {exc}")
        return {"stationTriplet": triplet, "name": triplet.split(":")[0]}


def get_daily_data(triplet, elements, begin_date=None, end_date=None):
    """
    Fetch daily data for a station.  Returns a dict keyed by element code,
    each value being a list of {"date": "YYYY-MM-DD", "value": float|None}.
    """
    if begin_date is None: begin_date = water_year_start()
    if end_date   is None: end_date   = today_str()
    params = {
    "stationTriplets": triplet,
    "elements": ",".join(elements),
    "beginDate": begin_date,
    "endDate": end_date,
    "duration": "DAILY",
    "returnFlags": "false",
    }

    data = _get("data", params)

    result = {}
    if not data:
        return result

    station_data = data[0] if isinstance(data, list) else data
    for elem_block in station_data.get("data", []):
        se = elem_block.get("stationElement", {}) or {}
        code = se.get("elementCode") or se.get("elementCd") or "?"
        values = elem_block.get("values", [])
        result[code] = [
            {"date": v.get("date", ""), "value": _cast(v.get("value"))}
            for v in values
        ]
    return result


def get_hourly_data(triplet, elements, days_back=7,
                    tz_offset_hours=DEFAULT_DATA_TZ_OFFSET):
    """Fetch recent hourly data. Returned dates are converted to DISPLAY_TZ."""
    begin = days_ago(days_back)
    end   = now_str(tz_offset_hours)  # station data time zone, so no hours are cut off

    params = {
    "stationTriplets": triplet,
    "elements": ",".join(elements),
    "beginDate": begin,
    "endDate": end,
    "duration": "HOURLY",
    "returnFlags": "false",
    }


    data = _get("data", params)

    result = {}
    if not data:
        return result

    station_data = data[0] if isinstance(data, list) else data
    for elem_block in station_data.get("data", []):
        se = elem_block.get("stationElement", {}) or {}
        code = se.get("elementCode") or se.get("elementCd") or "?"
        values = elem_block.get("values", [])
        result[code] = [
            {"date": _to_display_tz(v.get("date", ""), tz_offset_hours),
             "value": _cast(v.get("value"))}
            for v in values
        ]
    return result


def _cast(v):
    """Cast to float, return None on failure."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def fetch_snotel_station(triplet, has_wind=False):
    """
    Main entry point.  Fetches metadata + all available data for a station.

    Returns a dict:
        meta         : station metadata
        daily        : dict of element→[{date, value}, ...]  (water-year daily)
        hourly_wind  : dict of element→[{date, value}, ...]  (7-day hourly, or {})
        hourly_24h   : dict of element→[{date, value}, ...]  (24h hourly for snow/SWE/temp)
        caps         : {has_swe, has_depth, has_precip, has_temp, has_wind}
        table_rows   : list of row lists for the HTML table (most recent first)
        table_headers: list of column header strings
    """
    meta = get_station_metadata(triplet)
    tz_off = _data_tz_offset(meta)

    # Elements to request for daily data
    daily_elements = ["WTEQ", "SNWD", "PREC", "TAVG", "TMAX", "TMIN"]
    daily = get_daily_data(triplet, daily_elements)

    hourly_wind = {}
    if has_wind:
        hourly_wind = get_hourly_data(triplet, ["WSPDV", "WSPDX", "WDIRV"], days_back=7,
                                      tz_offset_hours=tz_off)

    # 48-hour hourly data for snow depth, SWE, and temperature (for plotting)
    hourly_48h_raw = get_hourly_data(triplet, ["WTEQ", "SNWD", "TOBS", "PREC"], days_back=3,
                                     tz_offset_hours=tz_off)

    # Debug: show what the raw hourly fetch returned
    print(f"[SNOTEL {triplet}] hourly_48h_raw keys: { {k: len(v) for k, v in hourly_48h_raw.items()} }")
    # Debug: show actual date range in raw data and the cutoffs being applied
    tobs_raw = hourly_48h_raw.get("TOBS", [])
    if tobs_raw:
        print(f"[SNOTEL {triplet}] TOBS raw date range: {tobs_raw[0]['date']} → {tobs_raw[-1]['date']}")
    print(f"[SNOTEL {triplet}] local_now={local_now().strftime('%Y-%m-%d %H:%M')} "
          f"cutoff_24h={(local_now()-datetime.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')} "
          f"cutoff_48h={(local_now()-datetime.timedelta(hours=48)).strftime('%Y-%m-%d %H:%M')}")

    # Some stations (e.g. Grandview 956) only report daily data, not hourly.
    # If hourly is empty, synthesise a daily-resolution series from daily data.
    # We skip the 48h cutoff for these since daily entries are sparse by nature.
    if not hourly_48h_raw:
        print(f"[SNOTEL {triplet}] no hourly data — falling back to daily for plots")
        daily_as_hourly = {}
        elem_map = {"WTEQ": "WTEQ", "SNWD": "SNWD", "PREC": "PREC",
                    "TAVG": "TOBS", "TMAX": "TOBS", "TMIN": "TOBS"}
        for elem, target in elem_map.items():
            entries = daily.get(elem, [])
            recent = [e for e in entries
                      if e["value"] is not None
                      and e["date"] >= days_ago(7)]
            if recent and target not in daily_as_hourly:
                daily_as_hourly[target] = [
                    {"date": e["date"] + " 12:00:00", "value": e["value"]}
                    for e in recent
                ]
        print(f"[SNOTEL {triplet}] daily_as_hourly keys: { {k: len(v) for k, v in daily_as_hourly.items()} }")
        # For daily fallback, skip the 48h cutoff — use all available daily entries
        hourly_24h       = {k: sorted(v, key=lambda e: e["date"]) for k, v in daily_as_hourly.items()}
        _cutoff_24h      = local_now() - datetime.timedelta(hours=24)
        hourly_24h_stats = {k: [e for e in v if _parse_dt(e["date"]) >= _cutoff_24h]
                            for k, v in hourly_24h.items()}
        print(f"[SNOTEL {triplet}] hourly_24h keys with data (daily fallback): "
              f"{ {k: len([e for e in vals if e['value'] is not None]) for k, vals in hourly_24h.items()} }")
    else:
        # Normal hourly station — apply 48h and 24h cutoffs
        # AWDB returns UTC timestamps; use utcnow() to avoid local-time offset errors
        _now_ak     = local_now()
        _cutoff_48h = _now_ak - datetime.timedelta(hours=48)
        hourly_24h = {k: [e for e in sorted(v, key=lambda e: e["date"])
                          if _parse_dt(e["date"]) >= _cutoff_48h]
                      for k, v in hourly_48h_raw.items()}
        _cutoff_24h = _now_ak - datetime.timedelta(hours=24)
        hourly_24h_stats = {k: [e for e in v if _parse_dt(e["date"]) >= _cutoff_24h]
                            for k, v in hourly_24h.items()}
        print(f"[SNOTEL {triplet}] hourly_24h keys with data: "
              f"{ {k: len([v for v in vals if v['value'] is not None]) for k, vals in hourly_24h.items()} }")
        tobs_stats = hourly_24h_stats.get("TOBS", [])
        if tobs_stats:
            print(f"[SNOTEL {triplet}] TOBS stats range: {tobs_stats[0]['date']} → {tobs_stats[-1]['date']} ({len(tobs_stats)} entries)")
        else:
            print(f"[SNOTEL {triplet}] TOBS stats: EMPTY after 24h cutoff")



    caps = {
        "has_swe":    bool(daily.get("WTEQ")),
        "has_depth":  bool(daily.get("SNWD")),
        "has_precip": bool(daily.get("PREC")),
        "has_temp":   bool(daily.get("TAVG") or daily.get("TMAX")),
        "has_wind":   bool(hourly_wind.get("WSPDV") and hourly_wind.get("WDIRV")),
        "has_hourly_swe":   bool(hourly_24h.get("WTEQ")),
        "has_hourly_depth": bool(hourly_24h.get("SNWD")),
        "has_hourly_temp":  bool(hourly_24h.get("TOBS")),
    }

    # Build HTML table rows from 24h stats slice (most recent first)
    # Use TOBS, SNWD, WTEQ, PREC — all available hourly
    hourly_headers   = ["Date/Time", "Temp (°F)", "Snow Depth (in)", "SWE (in)", "Precip (in)"]
    hourly_elem_keys = ["TOBS", "SNWD", "WTEQ", "PREC"]
    hourly_24h_table = hourly_24h_stats  # alias: table uses 24h only

    # Add wind columns for SnoLite stations that report wind
    has_wind_data = bool(hourly_wind.get("WSPDV") or hourly_wind.get("WSPDX"))
    wind_by_time  = {}
    if has_wind_data:
        hourly_headers   = hourly_headers + ["Wind Spd (mph)", "Wind Gust (mph)", "Wind Dir (°)"]
        wind_elem_keys   = ["WSPDV", "WSPDX", "WDIRV"]
        wind_by_time     = {ek: {v["date"]: v["value"] for v in hourly_wind.get(ek, [])}
                            for ek in wind_elem_keys}

    # Build a combined time index from all available hourly elements
    # For wind stations, include wind timestamps too so rows align
    wind_dates = {v["date"] for ek in (["WSPDV"] if has_wind_data else [])
                  for v in hourly_wind.get(ek, []) if v["value"] is not None}
    all_hourly_times = sorted(
        {v["date"] for ek in hourly_elem_keys for v in hourly_24h_table.get(ek, []) if v["value"] is not None}
        | wind_dates,
        reverse=True  # most recent first
    )
    hourly_by_time = {ek: {v["date"]: v["value"] for v in hourly_24h_table.get(ek, [])}
                      for ek in hourly_elem_keys}

    table_rows = []
    for dt in all_hourly_times:
        try:
            # Format as "Mar 01 14:00" from any AWDB date string
            _ds = dt.strip().replace('T', ' ')
            _d  = datetime.datetime.strptime(_ds[:16], '%Y-%m-%d %H:%M')
            label = _d.strftime('%b %d %H:%M')
        except Exception:
            label = dt
        row = [label]
        for ek in hourly_elem_keys:
            val = hourly_by_time[ek].get(dt)
            row.append("" if val is None else f"{val:.1f}")
        if has_wind_data:
            for ek in wind_elem_keys:
                val = wind_by_time[ek].get(dt)
                row.append("" if val is None else f"{val:.1f}")
        table_rows.append(row)
    headers = hourly_headers

    return {
        "meta":          meta,
        "daily":         daily,
        "hourly_wind":   hourly_wind,
        "hourly_24h":    hourly_24h,       # 48h of data for plotting
        "hourly_24h_stats": hourly_24h_stats,  # 24h slice for summary stats
        "caps":          caps,
        "table_rows":    table_rows,
        "table_headers": headers,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _vals(data_list):
    """Extract non-None float values from [{date, value}, ...] list."""
    return np.array([v["value"] for v in data_list if v["value"] is not None],
                    dtype=float)


def _dates(data_list):
    return [v["date"] for v in data_list if v["value"] is not None]


def compute_snotel_stats(station_data):
    """Return list of (label, value_str) tuples for the stats chips.
    All stats use the 24h slice (hourly_24h_stats); plots use the full 48h hourly_24h."""
    stats = []
    daily          = station_data["daily"]
    hourly_24h     = station_data.get("hourly_24h_stats", station_data.get("hourly_24h", {}))  # 24h only
    meta      = station_data["meta"]

    # Temperature stats from 24h hourly TOBS
    tobs_entries = [v for v in hourly_24h.get("TOBS", []) if v["value"] is not None]
    if tobs_entries:
        t_vals = np.array([e["value"] for e in tobs_entries])
        stats.append(("Max Temp",  f"{t_vals.max():.0f} °F"))
        stats.append(("Min Temp",  f"{t_vals.min():.0f} °F"))
        stats.append(("Avg Temp",  f"{t_vals.mean():.1f} °F"))
    elif daily.get("TMAX") and daily.get("TMIN"):
        # Fallback to daily
        mx = _vals(daily["TMAX"])
        mn = _vals(daily["TMIN"])
        if len(mx):
            stats.append(("Max Temp",  f"{mx[-1]:.0f} °F"))
        if len(mn):
            stats.append(("Min Temp",  f"{mn[-1]:.0f} °F"))

    # 24h snow total from hourly SNWD (max - min in last 24h as proxy, or diff)
    snwd_entries = [v for v in hourly_24h.get("SNWD", []) if v["value"] is not None]
    if len(snwd_entries) >= 2:
        sn_vals = np.array([e["value"] for e in snwd_entries])
        snow_24h = max(sn_vals[-1] - sn_vals[0], 0.0)
        stats.append(("24h Snow",  f"{snow_24h:.1f} in"))

    # 24h SWE change from hourly WTEQ
    swe_entries = [v for v in hourly_24h.get("WTEQ", []) if v["value"] is not None]
    if len(swe_entries) >= 2:
        swe_vals = np.array([e["value"] for e in swe_entries])
        swe_24h = max(swe_vals[-1] - swe_vals[0], 0.0)
        stats.append(("24h SWE",   f"{swe_24h:.2f} in"))

    # Wind stats from hourly_wind — filter to 24h before computing
    hourly_wind = station_data.get("hourly_wind", {})
    _wind_cutoff = local_now() - datetime.timedelta(hours=24)
    def _wind_24h(entries):
        return [v for v in entries if v["value"] is not None
                and _parse_dt(v["date"]) >= _wind_cutoff]
    wspd_entries = _wind_24h(hourly_wind.get("WSPDV", []))
    wgst_entries = _wind_24h(hourly_wind.get("WSPDX", []))
    wdir_entries = _wind_24h(hourly_wind.get("WDIRV", []))
    if wdir_entries:
        import math
        sin_sum = sum(math.sin(math.radians(e["value"])) for e in wdir_entries)
        cos_sum = sum(math.cos(math.radians(e["value"])) for e in wdir_entries)
        avg_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        cardinals = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        card = cardinals[int((avg_deg + 11.25) / 22.5) % 16]
        stats.append(("Wind Dir", card))
    if wspd_entries:
        avg_spd = sum(e["value"] for e in wspd_entries) / len(wspd_entries)
        stats.append(("Avg Wind", f"{avg_spd:.1f} mph"))
    if wgst_entries:
        stats.append(("Max Gust", f"{max(e['value'] for e in wgst_entries):.1f} mph"))

    elev = meta.get("elevation")
    if elev:
        stats.append(("Elevation", f"{int(elev):,} ft"))

    return stats


# ---------------------------------------------------------------------------
# Plot helpers — dark theme shared with Windrose_Plots.py
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='lightgray', labelsize=16)
    ax.spines[:].set_color('#444466')
    ax.yaxis.label.set_color('lightgray')
    ax.xaxis.label.set_color('lightgray')


def _add_minor_hgrid(ax, y_min, y_max, minor_step, major_step=None):
    """Add minor horizontal gridlines at minor_step intervals, major at major_step.
    When major_step==5 and minor_step==1 (temp/snow), also sets tick locators."""
    import math
    lo = math.floor(y_min / minor_step) * minor_step
    hi = math.ceil(y_max / minor_step) * minor_step + minor_step
    vals = np.arange(lo, hi, minor_step)
    for y in vals:
        ax.axhline(y, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
    if major_step:
        lo2 = math.floor(y_min / major_step) * major_step
        hi2 = math.ceil(y_max / major_step) * major_step + major_step
        for y in np.arange(lo2, hi2, major_step):
            ax.axhline(y, color='white', linewidth=1.1, linestyle='-', zorder=1)
        # Snap ticks to major/minor grid intervals for temp & snow (step==5/1)
        if major_step == 5 and minor_step == 1:
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))


def _add_6hr_vlines_dates(ax, dates, x):
    """Add vertical gridlines at 0600, 1200, 1800, 0000 for date-indexed axes."""
    for i, d in enumerate(dates):
        # For daily data, mark Jan 1 / Oct 1 etc — skip for daily
        pass


def _add_6hr_vlines_times(ax, times, x):
    """Add vertical gridlines at 0600, 1200, 1800, 0000 for time-indexed axes."""
    for i, t in enumerate(times):
        hhmm = str(t).strip()[:5]
        if hhmm in ('06:00', '12:00', '18:00', '00:00', '0:00'):
            ax.axvline(x=i, color='#7777bb', linewidth=1.3, linestyle='-',
                       alpha=0.95, zorder=8)


def _hhmm(date_str):
    """Extract HH:MM from any AWDB date format robustly.
    Handles: '2026-02-28T14:00:00', '2026-02-28 14:00:00', '2026-02-28 14:00'
    """
    for sep in ('T', ' '):
        if sep in date_str:
            return date_str.split(sep)[1][:5]
    return date_str[-5:]


def _parse_dt(date_str):
    """Parse any AWDB date string to datetime.
    Hourly timestamps have already been converted to DISPLAY_TZ (naive), so
    we parse them as naive datetimes and compare against local_now() cutoffs."""
    s = date_str.strip().replace('T', ' ')
    if len(s) == 16:          # 'YYYY-MM-DD HH:MM'
        s += ':00'
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def _x_ticks(ax, dates, n=12):
    """Apply sparse, rotated date tick labels."""
    x   = np.arange(len(dates))
    step = max(1, len(x) // n)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(dates[::step], fontsize=13.0,
                       color='lightgray', rotation=35, ha='right')
    ax.set_xlim(-0.5, len(x) - 0.5)


# ---------------------------------------------------------------------------
# Individual plot makers
# ---------------------------------------------------------------------------

def make_swe_plot(ax, daily):
    """Snow water equivalent line chart for the full water year."""
    entries = [v for v in daily.get("WTEQ", []) if v["value"] is not None]
    if not entries:
        ax.text(0.5, 0.5, "No SWE data", transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return
    dates = [e["date"] for e in entries]
    vals  = np.array([e["value"] for e in entries])
    x     = np.arange(len(dates))

    _style_ax(ax)
    _add_minor_hgrid(ax, 0, vals.max() if len(vals) else 1, 0.1, 0.5)
    ax.plot(x, vals, color='#44ccff', linewidth=2, zorder=3)
    ax.fill_between(x, vals, alpha=0.2, color='#44ccff')
    ax.set_title("Snow Water Equivalent (in)", color='white',
                 fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel("SWE (in)", fontsize=18.0)
    _x_ticks(ax, dates)


def make_snow_depth_plot(ax, daily):
    """Snow depth bar chart for the full water year."""
    entries = [v for v in daily.get("SNWD", []) if v["value"] is not None]
    if not entries:
        ax.text(0.5, 0.5, "No snow depth data", transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return
    dates = [e["date"] for e in entries]
    vals  = np.array([e["value"] for e in entries])
    x     = np.arange(len(dates))

    _style_ax(ax)
    _add_minor_hgrid(ax, 0, vals.max() if len(vals) else 1, 1, 5)
    ax.bar(x, vals, color='#7799ff', alpha=0.85, zorder=2, width=1.0)
    ax.plot(x, vals, color='#aabbff', linewidth=1.0, alpha=0.6, zorder=3)
    ax.set_title("Snow Depth (in)", color='white',
                 fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel("Depth (in)", fontsize=18.0)
    _x_ticks(ax, dates)


def make_precip_plot(ax, daily):
    """Accumulated precipitation (water year) line chart."""
    entries = [v for v in daily.get("PREC", []) if v["value"] is not None]
    if not entries:
        ax.text(0.5, 0.5, "No precip data", transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return
    dates = [e["date"] for e in entries]
    vals  = np.array([e["value"] for e in entries])
    x     = np.arange(len(dates))

    _style_ax(ax)
    _add_minor_hgrid(ax, 0, vals.max() if len(vals) else 1, 0.1, 0.5)
    ax.plot(x, vals, color='#55ddaa', linewidth=2, zorder=3)
    ax.fill_between(x, vals, alpha=0.18, color='#55ddaa')
    ax.set_title("Accumulated Precipitation (in)", color='white',
                 fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel("Precip (in)", fontsize=18.0)
    _x_ticks(ax, dates)


def make_temp_plot(ax, daily):
    """Avg/Max/Min temperature ribbon for the water year. Tightly auto-scaled; 32F only if in range."""
    avg_e = [v for v in daily.get("TAVG", []) if v["value"] is not None]
    max_e = [v for v in daily.get("TMAX", []) if v["value"] is not None]
    min_e = [v for v in daily.get("TMIN", []) if v["value"] is not None]

    if not avg_e and not max_e:
        ax.text(0.5, 0.5, "No temperature data", transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return

    base  = avg_e or max_e
    dates = [e["date"] for e in base]
    x     = np.arange(len(dates))

    _style_ax(ax)

    all_vals = [e["value"] for e in avg_e + max_e + min_e if e["value"] is not None]
    y_lo, y_hi, span = 0, 40, 40
    if all_vals:
        v_min, v_max = min(all_vals), max(all_vals)
        span = max(v_max - v_min, 1.0)
        pad  = max(span * 0.12, 5)
        y_lo = v_min - pad
        y_hi = v_max + pad
        _add_minor_hgrid(ax, y_lo, y_hi, 1, 5)

    if max_e and min_e:
        mx_map = {e["date"]: e["value"] for e in max_e}
        mn_map = {e["date"]: e["value"] for e in min_e}
        mx_vals = np.array([mx_map.get(d, np.nan) for d in dates])
        mn_vals = np.array([mn_map.get(d, np.nan) for d in dates])
        ax.fill_between(x, mn_vals, mx_vals, alpha=0.15, color='#ff9944', zorder=2)

    date_set = {e["date"]: e["value"] for e in base}
    avg_vals = np.array([date_set.get(d, np.nan) for d in dates])
    ax.plot(x, avg_vals, color='#ff9944', linewidth=1.8, zorder=3, label='Avg')

    if y_lo <= 32 <= y_hi:
        ax.axhline(32, color='#6688aa', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        ax.text(len(x) - 1, 32 + span * 0.02, '32°F', color='#6688aa', fontsize=14.0, ha='right')

    ax.set_ylim(y_lo, y_hi)
    ax.autoscale(False)
    ax.set_title("Temperature (°F)", color='white', fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel("°F", fontsize=18.0)
def make_wind_bar(ax, times, vals, title, ylim, dirs=None):
    """Single wind bar chart (speed or gust) with tiered y-axis, gridlines, barbs."""
    x      = np.arange(len(times))
    cmap   = matplotlib.colormaps.get_cmap('plasma')
    norm   = np.clip(np.array(vals) / max(ylim, 1), 0, 1)
    colors = [cmap(0.15 + 0.7 * v) for v in norm]

    ax.set_facecolor('#1a1a2e')

    # Major gridlines every 5 mph only
    for y_val in range(0, ylim + 1, 5):
        ax.axhline(y_val, color='white', linewidth=1.1, linestyle='-', zorder=1)

    # Vertical 6-hour gridlines
    _add_6hr_vlines_times(ax, times, x)

    ax.bar(x, vals, color=colors, width=0.8, zorder=2, edgecolor='none')
    ax.plot(x, vals, color='white', linewidth=0.8, alpha=0.4, zorder=3)

    # Wind barbs
    if dirs is not None and len(dirs) > 0:
        step = max(1, len(x) // 30)
        for i in range(0, len(x), step):
            spd = float(vals[i]) if i < len(vals) else 0
            if spd < 1:
                continue
            deg = float(dirs[i]) if i < len(dirs) else 0
            rad = np.deg2rad(deg)
            u = -np.sin(rad) * spd
            v = -np.cos(rad) * spd
            y_pos = min(spd, ylim * 0.95)
            ax.barbs(x[i], y_pos, u, v,
                     length=6, linewidth=0.8,
                     barb_increments=dict(half=5, full=10, flag=50),
                     color='white', alpha=0.75, zorder=5)

    ax.set_title(title, color='white', fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel('mph', color='lightgray', fontsize=18.0)
    ax.tick_params(colors='lightgray', labelsize=16)
    ax.spines[:].set_color('#444466')

    step = max(1, len(times) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(times[::step], fontsize=13.0,
                       color='lightgray', rotation=35, ha='right')
    ax.grid(False)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(0, ylim)

    if len(vals):
        peak_i = int(np.argmax(vals))
        ax.annotate(f'{vals[peak_i]:.0f}',
                    xy=(x[peak_i], vals[peak_i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=14.0, color='white', fontweight='bold')


def tiered_ylim(peak):
    if   peak > 100: return 150
    elif peak > 50:  return 100
    else:            return 50


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

SPEED_BINS = [0, 3, 6, 9, 12, 15]
GUST_BINS  = [0, 5, 10, 15, 20, 25]


def _make_24h_snow_depth(station_name, station_data):
    """24-hour hourly snow depth as a standalone line chart."""
    hourly_24h  = station_data.get("hourly_24h", {})
    snwd_entries = [v for v in hourly_24h.get("SNWD", []) if v["value"] is not None]
    if not snwd_entries:
        return None

    times     = [_hhmm(e["date"]) for e in snwd_entries]
    snwd_vals = np.array([e["value"] for e in snwd_entries])
    x         = np.arange(len(times))
    tick_step = max(1, len(x) // 12)

    v_min = float(snwd_vals.min()); v_max = float(snwd_vals.max())
    span  = max(v_max - v_min, 0.5)
    pad   = max(span * 0.12, 5)
    y_lo  = v_min - pad; y_hi = v_max + pad

    fig, ax = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout='constrained')
    ax.set_facecolor('#1a1a2e'); ax.spines[:].set_color('#444466')
    ax.tick_params(colors='lightgray', labelsize=16)
    _add_minor_hgrid(ax, y_lo, y_hi, 1, 5)
    _add_6hr_vlines_times(ax, times, x)
    ax.fill_between(x, y_lo, snwd_vals, alpha=0.2, color='#7799ff', zorder=2)
    ax.plot(x, snwd_vals, color='#8899ff', linewidth=3.0, zorder=3, label='Snow Depth (in)')
    ax.set_ylim(y_lo, y_hi); ax.autoscale(False)
    ax.set_ylabel('Snow Depth (in)', color='#7799ff', fontsize=18.0)
    ax.yaxis.label.set_color('#7799ff'); ax.tick_params(axis='y', colors='#7799ff')
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(times[::tick_step], fontsize=13.0, color='lightgray', rotation=30, ha='right')
    ax.set_xlim(-0.5, len(x) - 0.5); ax.grid(False)
    ax.legend(loc='upper left', fontsize=14.0, facecolor='#16213e',
              edgecolor='#666688', labelcolor='white')
    fig.suptitle(f'{station_name} — 48-Hour Snow Depth',
                 fontsize=28.0, fontweight='bold', color='white')
    return fig


def _make_24h_swe_precip(station_name, station_data):
    """
    24-hour SWE (left axis, cyan) + cumulative Precip (right axis, green).
    Both use hourly AWDB data on the same 24-hour time axis.
    SWE  = hourly WTEQ  (snow water equivalent, inches, cumulative water year)
    Precip = hourly PREC (accumulated precipitation, inches, cumulative water year)
    """
    hourly_24h  = station_data.get("hourly_24h", {})

    swe_entries  = [v for v in hourly_24h.get("WTEQ", []) if v["value"] is not None]
    prec_entries = [v for v in hourly_24h.get("PREC", []) if v["value"] is not None]

    if not swe_entries and not prec_entries:
        return None

    # Build shared time axis from union of both series timestamps
    all_times = sorted({e["date"] for e in swe_entries + prec_entries})
    swe_map   = {e["date"]: e["value"] for e in swe_entries}
    prec_map  = {e["date"]: e["value"] for e in prec_entries}

    times = [_hhmm(t) for t in all_times]
    x     = np.arange(len(times))
    tick_step = max(1, len(x) // 12)

    swe_vals  = np.array([swe_map.get(t, np.nan)  for t in all_times])
    prec_vals = np.array([prec_map.get(t, np.nan) for t in all_times])

    fig, ax1 = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout="constrained")
    ax1.set_facecolor("#1a1a2e")
    ax1.spines[:].set_color("#444466")
    ax1.tick_params(colors="lightgray", labelsize=16)
    _add_6hr_vlines_times(ax1, times, x)

    ax2 = None

    # ── Left axis: SWE ───────────────────────────────────────────────────────
    swe_ok = not np.all(np.isnan(swe_vals))
    if swe_ok:
        sv_min = float(np.nanmin(swe_vals)); sv_max = float(np.nanmax(swe_vals))
        sv_pad = max((sv_max - sv_min) * 0.12, 1)
        sv_lo  = sv_min - sv_pad;   sv_hi  = sv_max + sv_pad
        _add_minor_hgrid(ax1, sv_lo, sv_hi, 0.1, 0.5)
        safe = np.where(np.isnan(swe_vals), sv_lo, swe_vals)
        ax1.fill_between(x, sv_lo, safe, alpha=0.2, color="#44ccff", zorder=2)
        ax1.plot(x, swe_vals, color="#44ddff", linewidth=3.0, zorder=4, label="SWE (in)")
        ax1.set_ylim(sv_lo, sv_hi); ax1.autoscale(False)
    ax1.set_ylabel("SWE (in)", color="#44ccff", fontsize=18.0)
    ax1.yaxis.label.set_color("#44ccff")
    ax1.tick_params(axis="y", colors="#44ccff")

    # ── Right axis: Precip ───────────────────────────────────────────────────
    prec_ok = not np.all(np.isnan(prec_vals))
    if prec_ok:
        pv_min = float(np.nanmin(prec_vals)); pv_max = float(np.nanmax(prec_vals))
        pv_pad = max((pv_max - pv_min) * 0.12, 1)
        pv_lo  = pv_min - pv_pad;    pv_hi  = pv_max + pv_pad
        ax2 = ax1.twinx()
        ax2.set_facecolor("#1a1a2e"); ax2.spines[:].set_color("#444466")
        _add_minor_hgrid(ax2, pv_lo, pv_hi, 0.1, 0.5)
        safe2 = np.where(np.isnan(prec_vals), pv_lo, prec_vals)
        ax2.fill_between(x, pv_lo, safe2, alpha=0.18, color="#55ddaa", zorder=2)
        ax2.plot(x, prec_vals, color="#44ffcc", linewidth=3.0, zorder=5,
                 label="Precip (in)")
        ax2.set_ylim(pv_lo, pv_hi); ax2.autoscale(False)
        ax2.set_ylabel("Precip (in)", color="#55ddaa", fontsize=18.0)
        ax2.yaxis.label.set_color("#55ddaa")
        ax2.tick_params(colors="#55ddaa", labelsize=16)

    ax1.set_xticks(x[::tick_step])
    ax1.set_xticklabels(times[::tick_step],
                        fontsize=13.0, color="lightgray", rotation=30, ha="right")
    ax1.set_xlim(-0.5, len(x) - 0.5); ax1.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
    if h1 or h2:
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=14.0,
                   facecolor="#16213e", edgecolor="#666688", labelcolor="white")

    suffix = " & Precip" if prec_ok else " (no precip data)"
    fig.suptitle(f"{station_name} — 48-Hour SWE{suffix}",
                 fontsize=28.0, fontweight="bold", color="white")
    return fig
def _make_24h_temp(station_name, station_data):
    """24-hour hourly temperature line chart, tightly auto-scaled. Returns figure or None."""
    hourly_24h = station_data.get("hourly_24h", {})
    t_entries = [v for v in hourly_24h.get("TOBS", []) if v["value"] is not None]
    if not t_entries:
        return None

    times = [_hhmm(e["date"]) for e in t_entries]
    vals  = np.array([e["value"] for e in t_entries])
    x     = np.arange(len(times))
    tick_step = max(1, len(x) // 12)

    t_min = float(vals.min())
    t_max = float(vals.max())
    span  = max(t_max - t_min, 1.0)
    pad   = max(span * 0.12, 5)
    y_lo  = t_min - pad
    y_hi  = t_max + pad

    fig, ax = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout='constrained')
    ax.set_facecolor('#1a1a2e')
    ax.spines[:].set_color('#444466')
    ax.tick_params(colors='lightgray', labelsize=16)

    _add_minor_hgrid(ax, y_lo, y_hi, 1, 5)
    _add_6hr_vlines_times(ax, times, x)

    # fill_between with explicit y_lo baseline to prevent axis expansion
    ax.fill_between(x, y_lo, vals, alpha=0.15, color='#ff9944', zorder=2)
    ax.plot(x, vals, color='#ffbb55', linewidth=3.0, zorder=3)

    # 32°F only if it actually falls in the visible range
    if y_lo <= 32 <= y_hi:
        ax.axhline(32, color='#6688aa', linestyle='--', linewidth=1.0, alpha=0.9, zorder=2)
        ax.text(x[-1] if len(x) else 0, 32 + span * 0.02,
                '32°F', color='#6688aa', fontsize=14.0, ha='right')

    ax.set_ylim(y_lo, y_hi)
    ax.autoscale(False)
    ax.set_ylabel('°F', color='lightgray', fontsize=18.0)
    ax.yaxis.label.set_color('lightgray')
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(times[::tick_step], fontsize=13.0, color='lightgray', rotation=30, ha='right')
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.grid(False)

    fig.suptitle(f'{station_name} — 48-Hour Temperature',
                 fontsize=28.0, fontweight='bold', color='white')
    return fig
def _make_seasonal_swe_depth(station_name, station_data):
    """Overlaid seasonal SWE (line+fill) and snow depth (bars) on dual y-axes."""
    daily = station_data["daily"]
    swe_entries  = [v for v in daily.get("WTEQ", []) if v["value"] is not None]
    snwd_entries = [v for v in daily.get("SNWD", []) if v["value"] is not None]

    if not swe_entries and not snwd_entries:
        return None

    all_dates = sorted({e["date"] for e in swe_entries + snwd_entries})
    swe_map  = {e["date"]: e["value"] for e in swe_entries}
    snwd_map = {e["date"]: e["value"] for e in snwd_entries}

    dates = all_dates
    x = np.arange(len(dates))

    fig, ax1 = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout='constrained')
    ax1.set_facecolor('#1a1a2e')
    ax1.spines[:].set_color('#444466')
    ax1.tick_params(colors='lightgray', labelsize=16)

    swe_vals  = np.array([swe_map.get(d, np.nan)  for d in dates])
    snwd_vals = np.array([snwd_map.get(d, np.nan) for d in dates])

    if not np.all(np.isnan(swe_vals)):
        swe_max = np.nanmax(swe_vals)
        ax1.plot(x, swe_vals, color='#44ccff', linewidth=2, zorder=4, label='SWE (in)')
        ax1.fill_between(x, swe_vals, alpha=0.2, color='#44ccff', zorder=3)
        ax1.set_ylabel('SWE (in)', color='#44ccff', fontsize=18.0)
        ax1.yaxis.label.set_color('#44ccff')
        ax1.tick_params(axis='y', colors='#44ccff')

    if not np.all(np.isnan(snwd_vals)):
        ax2 = ax1.twinx()
        ax2.set_facecolor('#1a1a2e')
        ax2.spines[:].set_color('#444466')
        ax2.bar(x, snwd_vals, color='#7799ff', alpha=0.55, zorder=2, width=1.0, label='Snow Depth (in)')
        ax2.set_ylabel('Snow Depth (in)', color='#7799ff', fontsize=18.0)
        ax2.tick_params(colors='#7799ff', labelsize=16)

    # Monthly vertical gridlines for seasonal chart
    import datetime as _dt
    for _i, _d in enumerate(dates):
        try:
            _parsed = _dt.date.fromisoformat(_d)
            if _parsed.day == 1:  # first of month
                ax1.axvline(x=_i, color='#7777bb', linewidth=1.3,
                            linestyle='-', alpha=0.95, zorder=8)
                ax1.text(_i + 0.5, ax1.get_ylim()[1] * 0.97,
                         _parsed.strftime('%b'), fontsize=13.0,
                         color='#aaaacc', va='top')
        except Exception:
            pass
    _x_ticks(ax1, dates)
    ax1.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    try:
        h2, l2 = ax2.get_legend_handles_labels()
    except NameError:
        h2, l2 = [], []
    ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=14.0,
               facecolor='#16213e', edgecolor='#666688', labelcolor='white')

    fig.suptitle(f'{station_name} — Seasonal Snowpack (SWE & Depth)',
                 fontsize=28.0, fontweight='bold', color='white')
    return fig


def build_snotel_plots(station_name, station_data, triplet):
    """
    Build and return a dict of matplotlib figures for one SNOTEL station.

    Returns:
        {
          'snow':         Figure  — seasonal SWE + depth overlaid
          'precip_temp':  None    — removed per spec
          '24h_swe':      Figure  — 24-hour SWE + depth dual-axis
          '24h_temp':     Figure  — 24-hour temperature
          'wind':         Figure or None
          'stats':        [(label, value), ...]
          'station_url':  str
          'date_range':   str
        }
    """
    daily       = station_data["daily"]
    hourly_wind = station_data["hourly_wind"]
    caps        = station_data["caps"]
    stats       = compute_snotel_stats(station_data)

    all_dates = sorted({v["date"]
                        for vals in daily.values() for v in vals
                        if v["value"] is not None})
    date_range = f"{all_dates[0]}  –  {all_dates[-1]}" if all_dates else "—"

    sitenum    = triplet.split(":")[0]
    station_url = STATION_URL_TEMPLATE.format(sitenum=sitenum)

    cmap   = matplotlib.colormaps.get_cmap(CMAP_NAME)
    n      = len(SPEED_BINS)
    colors = [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]

    result = {
        "snow":         None,
        "precip_temp":  None,   # eliminated per spec
        "24h_depth":    None,
        "24h_swe":      None,
        "24h_temp":     None,
        "wind":         None,
        "stats":        stats,
        "station_url":  station_url,
        "date_range":   date_range,
    }

    # ── Seasonal SWE + depth overlaid ──────────────────────────────────
    result["snow"] = _make_seasonal_swe_depth(station_name, station_data)

    # ── 24-hour snow depth (standalone) ────────────────────────────────
    result["24h_depth"] = _make_24h_snow_depth(station_name, station_data)

    # ── 24-hour SWE + Precip overlaid ──────────────────────────────────
    result["24h_swe"] = _make_24h_swe_precip(station_name, station_data)

    # ── 24-hour temperature ─────────────────────────────────────────────
    result["24h_temp"] = _make_24h_temp(station_name, station_data)

    # ── Wind figure: 2×2 roses + bar charts (SnoLite only) ────────────
    if caps["has_wind"]:
        spd_entries = hourly_wind.get("WSPDV", [])
        gst_entries = hourly_wind.get("WSPDX", [])
        dir_entries = hourly_wind.get("WDIRV", [])

        dates_set = {e["date"] for e in dir_entries if e["value"] is not None}
        spd_map   = {e["date"]: e["value"] for e in spd_entries if e["value"] is not None}
        gst_map   = {e["date"]: e["value"] for e in gst_entries if e["value"] is not None}
        dir_map   = {e["date"]: e["value"] for e in dir_entries if e["value"] is not None}

        common = sorted(d for d in dates_set if d in spd_map and d in dir_map)
        times  = [_hhmm(d) for d in common]

        dirs   = np.array([dir_map[d] for d in common])
        speeds = np.array([spd_map[d] for d in common])
        gusts  = np.array([gst_map.get(d, spd_map[d]) for d in common])

        spd_ylim  = tiered_ylim(speeds.max() if len(speeds) else 0)
        gust_ylim = tiered_ylim(gusts.max()  if len(gusts)  else 0)

        fig  = plt.figure(figsize=(15.0, 10), facecolor=BG)
        gs   = gridspec.GridSpec(2, 2, figure=fig,
                                 hspace=0.45, wspace=0.35,
                                 left=0.06, right=0.97,
                                 top=0.90, bottom=0.08)
        ax_sr = fig.add_subplot(gs[0, 0], projection='polar')
        ax_sb = fig.add_subplot(gs[0, 1])
        ax_gr = fig.add_subplot(gs[1, 0], projection='polar')
        ax_gb = fig.add_subplot(gs[1, 1])

        ax_sr.set_facecolor('#1a1a2e')
        ax_gr.set_facecolor('#1a1a2e')

        make_wind_rose(ax_sr, dirs, speeds, 'Avg Wind Speed', SPEED_BINS, colors)
        make_wind_rose(ax_gr, dirs, gusts,  'Max Gust',       GUST_BINS,  colors)
        make_wind_bar(ax_sb, times, speeds, 'Avg Wind Speed (mph)', spd_ylim, dirs=dirs)
        make_wind_bar(ax_gb, times, gusts,  'Max Gust (mph)',       gust_ylim, dirs=dirs)

        fig.suptitle(f'{station_name} — Wind (Last 7 Days)',
                     fontsize=28.0, fontweight='bold', color='white', y=0.97)
        result["wind"] = fig

    return result