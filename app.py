"""
WCMAC Weather Dashboard - Flask Backend
=======================================
Run locally:
    python app.py
    open http://localhost:5000

Files needed in the same directory:
    app.py
    Windrose_Plots.py
    Snotel_Plots.py
    Synoptic_Plots.py
    dashboard.html
"""

import io
import base64
import os
import sys
import traceback
import gc

import numpy as np

# Matplotlib backend must be set BEFORE pyplot is imported
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 50  # suppress spurious warning; app.py closes all figs
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, Response

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import Windrose_Plots as wr

import importlib
importlib.invalidate_caches()
import Snotel_Plots as sn
import Synoptic_Plots as sy

app = Flask(__name__)

# Lower DPI on Render free tier to stay within 512MB RAM limit
# Set PLOT_DPI env var to override (e.g. 150 locally, 90 on free tier)
_PLOT_DPI = int(os.environ.get('PLOT_DPI', '100'))

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Not found'}), 404

DASHBOARD_TITLE = 'WCMAC Weather Dashboard'

# Stations shown in the Network Summary: the default SNOTEL + RAWS lists
# (i.e. the stations from stations.xlsx). Matched by station name.
SUMMARY_STATION_NAMES = (
    [s['name'] for s in sn.DEFAULT_SNOTEL_STATIONS] +
    [s['name'] for s in sy.DEFAULT_SYNOPTIC_STATIONS]
)


_DASHBOARD_HTML = open(os.path.join(_HERE, 'dashboard.html'), encoding='utf-8').read()


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return Response(_DASHBOARD_HTML, mimetype='text/html')


@app.route('/dashboard_config')
def dashboard_config():
    """Title + default station lists, for the front end."""
    return jsonify({
        'title':            DASHBOARD_TITLE,
        'snotel_stations':  sn.DEFAULT_SNOTEL_STATIONS,
        'weather_stations': sy.DEFAULT_SYNOPTIC_STATIONS,
        'summary_stations': SUMMARY_STATION_NAMES,
    })


@app.route('/synoptic_health')
def synoptic_health():
    """Reachability probe for the RAWS data source. Returns {ok: true/false}.
    Checks IEM (the default RAWS source); kept under the old route name so
    the existing front end keeps working."""
    try:
        import requests as _req
        import datetime as _dth
        st = sy.DEFAULT_SYNOPTIC_STATIONS[0] if sy.DEFAULT_SYNOPTIC_STATIONS else {}
        if st.get('source', 'synoptic') == 'synoptic':
            r = _req.get(
                "https://api.synopticdata.com/v2/stations/latest",
                params={"token": sy.SYNOPTIC_TOKEN, "stid": st.get('stid', ''),
                        "recent": 180},
                timeout=6,
            )
            r.raise_for_status()
            return jsonify({"ok": bool(r.json().get("STATION")), "source": "synoptic"})
        r = _req.get(
            sy.IEM_OBHISTORY,
            params={"station": st.get('stid', 'PNTM8'),
                    "network": st.get('network', 'MT_DCP'),
                    "date": _dth.date.today().isoformat()},
            headers=sy.HEADERS, timeout=8,
        )
        r.raise_for_status()
        return jsonify({"ok": True, "source": "iem"})
    except Exception as exc:
        return jsonify({"ok": False, "reason": str(exc)})


@app.route('/snotel_stations')
def snotel_stations():
    return jsonify(sn.DEFAULT_SNOTEL_STATIONS)


@app.route('/synoptic_stations')
def synoptic_stations():
    """Returns the default WCMAC weather (RAWS) stations."""
    return jsonify(sy.DEFAULT_SYNOPTIC_STATIONS)


@app.route('/synoptic_plot', methods=['POST'])
def synoptic_plot():
    data = request.get_json(force=True)
    name = data.get('name', 'Station')
    stid = data.get('stid', '')
    if not stid:
        return jsonify({'error': 'No STID provided'}), 400
    try:
        # Source (IEM vs Synoptic) comes from the request or the default list
        station_data = sy.fetch_station(stid, source=data.get('source'),
                                        network=data.get('network'), name=name)
        plots        = sy.build_synoptic_plots(name, station_data)

        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=_PLOT_DPI, bbox_inches='tight',
                        facecolor=sy.BG)
            buf.seek(0)
            enc = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return enc

        images = {}
        for key in ('temp', 'wind', 'precip'):
            if plots.get(key) is not None:
                images[key] = fig_to_b64(plots[key])

        records = station_data['records']
        times   = [r['time_label'] for r in records]
        plot_meta = {}
        if station_data['has_temp']:
            temps = [r['temp_air'] for r in records if r['temp_air'] is not None]
            plot_meta['temp'] = {
                'x_label': 'Time', 'x_times': times,
                'x_count': len(times),
                'x_start': times[0]  if times else '',
                'x_end':   times[-1] if times else '',
                'y_label': '°F',
                'y_min': float(min(temps)) if temps else 0.0,
                'y_max': float(max(temps)) if temps else 0.0,
            }
        if station_data['has_wind']:
            winds = [r['wind_gust'] or r['wind_spd'] for r in records
                     if (r['wind_gust'] or r['wind_spd']) is not None]
            plot_meta['wind'] = {
                'x_label': 'Time', 'x_times': times,
                'x_count': len(times),
                'x_start': times[0]  if times else '',
                'x_end':   times[-1] if times else '',
                'y_label': 'mph',
                'y_min': 0.0,
                'y_max': float(max(winds)) if winds else 0.0,
            }

        return jsonify({
            'images':        images,
            'stats':         plots['stats'],
            'date_range':    plots['date_range'],
            'table_headers': station_data['table_headers'],
            'table_rows':    station_data['table_rows'],
            'plot_meta':     plot_meta,
            'meta':          station_data['meta'],
            'records':       [{'dt': r['dt'].isoformat(),
                                'wind_spd':  r['wind_spd'],
                                'wind_gust': r['wind_gust']} for r in station_data['records']],
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/snotel_search')
def snotel_search():
    """Search NRCS AWDB by station name fragment. Query param: q=<text>"""
    q = request.args.get('q', '').strip().lower()
    if not q:
        return jsonify([])
    try:
        try:
            import requests as _req
        except ImportError:
            return jsonify({"error": "pip install requests"}), 500
        r = _req.get(
            f"{sn.AWDB_BASE}/stations",
            params={"networkCds": "SNTL,SNTLT,SNOW", "stateCds": sn.SNOTEL_SEARCH_STATES,
                    "activeOnly": "true"},
            headers=sn.HEADERS, timeout=15,
        )
        r.raise_for_status()
        matches = [
            {"name": s.get("name", ""), "triplet": s.get("stationTriplet", ""),
             "elevation": s.get("elevation"), "has_wind": False}
            for s in r.json() if q in s.get("name", "").lower()
        ]
        return jsonify(matches[:20])
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route('/synoptic_search')
def synoptic_search():
    """Search Synoptic Mesonet by station name or STID. Query param: q=<text>"""
    q     = request.args.get('q', '').strip()
    state = request.args.get('state', '').strip() or None
    if not q:
        return jsonify([])
    try:
        results = sy.search_synoptic_stations(q, state=state, limit=20)
        return jsonify(results)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json(force=True)
    name = data.get('name', 'Station')
    url  = data.get('url', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        records, caps, raw_rows, raw_hdrs = wr.fetch_station_data(url)
        # Trim 24h subset for wind rose; full records for time-series plots
        import datetime as _dwx
        def _parse_wx_dt(r):
            d, t = r.get('date',''), r.get('time','')
            s = f"{d} {t}".strip()
            if '24:00' in s:
                try: return _dwx.datetime.strptime(d, '%m/%d/%Y') + _dwx.timedelta(days=1)
                except: pass
            for fmt in ('%m/%d/%Y %H:%M', '%-m/%-d/%Y %H:%M'):
                try: return _dwx.datetime.strptime(s, fmt)
                except: pass
            return _dwx.datetime.min
        _wx_24h_cutoff = _dwx.datetime.now() - _dwx.timedelta(hours=24)
        records_rose = [r for r in records if _parse_wx_dt(r) >= _wx_24h_cutoff]
        plots = wr.build_plots_separately(name, records, caps, raw_rows, raw_hdrs,
                                          records_rose=records_rose if records_rose else records)

        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=_PLOT_DPI, bbox_inches='tight', facecolor=wr.BG)
            buf.seek(0)
            enc = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return enc

        images = {}
        for key in ('temp', 'wind', 'precip'):
            if plots.get(key) is not None:
                images[key] = fig_to_b64(plots[key])

        # Build axis metadata for hover tooltips
        times_ordered = [f"{r['time']}" for r in reversed(records)]
        plot_meta = {}
        if caps['has_temp']:
            temps = wr.arr(records, 'temp')
            plot_meta['temp'] = {
                'x_label': 'Time', 'x_start': times_ordered[0] if times_ordered else '',
                'x_end': times_ordered[-1] if times_ordered else '',
                'x_count': len(times_ordered), 'x_times': times_ordered,
                'y_label': '°F', 'y_min': float(temps.min()), 'y_max': float(temps.max()),
            }
        if caps['has_wind']:
            speeds = wr.arr(records, 'wind_spd')
            gusts  = wr.arr(records, 'gust')
            plot_meta['wind'] = {
                'x_label': 'Time', 'x_start': times_ordered[0] if times_ordered else '',
                'x_end': times_ordered[-1] if times_ordered else '',
                'x_count': len(times_ordered), 'x_times': times_ordered,
                'y_label': 'mph', 'y_min': 0.0, 'y_max': float(gusts.max()),
            }
        if caps.get('has_snow') or caps.get('has_precip'):
            plot_meta['precip'] = {
                'x_label': 'Time', 'x_start': times_ordered[0] if times_ordered else '',
                'x_end': times_ordered[-1] if times_ordered else '',
                'x_count': len(times_ordered), 'x_times': times_ordered,
                'y_label': 'in', 'y_min': 0.0,
                'y_max': float(wr.arr(records, 'new_snow').max()) if caps.get('has_snow')
                         else float(wr.arr(records, 'precip_1hr').max()),
            }

        table_headers = ['Date', 'Time'] + [' '.join(h.split()) for h in plots['raw_hdrs']]
        return jsonify({
            'images': images, 'stats': plots['stats'],
            'date_range': plots['date_range'],
            'table_headers': table_headers, 'table_rows': plots['raw_rows'],
            'plot_meta': plot_meta,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500
    finally:
        plt.close('all')
        gc.collect()


@app.route('/snotel_plot', methods=['POST'])
def snotel_plot():
    data     = request.get_json(force=True)
    name     = data.get('name',     'Station')
    triplet  = data.get('triplet',  '')
    has_wind = data.get('has_wind', False)
    if not triplet:
        return jsonify({'error': 'No triplet provided'}), 400
    try:
        station_data = sn.fetch_snotel_station(triplet, has_wind=has_wind)
        plots        = sn.build_snotel_plots(name, station_data, triplet)

        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=_PLOT_DPI, bbox_inches='tight', facecolor=sn.BG)
            buf.seek(0)
            enc = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return enc

        images = {}
        for key in ('snow', 'precip_temp', '24h_depth', '24h_swe', '24h_temp', 'wind'):
            if plots.get(key) is not None:
                images[key] = fig_to_b64(plots[key])

        # Build axis metadata for hover tooltips
        daily   = station_data['daily']
        h24     = station_data.get('hourly_24h', {})
        all_dates = sorted({v["date"] for vals in daily.values() for v in vals if v["value"] is not None})
        swe_vals  = sn._vals(daily.get('WTEQ', []))
        depth_vals= sn._vals(daily.get('SNWD', []))
        tobs      = [v['value'] for v in h24.get('TOBS', []) if v['value'] is not None]
        h24_swe   = [v['value'] for v in h24.get('WTEQ', []) if v['value'] is not None]
        h24_snwd  = [v['value'] for v in h24.get('SNWD', []) if v['value'] is not None]
        h24_times = [v['date'][-8:-3] for v in h24.get('TOBS', []) if v['value'] is not None]

        # Gather 24h precip times for metadata
        h24_prcp  = [v['value'] for v in h24.get('PREC', []) if v['value'] is not None]
        h24_depth = [v['value'] for v in h24.get('SNWD', []) if v['value'] is not None]
        h24_depth_times = [v['date'][-8:-3] for v in h24.get('SNWD', []) if v['value'] is not None]

        plot_meta = {
            'snow': {
                'x_label': 'Date', 'x_start': all_dates[0] if all_dates else '',
                'x_end': all_dates[-1] if all_dates else '', 'x_count': len(all_dates),
                'x_times': all_dates,
                'y_label': 'in', 'y_min': 0.0,
                'y_max': float(max(swe_vals.max() if len(swe_vals) else 0,
                                   depth_vals.max() if len(depth_vals) else 0)),
            },
            '24h_depth': {
                'x_label': 'Time',
                'x_start': h24_depth_times[0] if h24_depth_times else '',
                'x_end': h24_depth_times[-1] if h24_depth_times else '',
                'x_count': len(h24_depth_times), 'x_times': h24_depth_times,
                'y_label': 'in', 'y_min': float(min(h24_depth)) if h24_depth else 0.0,
                'y_max': float(max(h24_depth)) if h24_depth else 0.0,
            },
            '24h_swe': {
                'x_label': 'Time',
                'x_start': h24_times[0] if h24_times else '',
                'x_end': h24_times[-1] if h24_times else '',
                'x_count': len(h24_times) or len(h24_swe),
                'x_times': h24_times,
                'y_label': 'in', 'y_min': 0.0,
                'y_max': float(max(max(h24_swe) if h24_swe else 0,
                                   max(h24_prcp) if h24_prcp else 0)),
            },
            '24h_temp': {
                'x_label': 'Time',
                'x_start': h24_times[0] if h24_times else '',
                'x_end': h24_times[-1] if h24_times else '',
                'x_count': len(tobs), 'x_times': h24_times,
                'y_label': '°F',
                'y_min': float(min(tobs)) if tobs else 0.0,
                'y_max': float(max(tobs)) if tobs else 0.0,
            },
        }

        # Serialize hourly_wind for summary overlay (SnoLite stations only)
        # Trim to the most recent 24 entries (hourly_wind is fetched with days_back=7)
        hw_raw = station_data.get('hourly_wind', {})
        import datetime as _dt
        def _parse_dt_hw(d):
            s = d.strip().replace('T', ' ')
            if len(s) == 16: s += ':00'
            return _dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        _hw_cutoff_dt = sn.local_now() - _dt.timedelta(hours=24)
        hourly_wind_json = {
            elem: [{'date': v['date'], 'value': v['value']}
                   for v in sorted(entries, key=lambda e: e['date'])
                   if v['value'] is not None and _parse_dt_hw(v['date']) >= _hw_cutoff_dt]
            for elem, entries in hw_raw.items()
        }

        return jsonify({
            'images':        images,
            'stats':         plots['stats'],
            'date_range':    plots['date_range'],
            'station_url':   plots['station_url'],
            'table_headers': station_data['table_headers'],
            'table_rows':    station_data['table_rows'],
            'plot_meta':     plot_meta,
            'hourly_wind':   hourly_wind_json,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500
    finally:
        plt.close('all')
        gc.collect()


@app.route('/summary_plots', methods=['POST'])
def summary_plots():
    """Build network summary charts from pre-loaded station data."""
    try:
        payload  = request.get_json(force=True)
        stations = payload.get('stations', [])
        if not stations:
            return jsonify({'error': 'No stations provided'}), 400

        BG_COL = '#0f0f1a'
        matplotlib.rcParams['font.family'] = 'DejaVu Sans Mono'
        images = {}

        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=_PLOT_DPI, bbox_inches='tight', facecolor=BG_COL)
            buf.seek(0)
            enc = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return enc

        def style_ax(ax):
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='lightgray', labelsize=18.0)
            ax.spines[:].set_color('#444466')
            ax.yaxis.label.set_color('lightgray')
            ax.xaxis.label.set_color('lightgray')
            ax.grid(axis='y', color='#333355', linestyle='--', linewidth=0.6, zorder=1)

        loaded = [s for s in stations if s.get('data')]

        # Network Summary uses only the WCMAC stations (stations.xlsx list).
        # If none of the posted stations match (e.g. renamed), keep them all.
        _wanted = {n.strip().lower() for n in SUMMARY_STATION_NAMES}
        _matched = [s for s in loaded if str(s.get('name', '')).strip().lower() in _wanted]
        if _matched:
            loaded = _matched

        # ── Items 3 & 6: Two pooled temperature box plots (all stations, auto-scale) ──
        # Collect all hourly temp values from wx table_rows (col index 2 = temp after date/time)
        # and from snotel hourly_24h TOBS or daily TAVG
        all_max_temps = []
        all_min_temps = []

        for s in loaded:
            data = s.get('data', {})
            net  = s.get('net', 'wx')

            if net == 'wx':
                # wx table_rows: [date, time, temp, ...] — col 2 is temp
                rows = data.get('table_rows', [])
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[2]))
                    except (ValueError, TypeError, IndexError):
                        pass
                if vals:
                    all_max_temps.append(max(vals))
                    all_min_temps.append(min(vals))

            elif net == 'snotel':
                # Try stats first for max/min temp
                stats = data.get('stats', [])
                stat_dict = {pair[0]: pair[1] for pair in stats if isinstance(pair, (list, tuple)) and len(pair) == 2}
                # Parse "XX °F" strings
                def parse_f(s):
                    try:
                        return float(str(s).replace('°F','').replace('°','').strip())
                    except:
                        return None
                mx = parse_f(stat_dict.get('Max Temp'))
                mn = parse_f(stat_dict.get('Min Temp'))
                if mx is not None:
                    all_max_temps.append(mx)
                if mn is not None:
                    all_min_temps.append(mn)

            elif net == 'synoptic':
                stats = data.get('stats', [])
                stat_dict = {pair[0]: pair[1] for pair in stats if isinstance(pair, (list, tuple)) and len(pair) == 2}
                def _parse_syn_f(sv):
                    try:
                        return float(str(sv).replace('°F','').replace('°','').strip())
                    except:
                        return None
                mx = _parse_syn_f(stat_dict.get('Max Temp'))
                mn = _parse_syn_f(stat_dict.get('Min Temp'))
                if mx is not None:
                    all_max_temps.append(mx)
                if mn is not None:
                    all_min_temps.append(mn)

        # ── Side-by-side box plots — shared y-axis, compact figure ──────
        if all_max_temps or all_min_temps:
            # Compute shared y range across both datasets
            all_temp_vals = all_max_temps + all_min_temps
            if all_temp_vals:
                shared_min = min(all_temp_vals)
                shared_max = max(all_temp_vals)
                shared_span = max(shared_max - shared_min, 1.0)
                shared_pad  = max(shared_span * 0.15, 2.0)
                shared_lo   = shared_min - shared_pad
                shared_hi   = shared_max + shared_pad
            else:
                shared_lo, shared_hi = -20, 50

            fig, (ax_max, ax_min) = plt.subplots(
                1, 2, figsize=(7.5, 4.8), facecolor=BG_COL, layout='constrained',
                sharey=True)

            def _draw_box(ax, vals, label, color, show_ylabel):
                style_ax(ax)
                if not vals:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', color='gray', fontsize=16.0)
                    ax.set_title(label, color='white', fontsize=18.0, fontweight='bold')
                    return
                ax.boxplot([vals], patch_artist=True, notch=False, widths=0.42,
                           medianprops=dict(color='white', linewidth=2),
                           boxprops=dict(facecolor=color, alpha=0.55, linewidth=1),
                           whiskerprops=dict(color='lightgray', linewidth=1),
                           capprops=dict(color='lightgray', linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor=color,
                                          markeredgecolor='none', alpha=0.7, markersize=3.5))
                jitter = np.random.default_rng(42).uniform(-0.09, 0.09, len(vals))
                ax.scatter([1 + j for j in jitter], vals, color='white', s=14,
                           zorder=5, alpha=0.85, linewidths=0)
                # Minor gridlines every 5°F, major every 10°F
                import math as _math
                _lo5 = _math.floor(shared_lo / 5) * 5
                for _y in range(_lo5, int(shared_hi) + 6, 5):
                    ax.axhline(_y, color='#3a3a60', linewidth=0.55, linestyle='-', zorder=1)
                for _y in range(_math.floor(shared_lo / 10) * 10, int(shared_hi) + 11, 10):
                    ax.axhline(_y, color='#6666aa', linewidth=1.0, linestyle='-', zorder=1)
                if shared_lo <= 32 <= shared_hi:
                    ax.axhline(32, color='#6688aa', linestyle='-', linewidth=1.2, alpha=0.85)
                    if show_ylabel:
                        ax.text(1.45, 32 + shared_span * 0.02, '32°F',
                                color='#6688aa', fontsize=12.0)
                ax.set_xticks([1])
                ax.set_xticklabels([f'n={len(vals)}'], color='lightgray', fontsize=14.0)
                ax.set_title(label, color='white', fontsize=18.0, fontweight='bold', pad=4)
                if show_ylabel:
                    ax.set_ylabel('°F', color='lightgray', fontsize=16.0)

            _draw_box(ax_max, all_max_temps, 'Max Temp', '#ff9944', show_ylabel=True)
            _draw_box(ax_min, all_min_temps, 'Min Temp', '#44aaff', show_ylabel=False)

            # Apply shared y limits after drawing so boxplot doesn't fight us
            for ax in (ax_max, ax_min):
                ax.set_ylim(shared_lo, shared_hi)
                ax.autoscale(False)

            fig.suptitle('Temperature Summary — All Stations (°F)',
                         fontsize=20.0, fontweight='bold', color='white')
            images['temp_max_box'] = fig_to_b64(fig)
            images['temp_min_box'] = None
        # ── Overlaid wind charts — distinct per-station colors ──────────────
        # Assigned in order to whichever stations report wind
        WIND_COLORS = [
            '#00e5ff',  # cyan
            '#ff6f00',  # deep orange
            '#00e676',  # green
            '#ff3d57',  # red
            '#d4a0ff',  # lavender
            '#ffeb3b',  # yellow
            '#4fc3f7',  # sky blue
            '#ff80ab',  # pink
            '#b9f542',  # lime
            '#a0522d',  # brown
            '#00bcd4',  # teal
            '#ffa726',  # amber
        ]

        for wkey, w_label in [('wind_spd', 'Wind Speed'), ('gust', 'Wind Gust')]:
            # Gather per-station (times, values) pairs
            series_list = []
            for s in loaded:
                data = s.get('data', {})
                net  = s.get('net', 'wx')

                if net == 'wx':
                    hdrs = data.get('table_headers', [])
                    rows = data.get('table_rows', [])
                    wind_col = None
                    for i, h in enumerate(hdrs):
                        hl = h.lower()
                        if wkey == 'wind_spd' and ('spd' in hl or 'speed' in hl) and 'gust' not in hl:
                            wind_col = i; break
                        elif wkey == 'gust' and 'gust' in hl:
                            wind_col = i; break
                    if wind_col is None:
                        continue
                    times, dates, vals = [], [], []
                    for row in rows:
                        try:
                            times.append(row[1])            # HH:MM label
                            dates.append(row[0]+' '+row[1]) # 'MM/DD/YYYY HH:MM'
                            vals.append(float(row[wind_col]))
                        except (ValueError, TypeError, IndexError):
                            pass
                    if vals:
                        # raw_rows newest-first; reverse → oldest left, newest right
                        times, dates, vals = times[::-1], dates[::-1], vals[::-1]
                        # Apply true 24h datetime cutoff to match snotel series
                        import datetime as _dtwx
                        def _parse_wx(d):
                            d = d.strip()
                            parts = d.split(' ')
                            dp, tp = parts[0], parts[1] if len(parts) > 1 else '00:00'
                            if tp == '24:00':
                                return _dtwx.datetime.strptime(dp, '%m/%d/%Y') + _dtwx.timedelta(days=1)
                            return _dtwx.datetime.strptime(d, '%m/%d/%Y %H:%M')
                        _wx_cutoff = _dtwx.datetime.now() - _dtwx.timedelta(hours=24)
                        filtered = [(t, d, v) for t, d, v in zip(times, dates, vals)
                                    if _parse_wx(d) >= _wx_cutoff]
                        if filtered:
                            times, dates, vals = zip(*filtered)
                            times, dates, vals = list(times), list(dates), list(vals)
                        else:
                            times, dates, vals = [], [], []
                        if times:
                            series_list.append({'name': s['name'], 'times': times,
                                                'dates': dates, 'vals': vals})

                elif net == 'snotel':
                    # SnoLite hourly wind: WSPDV = avg speed, WSPDX = max gust
                    hw = data.get('hourly_wind', {})
                    elem = 'WSPDV' if wkey == 'wind_spd' else 'WSPDX'
                    entries = [v for v in hw.get(elem, []) if v['value'] is not None]
                    if not entries:
                        continue
                    # Keep only entries within the true past 24 hours
                    import datetime as _dt2
                    def _parse_dt2(d):
                        s = d.strip().replace('T', ' ')
                        if len(s) == 16: s += ':00'
                        return _dt2.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
                    _cutoff_dt2 = sn.local_now() - _dt2.timedelta(hours=24)
                    entries = [e for e in sorted(entries, key=lambda e: e['date'])
                               if _parse_dt2(e['date']) >= _cutoff_dt2]
                    def _hhmm(d):
                        # Handles "2026-02-28T14:00:00", "2026-02-28 14:00:00", "2026-02-28 14:00"
                        for sep in ('T', ' '):
                            if sep in d:
                                t = d.split(sep)[1][:5]
                                return t
                        return d[-5:]
                    times = [_hhmm(e['date']) for e in entries]
                    dates = [e['date'] for e in entries]
                    vals  = [float(e['value']) for e in entries]
                    series_list.append({'name': s['name'], 'times': times,
                                        'dates': dates, 'vals': vals})

                elif net == 'synoptic':
                    # Synoptic records list: each record has dt, wind_spd, wind_gust
                    records_syn = data.get('records', [])
                    import datetime as _dt3
                    _cutoff_syn = _dt3.datetime.now(_dt3.timezone.utc) - _dt3.timedelta(hours=24)
                    times, dates, vals = [], [], []
                    for rec in sorted(records_syn, key=lambda r: r['dt']):
                        dt = rec['dt']
                        # dt may be a datetime object or ISO string
                        if isinstance(dt, str):
                            dt = _dt3.datetime.fromisoformat(dt.replace('Z',''))
                        if dt < _cutoff_syn:
                            continue
                        val = rec.get('wind_spd') if wkey == 'wind_spd' else rec.get('wind_gust')
                        if val is None:
                            continue
                        times.append(dt.strftime('%H:%M'))
                        dates.append(dt.strftime('%Y-%m-%d %H:%M'))
                        vals.append(float(val))
                    if vals:
                        series_list.append({'name': s['name'], 'times': times,
                                            'dates': dates, 'vals': vals})

            # ── Grouped / side-by-side bars — one cluster per time slot ──────
            # Build union axis from full datetimes to avoid HH:MM collisions
            # across day boundaries, then label with HH:MM only
            import datetime as _dtaxis
            def _norm_date(d):
                """Normalize any date string to 'YYYY-MM-DD HH:MM' for consistent sorting.
                Handles: 'M/D/YYYY HH:MM' (wx, with possible 24:00 midnight),
                         'YYYY-MM-DD HH:MM', 'YYYY-MM-DDTHH:MM:SS' (AWDB)
                """
                d = d.strip()
                if '/' in d:
                    # wx format: 'M/D/YYYY HH:MM' — no leading zeros, and 24:00 means midnight
                    parts = d.split(' ')
                    date_part, time_part = parts[0], parts[1] if len(parts) > 1 else '00:00'
                    if time_part == '24:00':
                        # 24:00 = midnight = start of next day
                        try:
                            dt = _dtaxis.datetime.strptime(date_part, '%m/%d/%Y') + _dtaxis.timedelta(days=1)
                            return dt.strftime('%Y-%m-%d 00:00')
                        except Exception:
                            pass
                    for fmt in ('%m/%d/%Y', '%-m/%-d/%Y', '%m/%d/%Y'):
                        try: return _dtaxis.datetime.strptime(d, fmt+' %H:%M').strftime('%Y-%m-%d %H:%M')
                        except Exception: pass
                return d.replace('T', ' ')[:16]  # AWDB/ISO → 'YYYY-MM-DD HH:MM'

            # Build time axis from the INTERSECTION of station timestamps
            # (truncated to the hour) so all stations share the same x slots.
            # This keeps bars wide regardless of how many stations are loaded.
            from collections import defaultdict, Counter
            def _hour_key(d):
                nd = _norm_date(d)
                return nd[:13] + ':00'  # 'YYYY-MM-DD HH:00'

            n_series = len(series_list)
            if n_series == 0:
                continue

            # Count how many series have data at each hour slot
            hour_counts = Counter()
            for ser in series_list:
                seen_hrs = set()
                for d in ser.get('dates', ser['times']):
                    h = _hour_key(d)
                    if h not in seen_hrs:
                        seen_hrs.add(h)
                        hour_counts[h] += 1

            # Keep hours present in at least half the series (loose intersection)
            threshold = max(1, n_series // 2)
            all_iso_union = sorted(h for h, cnt in hour_counts.items() if cnt >= threshold)
            all_iso_union = all_iso_union[-24:]  # cap at 24 most recent hours
            all_times_union = [d[11:16] for d in all_iso_union]  # HH:MM labels
            n_times  = len(all_iso_union)
            date_idx = {nd: i for i, nd in enumerate(all_iso_union)}

            bar_w   = 0.8 / n_series
            offsets = [(i - (n_series - 1) / 2) * bar_w for i in range(n_series)]

            fig, ax = plt.subplots(figsize=(20, 9), facecolor=BG_COL, layout='constrained')
            ax.set_facecolor('#1a1a2e')
            ax.spines[:].set_color('#444466')
            ax.tick_params(colors='lightgray', labelsize=18.0)

            all_vals = []
            for idx, ser in enumerate(series_list):
                color = WIND_COLORS[idx % len(WIND_COLORS)]
                xs_list, ys_list = [], []
                # Aggregate each series to hourly max/avg before plotting
                hour_vals = defaultdict(list)
                for d, v in zip(ser.get('dates', ser['times']), ser['vals']):
                    h = _hour_key(d)
                    if h in date_idx and v is not None:
                        hour_vals[h].append(v)
                for h, vals in hour_vals.items():
                    agg = max(vals) if wkey == 'gust' else sum(vals)/len(vals)
                    xs_list.append(date_idx[h] + offsets[idx])
                    ys_list.append(agg)
                if not xs_list:
                    continue
                xs = np.array(xs_list); ys = np.array(ys_list)
                ax.bar(xs, ys, width=bar_w * 0.92, color=color, alpha=0.85,
                       label=ser['name'], edgecolor='none', zorder=3)
                all_vals.extend(ys_list)

            # Auto-scale y
            if all_vals:
                y_max = max(all_vals)
                if   y_max > 100: ylim = 150
                elif y_max > 50:  ylim = 100
                else:             ylim = max(50, int(y_max * 1.25) + 5)
                ax.set_ylim(0, ylim)
                for y in range(0, ylim + 1, 1):
                    ax.axhline(y, color='#383858', linewidth=0.5, linestyle='-', zorder=1)
                for y in range(0, ylim + 1, 5):
                    ax.axhline(y, color='#6666aa', linewidth=1.0, linestyle='-', zorder=1)

            tick_step = max(1, n_times // 12)
            ax.set_xticks(np.arange(n_times)[::tick_step])
            ax.set_xticklabels(all_times_union[::tick_step],
                               fontsize=16.0, color='lightgray', rotation=30, ha='right')
            ax.set_xlim(-0.5, n_times - 0.5)

            # 6-hour vertical gridlines
            _6hr = {'06:00', '12:00', '18:00', '00:00', '0:00', '6:00', '24:00'}
            for _i, _t in enumerate(all_times_union):  # all_times_union is already HH:MM
                if _t in _6hr:
                    ax.axvline(x=_i, color='#7777bb', linewidth=1.3,
                               linestyle='-', alpha=0.95, zorder=8)
            ax.set_ylabel('mph', color='lightgray', fontsize=18.0)
            ax.grid(False)
            ax.legend(loc='upper left', fontsize=16.0, facecolor='#16213e',
                      edgecolor='#666688', labelcolor='white',
                      ncol=min(n_series, 5))

            img_key = 'wind_spd_stack' if wkey == 'wind_spd' else 'wind_gust_stack'
            fig.suptitle(f'{w_label} — All Stations', fontsize=20.0,
                         fontweight='bold', color='white')
            images[img_key] = fig_to_b64(fig)

            # Record color per station name (only need to do this once)
            if 'series_colors' not in images:
                images['series_colors'] = {
                    ser['name']: WIND_COLORS[i % len(WIND_COLORS)]
                    for i, ser in enumerate(series_list)
                }

        return jsonify({'images': images, 'series_colors': images.pop('series_colors', {})})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500



if __name__ == '__main__':
    print()
    print('  +------------------------------------------+')
    print('  |  WCMAC Weather Dashboard                 |')
    print('  |  http://localhost:5000                   |')
    print('  +------------------------------------------+')
    print()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)