import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import requests

# =============================================================================
# CONFIGURATION — Edit this section to customize the output
# =============================================================================

STATIONS = [
    {"name": "Mile 43", "url": "https://wxstns.net/wxstns/aknet/MILE_43.html"},
    {"name": "Aly Mid",  "url": "https://wxstns.net/wxstns/aknet/ALY_MID_HR.html"},
    {"name": "Max's",    "url": "https://wxstns.net/wxstns/aknet/MAX'S_MTN.html"},
    {"name": "Aly Top",  "url": "https://wxstns.net/wxstns/aknet/ALY_TOP_HR.html"},
]

# Bin edges for wind speed / gust roses (mph)
SPEED_BINS = [0, 3, 6, 9, 12, 15]
GUST_BINS  = [0, 5, 10, 15, 20, 25]

BG        = '#0f0f1a'     # figure background color
CMAP_NAME = 'plasma'      # colormap for wind rose bars

# Font: change this one line to restyle all text in the figure.
# On Windows try: 'Consolas', 'Calibri', 'Segoe UI', 'Arial'
# For hacker vibe: 'Consolas', 'Courier New', 'Liberation Mono'
FONT_FAMILY = 'DejaVu Sans Mono'
matplotlib.rcParams['font.family'] = FONT_FAMILY

# =============================================================================
# HTML FETCH
# =============================================================================

def fetch_html(url):
    """Download the page, spoofing a browser User-Agent to avoid 406 errors."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.text


# =============================================================================
# COLUMN DETECTION
# The core of the flexible parsing. Instead of hard-coding column positions,
# we read the page's own header lines and map each column by name.
# =============================================================================

# Maps lowercase keywords found in the page header to standardised field names.
# Order matters — more specific matches should come first.
COLUMN_KEYWORD_MAP = [
    # keyword          field name
    ('new',            'new_snow'),   # "New Snow" before generic "Snow"
    ('depth',          'snow_depth'),
    ('snow',           'snow_depth'), # fallback if "depth" not present
    ('gust',           'gust'),
    ('dir',            'wind_dir'),
    ('spd',            'wind_spd'),   # catches "Wind Spd", "Spd"
    ('mph',            'wind_spd'),   # some pages use MPH as column header
    ('since',          'precip_0600'),
    ('inch',           'precip_1hr'),
    ('h2o',            'precip_1hr'), # generic water column
    ('precip',         'precip_1hr'),
    ('hum',            'humidity'),
    ('rel',            'humidity'),
    ('batt',           'battery'),
    ('volt',           'battery'),
    ('temp',           'temp'),
    ('air',            'temp'),       # some pages say "Air Temp"
]

def parse_headers(html):
    """
    Read the page's own column header lines and return two things:
      col_map   : ordered list of (col_index, field_name) for every data column
                  after date and time (index 0 and 1 are always date/time)
      raw_hdrs  : list of full multi-line header strings for the data table,
                  e.g. "Wind\nSpd\nMPH" instead of just "Wind"

    Strategy:
      The wxstns pages use FIXED-WIDTH columns. Each data column occupies a
      consistent character-position range across all header lines and data rows.
      We exploit this by:
        1. Finding the DATE/TIME header line -- it anchors the column positions.
        2. Collecting all subsequent header lines (before the first data row).
        3. Using regex to find each token's character-span in the first header line.
        4. For each subsequent header line, checking which characters fall within
           that same span and collecting any words found there.
        5. Stacking all words found for each column position into a single label
           (joined with newline) -- e.g. "Wind / Spd / MPH".
        6. Using the full stacked label for keyword matching (more context = better
           field detection) AND for display in the data table.
    """
    lines = html.split('\n')
    hdr_lines = []
    in_block  = False

    for line in lines:
        if 'DATE' in line and 'TIME' in line:
            in_block = True
            hdr_lines.append(line)
            continue
        if in_block:
            stripped = line.strip()
            if re.match(r'\d+/\d+/\d{4}', stripped):
                break   # first data row -- stop collecting headers
            if stripped:
                hdr_lines.append(line)

    if not hdr_lines:
        return [], []

    first_line = hdr_lines[0]

    # Find character spans of every token in the first (DATE/TIME) header line.
    # We'll use these spans to locate matching words in subsequent header lines.
    token_spans = [(m.start(), m.end(), m.group())
                   for m in re.finditer(r'\S+', first_line)]

    # Identify and skip the DATE and TIME spans -- data columns come after.
    data_spans = []
    skip = {'DATE', 'TIME'}
    for start, end, tok in token_spans:
        if tok in skip:
            continue
        data_spans.append((start, end, tok))

    if not data_spans:
        return [], []

    # For each data column, collect words from ALL header lines that fall
    # within (or overlap) that column's character span.
    # We expand the span slightly (+-2 chars) to catch tokens that are
    # slightly offset due to formatting quirks.
    col_words = [[] for _ in data_spans]

    for line in hdr_lines:
        for col_i, (start, end, _) in enumerate(data_spans):
            lo = max(0, start - 2)
            hi = end + 2
            segment = line[lo:hi] if lo < len(line) else ''
            words = segment.split()
            for w in words:
                if w and w not in col_words[col_i]:
                    col_words[col_i].append(w)

    # Build raw_hdrs: the full stacked label for each column, shown in the table.
    # e.g. ['Air\nTemp\nF', 'Wind\nSpd\nMPH', 'Wind\nDir\nAVG', ...]
    raw_hdrs = ['\n'.join(words) for words in col_words]

    # Build col_map: match each column's full label to a standardised field name.
    col_map = []
    used_fields = set()
    for i, words in enumerate(col_words):
        combined = ' '.join(words).lower()
        matched = None
        for keyword, field in COLUMN_KEYWORD_MAP:
            if keyword in combined and field not in used_fields:
                matched = field
                used_fields.add(field)
                break
        col_map.append((i + 2, matched))  # +2 because col 0=date, 1=time

    return col_map, raw_hdrs


def parse_data_rows(html, col_map):
    """
    Parse every data row into a list of record dicts using col_map to name fields.

    Each data row is split into tokens. We always grab date (index 0) and time
    (index 1), then use col_map to extract and name the remaining columns.
    Unknown columns (field=None) are stored as 'col_N' so nothing is lost.

    Returns list of dicts like:
        {'date': '2/28/2026', 'time': '09:00', 'temp': -8, 'wind_spd': 2, ...}
    Values are cast to int or float where possible, otherwise kept as strings.
    """
    records = []
    for line in html.split('\n'):
        stripped = line.strip()
        if not re.match(r'\d+/\d+/\d{4}', stripped):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue

        rec = {'date': tokens[0], 'time': tokens[1]}
        for col_idx, field_name in col_map:
            if col_idx >= len(tokens):
                continue
            raw_val = tokens[col_idx]
            # Cast to numeric where possible
            try:
                val = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = raw_val
            name = field_name if field_name else f'col_{col_idx}'
            rec[name] = val

        records.append(rec)
    return records


def parse_raw_rows(html):
    """
    Return every data row as a list of raw string tokens (unchanged).
    Used for the 'Hourly Data' display table — shows everything on the page.
    """
    rows = []
    for line in html.split('\n'):
        stripped = line.strip()
        if re.match(r'\d+/\d+/\d{4}', stripped):
            tokens = stripped.split()
            if len(tokens) >= 3:
                rows.append(tokens)
    return rows


def detect_capabilities(records):
    """
    Inspect the parsed records to see which data types are available.
    Returns a dict of booleans:
        has_wind   : True if wind_spd, wind_dir, and gust are all present
        has_precip : True if precip_1hr or precip_0600 is present
        has_snow   : True if new_snow or snow_depth is present
        has_temp   : True if temp is present (almost always True)
    """
    if not records:
        return {'has_wind': False, 'has_precip': False,
                'has_snow': False,  'has_temp': False}
    sample = records[0]
    return {
        'has_wind':   all(f in sample for f in ('wind_spd', 'wind_dir', 'gust')),
        'has_precip': any(f in sample for f in ('precip_1hr', 'precip_0600')),
        'has_snow':   any(f in sample for f in ('new_snow', 'snow_depth')),
        'has_temp':   'temp' in sample,
    }


def fetch_station_data(url):
    """
    Main entry point. Fetches HTML, parses headers + data rows, detects
    what data types are available.

    Returns:
        records      : list of record dicts with named, typed fields
        caps         : dict of capability booleans from detect_capabilities()
        raw_rows     : list of raw string token lists (for the data table)
        raw_hdrs     : list of raw header label strings (for the data table)
    """
    html = fetch_html(url)
    col_map, raw_hdrs = parse_headers(html)
    records = parse_data_rows(html, col_map)
    raw_rows = parse_raw_rows(html)
    caps = detect_capabilities(records)

    if not records:
        raise ValueError(f"No data parsed from {url} — check the page format.")

    return records, caps, raw_rows, raw_hdrs


# =============================================================================
# STATISTICS
# =============================================================================

def arr(records, field, default=0.0):
    """Helper: extract a field from all records as a float numpy array."""
    return np.array([float(r.get(field, default)) for r in records])


def compute_stats(records, caps):
    """
    Compute summary statistics for whatever data is available.
    Returns a list of (label, value) tuples and a date_range string.
    The list is built conditionally — only adds rows for fields that exist.
    """
    stats = []
    date_range = f"{records[-1]['date']}  –  {records[0]['date']}"

    if caps['has_temp']:
        temps = arr(records, 'temp')
        stats += [
            ("Max Temp",    f"{int(temps.max())}°F"),
            ("Min Temp",    f"{int(temps.min())}°F"),
            ("Avg Temp",    f"{temps.mean():.1f}°F"),
        ]

    if caps['has_wind']:
        dirs   = arr(records, 'wind_dir')
        speeds = arr(records, 'wind_spd')
        gusts  = arr(records, 'gust')
        cardinals = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                     'S','SSW','SW','WSW','W','WNW','NW','NNW']
        # Circular mean — correct for wrap-around at 0°/360°
        sin_sum = float(np.sum(np.sin(np.deg2rad(dirs))))
        cos_sum = float(np.sum(np.cos(np.deg2rad(dirs))))
        avg_deg = float(np.degrees(np.arctan2(sin_sum, cos_sum))) % 360
        primary_dir = cardinals[int((avg_deg + 11.25) / 22.5) % 16]
        stats += [
            ("Primary Dir",  primary_dir),
            ("Avg Speed",    f"{speeds.mean():.1f} mph"),
            ("Max Gust",     f"{int(gusts.max())} mph"),
        ]

    if caps['has_snow']:
        snow_depth = arr(records, 'snow_depth')
        stats.append(("Snow Depth", f"{snow_depth.max():.2f} in"))
        if records and 'precip_0600' in records[0]:
            h2o = arr(records, 'precip_0600')
            stats.append(("H₂O/0600", f"{h2o.max():.2f} in"))
        elif records and 'new_snow' in records[0]:
            new_snow = arr(records, 'new_snow')
            stats.append(("New Snow", f"{new_snow.sum():.1f} in"))

    if caps['has_precip']:
        precip = arr(records, 'precip_1hr')
        stats.append(("Total Precip",  f"{precip.sum():.2f} in"))

    return stats, date_range


# =============================================================================
# PLOT PANELS
# =============================================================================

def make_wind_rose(ax, directions, magnitudes, title, speed_bins, colors):
    """
    Draw a wind rose on a polar axes.
    Compass divided into 16 bins (22.5° each), stacked by speed band.
    Directions are meteorological degrees (0=N, 90=E, CW).
    """
    # Ensure numpy arrays regardless of whether lists or arrays were passed in
    directions = np.asarray(directions, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    n_dir = 16
    bw = 360.0 / n_dir  # 22.5° per bin

    # Configure axes FIRST so bar theta values are interpreted correctly
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # clockwise, matching met convention

    # Bin met degrees into 16 N-clockwise slots.
    # Offset by half a bin so 0° (N) centres on bin 0.
    bin_idx = np.floor(((directions + bw / 2) % 360) / bw).astype(int) % n_dir
    bar_w = np.deg2rad(bw) * 0.9

    for i in range(n_dir):
        # With N-up/CW axes, met degrees map directly to theta
        theta = np.deg2rad(i * bw)
        mags = magnitudes[bin_idx == i]
        if len(mags) == 0:
            continue
        bottom = 0
        for j in range(len(speed_bins) - 1):
            count = int(np.sum((mags >= speed_bins[j]) & (mags < speed_bins[j+1])))
            if count:
                ax.bar(theta, count, width=bar_w, bottom=bottom,
                       color=colors[j], edgecolor='#1a1a2e', linewidth=0.5)
                bottom += count
        count = int(np.sum(mags >= speed_bins[-1]))
        if count:
            ax.bar(theta, count, width=bar_w, bottom=bottom,
                   color=colors[-1], edgecolor='#1a1a2e', linewidth=0.5)

    ax.set_thetagrids(np.arange(0, 360, 45),
                      ['N','NE','E','SE','S','SW','W','NW'],
                      color='white', fontsize=13.0, fontweight='bold')
    ax.set_title(title, fontsize=14.0, fontweight='bold', pad=14, color='white')
    ax.set_facecolor('#1a1a2e')
    ax.spines['polar'].set_color('#666688')
    ax.grid(color='#555577', linestyle='--', linewidth=0.5)
    for lbl in ax.get_yticklabels():
        lbl.set_color('lightgray'); lbl.set_fontsize(10.0)

    # Return legend info for external placement (caller draws legend in a side axes)
    leg_labels = [f'{speed_bins[j]}–{speed_bins[j+1]} mph'
                  for j in range(len(speed_bins)-1)]
    leg_labels.append(f'≥{speed_bins[-1]} mph')
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors]
    return handles, leg_labels


def draw_rose_legend(ax, handles, leg_labels, speed_bins):
    """Draw a wind speed legend in a plain axes (no ticks, no spines)."""
    ax.set_axis_off()
    ax.legend(handles, leg_labels,
              loc='center', fontsize=11.0,
              title='Speed', title_fontsize=12.0,
              facecolor='#16213e', edgecolor='#666688', labelcolor='white',
              framealpha=1.0).get_title().set_color('white')


def _add_6hr_vlines(ax, times, x):
    """Add vertical gridlines at 0600, 1200, 1800, 0000 (2400) positions."""
    for i, t in enumerate(times):
        hhmm = t.strip()[:5]
        if hhmm in ('06:00', '12:00', '18:00', '00:00', '0:00', '24:00'):
            ax.axvline(x=i, color='#7777bb', linewidth=1.3, linestyle='-',
                       alpha=0.95, zorder=8)


def make_temp_plot(ax, records):
    """
    Hourly temperature line chart, oldest→newest left to right.
    Auto-scales tightly to data range. 32°F shown only if within range.
    """
    times = [f"{r['time']}" for r in reversed(records)]
    temps = arr(records, 'temp')[::-1]
    x = np.arange(len(times))
    tick_step = max(1, len(x) // 10)

    t_min = float(temps.min())
    t_max = float(temps.max())
    span  = max(t_max - t_min, 1.0)
    pad   = max(span * 0.12, 5)
    y_lo  = t_min - pad
    y_hi  = t_max + pad

    ax.set_facecolor('#1a1a2e')

    # Minor horizontal gridlines at 1°F intervals (only within visible range)
    for y_val in range(int(np.floor(y_lo)), int(np.ceil(y_hi)) + 1):
        ax.axhline(y_val, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
    # Major gridlines every 5°F
    base5 = int(np.floor(y_lo / 5)) * 5
    for y_val in range(base5, int(np.ceil(y_hi)) + 5, 5):
        ax.axhline(y_val, color='white', linewidth=1.1, linestyle='-', zorder=1)

    _add_6hr_vlines(ax, times, x)

    # fill_between with explicit baseline = y_lo to avoid axis expansion
    ax.fill_between(x, y_lo, temps, alpha=0.15, color='#ff9944', zorder=2)
    ax.plot(x, temps, color='#ffaa55', linewidth=2.5, zorder=3)

    # 32°F line only if it actually falls inside the visible range
    if y_lo <= 32 <= y_hi:
        ax.axhline(32, color='#6688aa', linestyle='--', linewidth=1.0, alpha=0.9, zorder=2)
        ax.text(x[-1], 32 + span * 0.02, '32°F', color='#6688aa', fontsize=14.0, ha='right')

    # Set ylim LAST so nothing can expand it afterward
    ax.set_ylim(y_lo, y_hi)
    ax.set_title('Temperature (°F)', color='white', fontsize=22.0, fontweight='bold', pad=8)
    ax.set_ylabel('°F', color='lightgray', fontsize=18.0)
    ax.tick_params(colors='lightgray', labelsize=16.0)
    ax.spines[:].set_color('#444466')
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(times[::tick_step], fontsize=13.0, color='lightgray', rotation=30, ha='right')
    ax.autoscale(False)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(False)


def make_precip_snow_plot(ax, records, caps):
    """
    Combined precip / snow plot on a single axes.
    - If snow data exists: H2O since 0600 as line (left axis) + snow_depth as line (right axis)
    - If precip data exists (no snow): precip_1hr as bars
    - Both axes share the same x (time) axis
    """
    times = [r['time'] for r in reversed(records)]
    x = np.arange(len(times))
    tick_step = max(1, len(x) // 10)

    ax.set_facecolor('#1a1a2e')
    ax.spines[:].set_color('#444466')
    ax.tick_params(colors='lightgray', labelsize=16.0)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(times[::tick_step], fontsize=13.0, color='lightgray', rotation=30, ha='right')

    # Vertical 6-hour gridlines
    _add_6hr_vlines(ax, times, x)

    title_parts = []

    if caps['has_snow']:
        snow_depth  = arr(records, 'snow_depth')[::-1]
        # H2O since 0600 — use precip_0600 if available, fall back to precip_1hr
        has_h2o = 'precip_0600' in records[0]
        h2o_field = 'precip_0600' if has_h2o else 'precip_1hr'
        h2o_label = 'H₂O Since 0600 (in)' if has_h2o else 'Precip 1hr (in)'
        h2o        = arr(records, h2o_field)[::-1]

        # Left axis: H2O since 0600 as line with fill
        h2o_max = float(max(h2o.max(), 0.01))
        h2o_pad = max(h2o_max * 0.15, 1)
        h2o_lo  = 0.0; h2o_hi = h2o_max + h2o_pad
        for y_val in np.arange(0, h2o_hi + 0.1, 0.1):
            ax.axhline(y_val, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
        for y_val in np.arange(0, h2o_hi + 0.5, 0.5):
            ax.axhline(y_val, color='white', linewidth=1.1, linestyle='-', zorder=1)
        ax.fill_between(x, 0, h2o, alpha=0.20, color='#5599ff', zorder=2)
        ax.plot(x, h2o, color='#5599ff', linewidth=2.5, zorder=4,
                label=h2o_label)
        ax.set_ylim(h2o_lo, h2o_hi); ax.autoscale(False)
        ax.set_ylabel(h2o_label, color='#5599ff', fontsize=18.0)
        ax.yaxis.label.set_color('#5599ff')
        title_parts.append('H₂O')

        # Right axis: snow depth as line with fill
        ax2 = ax.twinx()
        ax2.set_facecolor('#1a1a2e'); ax2.spines[:].set_color('#444466')
        sd_min = float(snow_depth.min()); sd_max = float(snow_depth.max())
        sd_pad = max((sd_max - sd_min) * 0.12, 5)
        sd_lo  = max(sd_min - sd_pad, 0); sd_hi = sd_max + sd_pad
        ax2.fill_between(x, sd_lo, snow_depth, alpha=0.15, color='#44ddaa', zorder=2)
        ax2.plot(x, snow_depth, color='#44ddaa', linewidth=2.5,
                 zorder=5, label='Snow Depth (in)')
        ax2.set_ylim(sd_lo, sd_hi); ax2.autoscale(False)
        ax2.set_ylabel('Snow Depth (in)', color='#44ddaa', fontsize=18.0)
        ax2.tick_params(colors='#44ddaa', labelsize=14.0)
        title_parts.append('Depth')

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper left', fontsize=14.0,
                  facecolor='#16213e', edgecolor='#666688', labelcolor='white')

    elif caps['has_precip']:
        precip = arr(records, 'precip_1hr')[::-1]
        p_max = float(max(precip.max(), 0.01))
        p_pad = max(p_max * 0.15, 1)
        p_lo  = 0.0; p_hi = p_max + p_pad
        for y_val in np.arange(0, p_hi + 0.1, 0.1):
            ax.axhline(y_val, color='#bbbbcc', linewidth=0.6, linestyle='--', zorder=1)
        for y_val in np.arange(0, p_hi + 0.5, 0.5):
            ax.axhline(y_val, color='white', linewidth=1.1, linestyle='-', zorder=1)
        ax.fill_between(x, 0, precip, alpha=0.20, color='#5599ff', zorder=2)
        ax.plot(x, precip, color='#5599ff', linewidth=2.5, zorder=3,
                label='Precip 1hr (in)')
        ax.set_ylim(p_lo, p_hi); ax.autoscale(False)
        ax.set_ylabel('Precip (in)', color='#5599ff', fontsize=18.0)
        ax.yaxis.label.set_color('#5599ff')
        ax.legend(loc='upper left', fontsize=14.0,
                  facecolor='#16213e', edgecolor='#666688', labelcolor='white')
        title_parts.append('Precip')
    ax.set_title(' & '.join(title_parts) if title_parts else 'Snow', color='white', fontsize=22.0,
                 fontweight='bold', pad=8)


def _draw_wind_barbs(ax, x, vals, dirs_or_none, ylim, bar_height_frac=0.85):
    """
    Draw wind barbs on a bar chart.  Barbs are centred on each bar at the bar top.
    dirs_or_none: array of degrees (met convention, 0=N, 90=E, CW) or None to skip.
    """
    if dirs_or_none is None or len(dirs_or_none) == 0:
        return
    # Only draw every Nth barb to avoid crowding
    n = len(x)
    step = max(1, n // 30)
    for i in range(0, n, step):
        spd = float(vals[i])
        if spd < 1:
            continue
        deg = float(dirs_or_none[i])
        # Convert met deg → U/V for barbs (barbs show where wind COMES FROM)
        rad = np.deg2rad(deg)
        u = -np.sin(rad)   # positive = eastward
        v = -np.cos(rad)   # positive = northward
        # Place barb at bar top, clamped to 95% of ylim
        y_pos = min(spd, ylim * 0.95)
        ax.barbs(x[i], y_pos, u * spd, v * spd,
                 length=6, linewidth=0.8,
                 barb_increments=dict(half=5, full=10, flag=50),
                 color='white', alpha=0.75, zorder=5)


def make_wind_bar_chart(ax_spd, ax_gust, records):
    """
    Side-by-side column charts: hourly avg wind speed (left) and max gust (right).
    Bars are coloured by intensity — calm blues through orange/red for high values.
    A thin dashed line connects the peaks so trends are easy to read.
    Wind barbs overlaid at bar tops.
    """
    times  = [r['time'] for r in reversed(records)]
    speeds = arr(records, 'wind_spd')[::-1]
    gusts  = arr(records, 'gust')[::-1]
    dirs   = arr(records, 'wind_dir')[::-1] if 'wind_dir' in records[0] else None
    x      = np.arange(len(times))
    tick_step = max(1, len(x) // 12)

    def speed_colors(vals, vmax):
        """Map each value to a colour: blue → cyan → amber → red."""
        cmap = matplotlib.colormaps.get_cmap('plasma')
        norm = np.clip(vals / max(vmax, 1), 0, 1)
        return [cmap(0.15 + 0.7 * v) for v in norm]

    def tiered_ylim(vals):
        """0-50 normally, 0-100 if any value > 50, 0-150 if any value > 100."""
        peak = vals.max() if len(vals) else 0
        if   peak > 100: return 150
        elif peak > 50:  return 100
        else:            return 50

    for ax, vals, label, ylabel in [
        (ax_spd,  speeds, 'Avg Wind Speed (mph)', 'mph'),
        (ax_gust, gusts,  'Max Gust (mph)',        'mph'),
    ]:
        vmax  = vals.max() if len(vals) else 1
        ylim  = tiered_ylim(vals)
        colors = speed_colors(vals, ylim)   # colour relative to axis ceiling
        ax.set_facecolor('#1a1a2e')

        # Major gridlines every 5 mph
        for y_val in range(0, ylim + 1, 5):
            ax.axhline(y_val, color='white', linewidth=1.1, linestyle='-', zorder=1)

        # Vertical 6-hour gridlines
        _add_6hr_vlines(ax, times, x)

        ax.bar(x, vals, color=colors, width=0.8, zorder=2, edgecolor='none')
        # Peak trend line
        ax.plot(x, vals, color='white', linewidth=0.8, alpha=0.4, zorder=3)

        # Wind barbs
        _draw_wind_barbs(ax, x, vals, dirs, ylim)

        ax.set_title(label, color='white', fontsize=22.0, fontweight='bold', pad=8)
        ax.set_ylabel(ylabel, color='lightgray', fontsize=18.0)
        ax.tick_params(colors='lightgray', labelsize=16.0)
        ax.spines[:].set_color('#444466')
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(times[::tick_step], fontsize=13.0,
                           color='lightgray', rotation=30, ha='right')
        ax.grid(False)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, ylim)

        # Label the peak bar
        peak_i = int(np.argmax(vals))
        ax.annotate(f'{vals[peak_i]:.0f}',
                    xy=(x[peak_i], vals[peak_i]),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=14.0, color='white', fontweight='bold')


def make_stats_panel(ax, stats, station_name, date_range):
    """
    Horizontal summary card: stats are laid out as columns across the panel.
    Row 1 (top): parameter names in light blue
    Row 2 (bottom): corresponding values in bold white
    A rounded box sits behind each column for visual grouping.

    The station name and date range appear as a header above the card.
    Number of columns = number of stats items; spacing is calculated dynamically.
    """
    ax.set_facecolor(BG)
    ax.axis('off')

    # Station name and date range as header
    ax.text(0.5, 0.97, station_name, transform=ax.transAxes,
            fontsize=24.0, fontweight='bold', color='white', ha='center', va='top')
    ax.text(0.5, 0.85, date_range, transform=ax.transAxes,
            fontsize=16.0, color='#aaaacc', ha='center', va='top')

    # Divider under header
    ax.axhline(y=0.78, color='#555577', linewidth=0.8, xmin=0.01, xmax=0.99)

    n = len(stats)
    if n == 0:
        return

    # Evenly space each stat column across the full width (with small margins)
    col_w = 1.0 / n
    for i, (label, value) in enumerate(stats):
        # Centre x of this column
        cx = (i + 0.5) * col_w

        # Rounded background box for each stat cell
        rect = mpatches.FancyBboxPatch(
            (i * col_w + 0.01, 0.08), col_w - 0.02, 0.62,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor='#1a1a2e', transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        # Row 1 — parameter name (upper half of the cell)
        ax.text(cx, 0.58, label, transform=ax.transAxes,
                fontsize=max(13.0, 19.0 - 1.0 * n), color='#aaccff',
                ha='center', va='center', fontweight='bold')

        # Thin divider between name and value within the cell
        ax.axhline(y=0.42, color='#333355', linewidth=0.5,
                   xmin=i * col_w + 0.02, xmax=(i + 1) * col_w - 0.02)

        # Row 2 — value (lower half of the cell)
        ax.text(cx, 0.26, value, transform=ax.transAxes,
                fontsize=max(15.0, 22.0 - 1.0 * n), color='white',
                ha='center', va='center', fontweight='bold')


def make_data_table_panel(ax, raw_rows, raw_hdrs):
    """
    Full raw data table — every column exactly as it appears on the source page.
    Headers come from the page's own column labels. Columns are evenly spaced.
    Font size shrinks automatically if there are many columns.
    """
    ax.set_facecolor(BG)
    ax.axis('off')

    ax.text(0.5, 0.97, 'Hourly Data', transform=ax.transAxes,
            fontsize=24.0, fontweight='bold', color='white', ha='center', va='top')
    ax.axhline(y=0.94, color='#555577', linewidth=0.8, xmin=0.02, xmax=0.98)

    if not raw_rows:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                color='gray', ha='center', va='center')
        return

    max_cols = max(len(r) for r in raw_rows)
    all_hdrs = ['Date', 'Time'] + [
        raw_hdrs[i] if i < len(raw_hdrs) else f'C{i+3}'
        for i in range(max_cols - 2)
    ]
    n_cols = len(all_hdrs)
    col_x = [0.02 + 0.96 * i / max(n_cols - 1, 1) for i in range(n_cols)]

    hdr_fs  = max(5.5, 8.5 - 0.35 * n_cols)
    data_fs = max(5.0, 8.0 - 0.35 * n_cols)

    # Multi-line headers: each header may contain \n-separated sub-labels
    # (e.g. "Wind\nSpd\nMPH"). We stack them top-to-bottom within the header row.
    hdr_line_h = 0.025   # vertical spacing between sub-label lines
    hdr_top    = 0.915   # y position of the top sub-label line
    for hdr, x in zip(all_hdrs, col_x):
        sub_labels = hdr.split('\n')
        for li, sub in enumerate(sub_labels):
            ax.text(x, hdr_top - li * hdr_line_h, sub,
                    transform=ax.transAxes, fontsize=hdr_fs,
                    color='#aaccff', fontweight='bold', va='top', ha='left')
    # Position the divider below the tallest possible header (allow 3 sub-lines)
    ax.axhline(y=hdr_top - 3 * hdr_line_h - 0.01,
               color='#444466', linewidth=0.5, xmin=0.02, xmax=0.98)

    # Data rows start below the header block (which can be up to 3 lines tall)
    data_top = 0.82
    n_rows = min(len(raw_rows), 26)
    row_h = data_top / n_rows
    for i, row in enumerate(raw_rows[:n_rows]):
        y = data_top - i * row_h
        bg_col = '#1a1a2e' if i % 2 == 0 else BG
        rect = mpatches.FancyBboxPatch(
            (0.01, y - row_h * 0.85), 0.98, row_h * 0.9,
            boxstyle="round,pad=0.002", linewidth=0,
            facecolor=bg_col, transform=ax.transAxes, clip_on=True)
        ax.add_patch(rect)
        padded = row + [''] * (n_cols - len(row))
        for val, x in zip(padded[:n_cols], col_x):
            ax.text(x, y - row_h * 0.3, val, transform=ax.transAxes,
                    fontsize=data_fs, color='white', va='center', ha='left')

    if len(raw_rows) > n_rows:
        ax.text(0.5, 0.01, f'({len(raw_rows)-n_rows} more rows not shown)',
                transform=ax.transAxes, fontsize=14.0, color='#666688',
                ha='center', va='bottom')


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================



def build_plots_separately(station_name, records, caps, raw_rows, raw_hdrs,
                            speed_bins=None, gust_bins=None, records_rose=None):
    # records_rose: optional 24h subset used for wind roses (falls back to records)
    """
    Render each chart as its own tall, readable figure.
    Returns a dict:
        {
          'temp':   matplotlib Figure or None,
          'wind':   matplotlib Figure or None,  (speed + gust roses side by side)
          'precip': matplotlib Figure or None,
          'stats':  list of (label, value) tuples,
          'date_range': str,
          'raw_rows': list of token lists,
          'raw_hdrs': list of header strings,
        }
    Caller is responsible for closing all figures.
    """
    if speed_bins is None: speed_bins = SPEED_BINS
    if gust_bins  is None: gust_bins  = GUST_BINS

    # Stats use 24h slice (records_rose) if available; plots use full 48h records
    stats_records = records_rose if records_rose is not None else records
    stats, date_range = compute_stats(stats_records, caps)

    cmap   = matplotlib.colormaps.get_cmap(CMAP_NAME)
    n      = len(speed_bins)
    colors = [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]

    result = {
        'temp':       None,
        'wind':       None,
        'precip':     None,
        'stats':      stats,
        'date_range': date_range,
        'raw_rows':   raw_rows,
        'raw_hdrs':   raw_hdrs,
    }

    # ── Temperature ───────────────────────────────────────────────────
    if caps['has_temp']:
        fig, ax = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout='constrained')
        ax.set_facecolor(BG)
        make_temp_plot(ax, records)
        fig.suptitle(f'{station_name} — Temperature',
                     fontsize=28.0, fontweight='bold', color='white')
        result['temp'] = fig

    # ── Wind: roses + bar charts in one 2×3 figure ───────────────────
    if caps['has_wind']:
        # Mixed subplot types: polar (roses) + cartesian (bar charts)
        # 3 columns: [rose | bar chart | legend]
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(15.0, 12), facecolor=BG)
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

        rose_recs = records_rose if records_rose is not None else records
        handles, leg_labels = make_wind_rose(ax_spd_rose, arr(rose_recs, 'wind_dir'),
                       arr(rose_recs, 'wind_spd'), 'Avg Wind Speed',
                       speed_bins, colors)
        make_wind_rose(ax_gust_rose, arr(rose_recs, 'wind_dir'),
                       arr(rose_recs, 'gust'), 'Max Gust',
                       gust_bins, colors)
        draw_rose_legend(ax_leg, handles, leg_labels, speed_bins)
        make_wind_bar_chart(ax_spd_bar, ax_gust_bar, records)

        fig.suptitle(f'{station_name} — Wind',
                     fontsize=28.0, fontweight='bold', color='white', y=0.97)
        result['wind'] = fig

    # ── Snow / Precip ─────────────────────────────────────────────────
    if caps['has_snow'] or caps['has_precip']:
        fig, ax = plt.subplots(figsize=(15.0, 6), facecolor=BG, layout='constrained')
        ax.set_facecolor(BG)
        make_precip_snow_plot(ax, records, caps)
        fig.suptitle(f'{station_name} — Precipitation / Snow',
                     fontsize=28.0, fontweight='bold', color='white')
        result['precip'] = fig

    return result


def build_figure(station_name, records, caps, raw_rows, raw_hdrs,
                 speed_bins=None, gust_bins=None):
    """
    Build and return a matplotlib Figure for one station.

    This is the single source of truth for the figure layout — both
    plot_station() (for local use) and app.py (for the web dashboard)
    call this function, so any layout or font change only needs to be
    made here.

    Parameters
    ----------
    station_name : str
    records      : list of record dicts from fetch_station_data()
    caps         : capability dict from detect_capabilities()
    raw_rows     : list of raw token lists for the data table
    raw_hdrs     : list of column header strings for the data table
    speed_bins   : wind speed bin edges (default: SPEED_BINS)
    gust_bins    : wind gust bin edges  (default: GUST_BINS)

    Returns
    -------
    fig : matplotlib Figure (caller is responsible for closing it)
    """
    if speed_bins is None: speed_bins = SPEED_BINS
    if gust_bins  is None: gust_bins  = GUST_BINS

    stats, date_range = compute_stats(records, caps)

    cmap   = matplotlib.colormaps.get_cmap(CMAP_NAME)
    n      = len(speed_bins)
    colors = [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]

    has_wind       = caps['has_wind']
    has_snow_precip = caps['has_snow'] or caps['has_precip']

    fig = plt.figure(figsize=(15.0, 8), facecolor=BG, layout="constrained")

    outer = gridspec.GridSpec(1, 2, width_ratios=[5, 2.5],
                              figure=fig, wspace=0.25)

    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.35, height_ratios=[1, 4])
    ax_summary = fig.add_subplot(right_gs[0])
    ax_table   = fig.add_subplot(right_gs[1])
    make_stats_panel(ax_summary, stats, station_name, date_range)
    make_data_table_panel(ax_table, raw_rows, raw_hdrs)

    left_spec = outer[0]

    if has_wind and has_snow_precip:
        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=left_spec, hspace=0.45, wspace=0.35)
        ax_temp = fig.add_subplot(left_gs[0, 0])
        ax_spd  = fig.add_subplot(left_gs[0, 1], projection='polar')
        ax_gust = fig.add_subplot(left_gs[0, 2], projection='polar')
        ax_snow = fig.add_subplot(left_gs[1, :])
        ax_spd.set_facecolor('#1a1a2e')
        ax_gust.set_facecolor('#1a1a2e')
        make_temp_plot(ax_temp, records)
        h, ll = make_wind_rose(ax_spd,  arr(records, 'wind_dir'), arr(records, 'wind_spd'),
                       'Wind Speed', speed_bins, colors)
        ax_spd.legend(h, ll, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      fontsize=9.0, title='Speed', title_fontsize=10.0,
                      facecolor='#16213e', edgecolor='#666688', labelcolor='white'
                      ).get_title().set_color('white')
        make_wind_rose(ax_gust, arr(records, 'wind_dir'), arr(records, 'gust'),
                       'Wind Gust', gust_bins, colors)
        make_precip_snow_plot(ax_snow, records, caps)

    elif has_wind:
        left_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=left_spec, wspace=0.35)
        ax_temp = fig.add_subplot(left_gs[0, 0])
        ax_spd  = fig.add_subplot(left_gs[0, 1], projection='polar')
        ax_gust = fig.add_subplot(left_gs[0, 2], projection='polar')
        ax_spd.set_facecolor('#1a1a2e')
        ax_gust.set_facecolor('#1a1a2e')
        make_temp_plot(ax_temp, records)
        h, ll = make_wind_rose(ax_spd,  arr(records, 'wind_dir'), arr(records, 'wind_spd'),
                       'Wind Speed', speed_bins, colors)
        ax_spd.legend(h, ll, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      fontsize=9.0, title='Speed', title_fontsize=10.0,
                      facecolor='#16213e', edgecolor='#666688', labelcolor='white'
                      ).get_title().set_color('white')
        make_wind_rose(ax_gust, arr(records, 'wind_dir'), arr(records, 'gust'),
                       'Wind Gust', gust_bins, colors)

    elif has_snow_precip:
        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=left_spec, hspace=0.45)
        ax_temp = fig.add_subplot(left_gs[0])
        ax_snow = fig.add_subplot(left_gs[1])
        make_temp_plot(ax_temp, records)
        make_precip_snow_plot(ax_snow, records, caps)

    else:
        left_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=left_spec)
        ax_temp = fig.add_subplot(left_gs[0])
        make_temp_plot(ax_temp, records)

    fig.suptitle(f'{station_name} — Weather Summary',
                 fontsize=32.0, fontweight='bold', y=1.01, color='white')
    return fig


def plot_station(station_name, url,
                 speed_bins=SPEED_BINS, gust_bins=GUST_BINS,
                 save_path=None):
    """
    Fetch and plot one station. Layout adapts to whatever data is available:

      Always shown:
        - Temperature line chart
        - Station summary panel
        - Full raw data table

      Shown when available:
        - Wind speed rose  (if wind_spd + wind_dir + gust present)
        - Wind gust rose   (if wind_spd + wind_dir + gust present)
        - Snow/precip chart (if new_snow, snow_depth, or precip present)

    Layout logic:
      The left portion of the figure is divided into rows (top/bottom) and
      columns depending on what's available:
        - Top row:    Temperature  |  Wind Speed Rose  |  Wind Gust Rose
        - Bottom row: Snow/Precip  |  (wind roses span both rows if no snow)
      Right portion: Summary panel + Data table (always present)

    Parameters
    ----------
    station_name : str   Display name for the figure title
    url          : str   Full URL of the wxstns HTML page
    speed_bins   : list  Bin edges for the wind speed rose
    gust_bins    : list  Bin edges for the gust rose
    save_path    : str   Optional path to save the PNG

    Returns
    -------
    fig : matplotlib Figure
    """
    print(f"\nFetching '{station_name}' ...")
    records, caps, raw_rows, raw_hdrs = fetch_station_data(url)
    print(f"  {len(records)} records | wind={caps['has_wind']} "
          f"snow={caps['has_snow']} precip={caps['has_precip']}")

    fig = build_figure(station_name, records, caps, raw_rows, raw_hdrs,
                       speed_bins, gust_bins)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Saved → {save_path}")

    plt.show()
    return fig


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    for station in STATIONS:
        plot_station(
            station_name=station["name"],
            url=station["url"],
            save_path=f"{station['name'].replace(' ', '_')}_weather.png"
        )
