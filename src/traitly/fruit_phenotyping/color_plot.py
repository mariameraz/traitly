# traitly/internal_structure/color.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
from typing import List, Dict, Tuple, Optional

# ============================================================================
# THIRD-PARTY LIBRARIES
# ============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from traitly.fruit_phenotyping.color_analysis import (renumber_fruit_locule_map,
                                                      normalize_lab_values, 
                                                      normalize_hsv_values)





def plot_fruit_color_distribution(
    df: pd.DataFrame,
    fruit_id: Optional[int] = None,   # None = toda la imagen
    color_space: str = 'rgb',
) -> None:
    """
    Plot pixel-value distributions.

    Parameters
    ----------
    df          : DataFrame from get_fruit_color_distributions()
    fruit_id    : If None → aggregate all fruits (whole image).
                  If int  → single fruit.
    color_space : 'all' or comma-separated subset of {'rgb','lab','hsv','gray'}
    """
    # ------------------------------------------------------------------ #
    # Validate color_space                                                 #
    # ------------------------------------------------------------------ #
    color_space = color_space.lower().replace(' ', '')
    spaces = {'rgb', 'lab', 'hsv', 'gray'} if color_space == 'all' else set(color_space.split(','))
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    if not spaces.issubset(valid_spaces):
        raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    # ------------------------------------------------------------------ #
    # Select rows and build title                                          #
    # ------------------------------------------------------------------ #
    if fruit_id is None:
        # Aggregate: sum all histograms → represents the whole image
        subset = df
        label_info  = df['label'].iloc[0] if 'label' in df.columns else ''
        title_fruit = f"All fruits (n={len(df)})  |  label: {label_info}"
    else:
        subset = df[df['fruit_id'] == fruit_id]
        if subset.empty:
            raise ValueError(f"fruit_id {fruit_id} not found in DataFrame.")
        label_info  = subset['label'].iloc[0] if 'label' in subset.columns else ''
        title_fruit = f"Fruit ID {fruit_id}  |  label: {label_info}"

    # ------------------------------------------------------------------ #
    # Channel definitions                                                  #
    # ------------------------------------------------------------------ #
    channel_defs = []
    if 'rgb' in spaces:
        channel_defs += [
            ('R', 256,   0,  255, 'r_channel',     'firebrick',     'R (0–255)'),
            ('G', 256,   0,  255, 'g_channel',     'forestgreen',   'G (0–255)'),
            ('B', 256,   0,  255, 'b_channel',     'steelblue',     'B (0–255)'),
        ]
    if 'lab' in spaces:
        channel_defs += [
            ('L', 101,   0,  100, 'gray_channel',  'dimgray',       'L (0–100)'),
            ('a', 256, -128, 127, 'a_channel',     'palevioletred', 'a (−128–127)'),
            ('b', 256, -128, 127, 'b_lab_channel', 'goldenrod',     'b (−128–127)'),
        ]
    if 'hsv' in spaces:
        channel_defs += [
            ('H', 360,   0,  359, 'hue_gradient',  'steelblue',     'Hue (0–360°)'),
            ('S', 101,   0,  100, 'gray_channel',  'darkorange',    'Saturation (0–100%)'),
            ('V', 101,   0,  100, 'gray_channel',  'seagreen',      'Value (0–100%)'),
        ]
    if 'gray' in spaces:
        channel_defs += [
            ('Gray', 256, 0, 255, 'gray_channel',  'gray',          'Gray (0–255)'),
        ]

    # ------------------------------------------------------------------ #
    # Stats helpers                                                        #
    # ------------------------------------------------------------------ #
    def _weighted_mean_std(vals: np.ndarray, counts: np.ndarray):
        total = counts.sum()
        if total == 0:
            return np.nan, np.nan
        mean = float(counts @ vals / total)
        std  = float(np.sqrt(counts @ (vals - mean) ** 2 / total))
        return mean, std

    def _circ_stats_from_hist(h_vals: np.ndarray, h_counts: np.ndarray):
        """Reuses circular_mean_and_std_hue logic, histogram version."""
        total = h_counts.sum()
        if total == 0:
            return np.nan, np.nan
        w   = h_counts / total
        rad = np.deg2rad(h_vals)
        s   = np.sum(np.sin(rad) * w)
        c   = np.sum(np.cos(rad) * w)
        mean_deg = (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0
        R = np.clip(np.sqrt(s ** 2 + c ** 2), 1e-12, 1.0)
        std_deg  = np.rad2deg(np.sqrt(-2.0 * np.log(R)))
        return float(mean_deg), float(std_deg)

    def _get_counts(ch: str, lo: int, hi: int) -> np.ndarray:
        """Sum histogram bins across selected rows."""
        cols = [f'{ch}_{i}' for i in range(lo, hi + 1)]
        return subset[cols].sum(axis=0).values.astype(float)

    # ------------------------------------------------------------------ #
    # Pre-compute H mean (for S/V bar tint — not used since gray now)     #
    # ------------------------------------------------------------------ #
    h_mean_deg = 0.0
    if 'hsv' in spaces:
        _hc = _get_counts('H', 0, 359)
        h_mean_deg, _ = _circ_stats_from_hist(np.arange(360), _hc)

    # ------------------------------------------------------------------ #
    # Color-bar builders                                                   #
    # ------------------------------------------------------------------ #
    def _make_colorbar_img(bar_type: str, n: int) -> np.ndarray:
        img = np.zeros((1, n, 3), dtype=np.float32)

        if bar_type == 'hue_gradient':
            hues = np.linspace(0, 179, n).astype(np.uint8)
            hsv_strip = np.zeros((1, n, 3), dtype=np.uint8)
            hsv_strip[0, :, 0] = hues
            hsv_strip[0, :, 1] = 255
            hsv_strip[0, :, 2] = 255
            img = cv2.cvtColor(hsv_strip, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        elif bar_type == 'gray_channel':
            v = np.linspace(0, 1, n)
            img[0, :, 0] = v; img[0, :, 1] = v; img[0, :, 2] = v

        elif bar_type == 'r_channel':
            img[0, :, 0] = np.linspace(0, 1, n)

        elif bar_type == 'g_channel':
            img[0, :, 1] = np.linspace(0, 1, n)

        elif bar_type == 'b_channel':
            img[0, :, 2] = np.linspace(0, 1, n)

        elif bar_type == 'a_channel':
            t = np.linspace(0, 1, n)
            img[0, :, 0] = t; img[0, :, 1] = 1 - t; img[0, :, 2] = 0.2

        elif bar_type == 'b_lab_channel':
            t = np.linspace(0, 1, n)       # 0 = blue (-128), 1 = yellow (+127)
            img[0, :, 0] = t               # R
            img[0, :, 1] = t               # G
            img[0, :, 2] = 1 - t           # B

        return np.clip(img, 0, 1)

    # ------------------------------------------------------------------ #
    # Figure: 1 row × n_channels cols                                     #
    # ------------------------------------------------------------------ #
    n_ch = len(channel_defs)
    fig, axes = plt.subplots(1, n_ch, figsize=(5.5 * n_ch, 5.5))
    if n_ch == 1:
        axes = [axes]

    fig.suptitle(
        f"{title_fruit}  |  {color_space.upper()} distributions",
        fontweight='bold', fontsize=13, y=1.02
    )

    for ax, (ch, n_bins, lo, hi, bar_type, plot_color, xlabel) in zip(axes, channel_defs):
        counts = _get_counts(ch, lo, hi)
        x_vals = np.arange(lo, hi + 1)

        if ch == 'H':
            mean_val, std_val = _circ_stats_from_hist(x_vals, counts)
            stat_label = f'Circ. mean: {mean_val:.1f}°\nCirc. std:  {std_val:.1f}°'
        else:
            mean_val, std_val = _weighted_mean_std(x_vals, counts)
            stat_label = f'Mean: {mean_val:.1f}\nStd:  {std_val:.1f}'

        total = counts.sum()
        freq  = counts / total if total > 0 else counts

        ax.bar(x_vals, freq, width=1.0, color=plot_color, alpha=0.72)
        ax.axvline(mean_val, color='red', linewidth=2.0,
                   linestyle='--', label=stat_label)
        ax.set_xlim(lo, hi)
        ax.set_ylabel('Rel. frequency', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11, labelpad=32)
        ax.legend(fontsize=9.5, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)

        cb_ax = ax.inset_axes([0.04, -0.14, 0.96, 0.07])
        cb_img = _make_colorbar_img(bar_type, n_bins)
        cb_ax.imshow(cb_img, aspect='auto', extent=[lo, hi, 0, 1])
        cb_ax.set_xlim(lo, hi)
        cb_ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def get_fruit_color_samples(
    img: np.ndarray,
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    color_space: str = 'hsv',
    max_pixels: Optional[int] = 5000,
    renumber: bool = True,
    dark_threshold: int = 15,
    jitter: bool = False,
    jitter_strength: float = 0.5,
) -> pd.DataFrame:
    """
    Extract raw pixel values from total_pericarp for scatter plotting.

    Parameters
    ----------
    img, mask, contours, fruit_locule_map : same as get_fruit_color_distributions()
    fruit_id    : None → all fruits; int → single fruit
    color_space : 'all' or comma-separated subset of {'rgb','lab','hsv','gray'}
    max_pixels  : Max pixels to sample randomly (None = all)
    renumber    : Renumber fruit IDs from 1..n
    dark_threshold : Exclude pixels with gray <= threshold
    jitter      : Add small gaussian noise to smooth discretization (for plots only)
    jitter_strength : Scale of jitter noise

    Returns
    -------
    pd.DataFrame — one row per pixel:
        fruit_id, [R, G, B], [L, a, b], [H, S, V], [Gray]
        + R_norm, G_norm, B_norm (0–1, always present for point coloring)
    """
    color_space = color_space.lower().replace(' ', '')
    spaces = {'rgb', 'lab', 'hsv', 'gray'} if color_space == 'all' else set(color_space.split(','))
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    if not spaces.issubset(valid_spaces):
        raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    # Pre-convert color spaces once
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab_full  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if 'lab' in spaces else None
    hsv_full  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if 'hsv' in spaces else None

    if renumber:
        iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        iter_map = fruit_locule_map
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    # Filter to single fruit if requested
    if fruit_id is not None:
        if fruit_id not in iter_map:
            raise ValueError(f"fruit_id {fruit_id} not found.")
        iter_map = {fruit_id: iter_map[fruit_id]}

    all_records = []

    for fid, locule_indices in iter_map.items():
        original_fid = fruit_id_map[fid]
        if original_fid >= len(contours):
            continue
        fruit_contour = contours[original_fid]
        if fruit_contour is None or len(fruit_contour) == 0:
            continue

        x, y, w, h = cv2.boundingRect(fruit_contour)
        roi_img  = img      [y:y+h, x:x+w]
        roi_mask = mask     [y:y+h, x:x+w]
        roi_gray = gray_full[y:y+h, x:x+w]

        valid = (roi_mask > 0) & (roi_gray > dark_threshold)

        # Exclude locules
        if locule_indices:
            loc_mask = np.zeros((h, w), dtype=np.uint8)
            for loc_idx in locule_indices:
                if loc_idx >= len(contours):
                    continue
                lc = contours[loc_idx]
                if lc is None or len(lc) == 0:
                    continue
                lc = lc.copy()
                lc[:, :, 0] -= x
                lc[:, :, 1] -= y
                cv2.drawContours(loc_mask, [lc], -1, 255, cv2.FILLED)
            valid &= (loc_mask == 0)

        if not np.any(valid):
            continue

        n_valid = int(valid.sum())
        bgr_pixels = roi_img[valid].astype(np.float32)  # Nx3

        record: Dict[str, np.ndarray] = {
            'fruit_id': np.full(n_valid, fid, dtype=np.int32),
            # Clip to [0, 1] to avoid matplotlib color issues
            'R_norm': np.clip(bgr_pixels[:, 2] / 255.0, 0.0, 1.0),
            'G_norm': np.clip(bgr_pixels[:, 1] / 255.0, 0.0, 1.0),
            'B_norm': np.clip(bgr_pixels[:, 0] / 255.0, 0.0, 1.0),
        }

        if 'rgb' in spaces:
            record['R'] = bgr_pixels[:, 2]
            record['G'] = bgr_pixels[:, 1]
            record['B'] = bgr_pixels[:, 0]

        if 'gray' in spaces:
            record['Gray'] = roi_gray[valid].astype(np.float32)

        if 'lab' in spaces:
            roi_lab = lab_full[y:y+h, x:x+w]
            l_raw = roi_lab[:, :, 0][valid]
            a_raw = roi_lab[:, :, 1][valid]
            b_raw = roi_lab[:, :, 2][valid]
            l_norm, a_norm, b_norm = normalize_lab_values(l_raw, a_raw, b_raw)
            record['L'] = l_norm
            record['a'] = a_norm
            record['b'] = b_norm

        if 'hsv' in spaces:
            roi_hsv = hsv_full[y:y+h, x:x+w]
            h_raw = roi_hsv[:, :, 0][valid]
            s_raw = roi_hsv[:, :, 1][valid]
            v_raw = roi_hsv[:, :, 2][valid]
            h_norm, s_norm, v_norm = normalize_hsv_values(h_raw, s_raw, v_raw)
            record['H'] = h_norm   # 0–360
            record['S'] = s_norm   # 0–100
            record['V'] = v_norm   # 0–100

        all_records.append(pd.DataFrame(record))

    if not all_records:
        return pd.DataFrame()

    df_pixels = pd.concat(all_records, ignore_index=True)

    # Subsample
    if max_pixels is not None and len(df_pixels) > max_pixels:
        df_pixels = df_pixels.sample(n=max_pixels, random_state=42).reset_index(drop=True)

    # Optional jitter to smooth discretization artifacts (for visualization only)
    if jitter:
        jitter_scales = {
            'H': 1.0, 'S': 0.3, 'V': 0.3,
            'L': 0.3, 'a': 0.3, 'b': 0.3,
            'R': 0.5, 'G': 0.5, 'B': 0.5, 'Gray': 0.5,
        }
        rng = np.random.default_rng(42)
        for col, scale in jitter_scales.items():
            if col in df_pixels.columns:
                df_pixels[col] = df_pixels[col] + rng.normal(
                    0, scale * jitter_strength, size=len(df_pixels)
                )

    return df_pixels


def plot_fruit_color_scatter(
    img: np.ndarray,
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    color_space: str = 'rgb',
    max_pixels: int = 10000,
    plot_size: Tuple[int, int] = (18, 5),
    renumber: bool = True,
    dark_threshold: int = 15,
) -> None:
    """
    2D scatter plots between all pairs of channels in the chosen color space.
    Points are colored by their real RGB pixel color.

    Parameters
    ----------
    img, mask, contours, fruit_locule_map : raw image data
    fruit_id    : None → all fruits; int → single fruit
    color_space : single space — one of {'rgb', 'lab', 'hsv'}
    max_pixels  : pixels to sample (for performance)
    plot_size   : figure size (width, height)
    """
    color_space = color_space.lower().replace(' ', '')
    if ',' in color_space or color_space == 'all':
        raise ValueError(
            "plot_fruit_color_scatter() accepts a single color space "
            "(e.g. 'hsv', 'rgb', 'lab'). Use multiple calls for multiple spaces."
        )

    space_channels = {
        'hsv': [('H', 'Hue (0–360°)'),     ('S', 'Saturation (0–100)'), ('V', 'Value (0–100)')],
        'rgb': [('R', 'R (0–255)'),         ('G', 'G (0–255)'),          ('B', 'B (0–255)')],
        'lab': [('L', 'L (0–100)'),         ('a', 'a (−128–127)'),       ('b', 'b (−128–127)')],
    }
    if color_space not in space_channels:
        raise ValueError(f"Invalid color_space '{color_space}'. Valid: {set(space_channels)}")

    channels = space_channels[color_space]
    pairs = [
        (channels[i], channels[j])
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
    ]

    # Extract pixel samples
    df_px = get_fruit_color_samples(
        img=img, mask=mask, contours=contours,
        fruit_locule_map=fruit_locule_map,
        fruit_id=fruit_id,
        color_space=color_space,
        max_pixels=max_pixels,
        renumber=renumber,
        dark_threshold=dark_threshold,
        jitter=True,
        jitter_strength=0.4
    )

    if df_px.empty:
        print("No pixels found for the selected fruit(s).")
        return

    # Clip to [0, 1] — required by matplotlib for RGB point colors
    point_colors = np.clip(df_px[['R_norm', 'G_norm', 'B_norm']].values, 0.0, 1.0)

    # Stats
    def _circ_mean_std(vals: np.ndarray):
        rad = np.deg2rad(vals)
        s, c = np.mean(np.sin(rad)), np.mean(np.cos(rad))
        mean_deg = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
        R = np.clip(np.sqrt(s**2 + c**2), 1e-12, 1.0)
        return mean_deg, np.rad2deg(np.sqrt(-2.0 * np.log(R)))

    def _stat_label(col: str) -> str:
        vals = df_px[col].values
        if col == 'H':
            m, s = _circ_mean_std(vals)
            return f'Circ. mean={m:.1f}°  Circ. std={s:.1f}°'
        return f'mean={np.mean(vals):.1f}  std={np.std(vals):.1f}'

    # Font sizes
    width = plot_size[0]
    title_fontsize = int(np.clip(8 + width * 0.6, 10, 24))
    label_fontsize = int(np.clip(6 + width * 0.4,  8, 18))
    tick_fontsize  = int(np.clip(5 + width * 0.3,  7, 14))

    # Title
    n_fruits = df_px['fruit_id'].nunique()
    title = (
        f"All fruits (n={n_fruits}) — {color_space.upper()} scatter  |  {len(df_px):,} pixels"
        if fruit_id is None else
        f"Fruit ID {fruit_id} — {color_space.upper()} scatter  |  {len(df_px):,} pixels"
    )

    # Figure
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=plot_size)
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(title, fontweight='bold', fontsize=title_fontsize, y=1.02)

    for ax, ((cx, lx), (cy, ly)) in zip(axes, pairs):
        ax.scatter(
            df_px[cx].values, df_px[cy].values,
            c=point_colors, alpha=0.6, s=10,
            linewidths=0, rasterized=True
        )
        ax.set_xlabel(f'{lx}\n{_stat_label(cx)}', fontsize=label_fontsize)
        ax.set_ylabel(f'{ly}\n{_stat_label(cy)}', fontsize=label_fontsize)
        ax.set_title(f'{cx} vs {cy}', fontweight='bold', fontsize=title_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

def compute_fruit_hue_index(
    df: pd.DataFrame,
    red_hue_ranges: List[Tuple[int, int]] = [(0, 20), (340, 360)],
    yellow_hue_range: Tuple[int, int] = (40, 70),
    homogeneity_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Compute a red/yellow color index per fruit from Hue histogram columns.

    Parameters
    ----------
    df                     : DataFrame from get_fruit_color_distributions()
                             must contain columns H_0..H_359
    red_hue_ranges         : list of (min°, max°) defining red hue zones
                             default covers 0–20° and 340–360° (wraps around 0)
    yellow_hue_range       : (min°, max°) defining yellow hue zone
    homogeneity_threshold  : red_ratio above this → 'red'
                             below (1 - threshold)  → 'yellow'
                             else                   → 'mixed'

    Returns
    -------
    pd.DataFrame with columns:
        image_name, fruit_id,
        red_pixels, yellow_pixels, total_hue_pixels,
        red_ratio,           # 0 = all yellow, 1 = all red
        yellow_ratio,        # 1 - red_ratio (among red+yellow pixels)
        color_homogeneity,   # 0.5 = perfectly mixed, 1.0 = perfectly uniform
        color_category       # 'red', 'yellow', or 'mixed'
    """
    h_cols_all = [f'H_{i}' for i in range(360)]

    # Validate that H columns exist
    missing = [c for c in h_cols_all if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Hue columns in df. Did you run get_fruit_color_distributions with color_space='hsv'?")

    # Build index sets for red and yellow bins
    red_bins = set()
    for lo, hi in red_hue_ranges:
        red_bins.update(range(lo, hi + 1))

    y_lo, y_hi = yellow_hue_range
    yellow_bins = set(range(y_lo, y_hi + 1))

    red_cols    = [f'H_{i}' for i in sorted(red_bins)    if f'H_{i}' in df.columns]
    yellow_cols = [f'H_{i}' for i in sorted(yellow_bins) if f'H_{i}' in df.columns]

    records = []
    for _, row in df.iterrows():
        red_px    = row[red_cols].sum()
        yellow_px = row[yellow_cols].sum()
        total_px  = red_px + yellow_px

        if total_px == 0:
            red_ratio         = np.nan
            yellow_ratio      = np.nan
            color_homogeneity = np.nan
            color_category    = 'unknown'
        else:
            red_ratio         = red_px / total_px
            yellow_ratio      = yellow_px / total_px
            # 0.5 = perfectly mixed, 1.0 = perfectly uniform (all one color)
            color_homogeneity = float(max(red_ratio, yellow_ratio))

            if red_ratio >= homogeneity_threshold:
                color_category = 'red'
            elif yellow_ratio >= homogeneity_threshold:
                color_category = 'yellow'
            else:
                color_category = 'mixed'

        records.append({
            'image_name':        row.get('image_name', ''),
            'fruit_id':          row['fruit_id'],
            'red_pixels':        int(red_px),
            'yellow_pixels':     int(yellow_px),
            'total_hue_pixels':  int(total_px),
            'red_ratio':         round(float(red_ratio),    4),
            'yellow_ratio':      round(float(yellow_ratio), 4),
            'color_homogeneity': round(float(color_homogeneity), 4),
            'color_category':    color_category,
        })

    return pd.DataFrame(records)


def plot_fruit_color_correlation(
    df: pd.DataFrame,
    fruit_id: Optional[int] = None,
    color_space: str = 'hsv',
    method: str = 'pearson',
    plot_size: Tuple[int, int] = (8, 6),
    cluster: bool = False,
    annotate: bool = True,
    triangle: str = 'full',
) -> pd.DataFrame:
    """
    Compute and plot correlation matrix between color channels.

    Parameters
    ----------
    df          : DataFrame from get_fruit_color_distributions()
    fruit_id    : None → all fruits; int → single fruit (warns if < 2)
    color_space : 'all' or comma-separated subset of {'rgb','lab','hsv','gray'}
    method      : 'pearson' or 'spearman'
    plot_size   : figure size
    cluster     : If True, reorder channels by hierarchical clustering
    annotate    : If True, show correlation values inside each cell
    triangle    : 'full', 'upper', or 'lower'

    Returns
    -------
    pd.DataFrame — correlation matrix (reordered if cluster=True)
    """


    # ------------------------------------------------------------------ #
    # Validate                                                             #
    # ------------------------------------------------------------------ #
    color_space = color_space.lower().replace(' ', '')
    spaces = {'rgb', 'lab', 'hsv', 'gray'} if color_space == 'all' else set(color_space.split(','))
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    if not spaces.issubset(valid_spaces):
        raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    method = method.lower()
    if method not in {'pearson', 'spearman'}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    if triangle not in {'full', 'upper', 'lower'}:
        raise ValueError("triangle must be 'full', 'upper', or 'lower'")

    # ------------------------------------------------------------------ #
    # Select rows                                                          #
    # ------------------------------------------------------------------ #
    if fruit_id is not None:
        subset = df[df['fruit_id'] == fruit_id]
        if subset.empty:
            raise ValueError(f"fruit_id {fruit_id} not found.")
        if len(subset) < 2:
            print("⚠️  Correlation needs more than 1 fruit — showing all fruits instead.")
            subset = df
    else:
        subset = df

    label_info = subset['label'].iloc[0] if 'label' in subset.columns else ''
    n = len(subset)

    # ------------------------------------------------------------------ #
    # Channel definitions                                                  #
    # ------------------------------------------------------------------ #
    channel_defs = []
    if 'rgb'  in spaces: channel_defs += [('R', 0, 255),    ('G', 0, 255),    ('B', 0, 255)]
    if 'lab'  in spaces: channel_defs += [('L', 0, 100),    ('a', -128, 127), ('b', -128, 127)]
    if 'hsv'  in spaces: channel_defs += [('H', 0, 359),    ('S', 0, 100),    ('V', 0, 100)]
    if 'gray' in spaces: channel_defs += [('Gray', 0, 255)]

    # ------------------------------------------------------------------ #
    # Weighted mean per channel per fruit                                  #
    # ------------------------------------------------------------------ #
    def _circ_mean_from_row(row, lo, hi):
        cols   = [f'H_{i}' for i in range(lo, hi + 1)]
        counts = row[cols].values.astype(float)
        total  = counts.sum()
        if total == 0:
            return np.nan
        w   = counts / total
        rad = np.deg2rad(np.arange(lo, hi + 1))
        s, c = np.sum(np.sin(rad) * w), np.sum(np.cos(rad) * w)
        return float((np.rad2deg(np.arctan2(s, c)) + 360) % 360)

    def _arith_mean_from_row(row, ch, lo, hi):
        cols   = [f'{ch}_{i}' for i in range(lo, hi + 1)]
        counts = row[cols].values.astype(float)
        total  = counts.sum()
        if total == 0:
            return np.nan
        return float(counts @ np.arange(lo, hi + 1) / total)

    means_data: Dict[str, List[float]] = {ch: [] for ch, _, _ in channel_defs}
    for _, row in subset.iterrows():
        for ch, lo, hi in channel_defs:
            if ch == 'H':
                means_data[ch].append(_circ_mean_from_row(row, lo, hi))
            else:
                means_data[ch].append(_arith_mean_from_row(row, ch, lo, hi))

    means_df = pd.DataFrame(means_data)

    # ------------------------------------------------------------------ #
    # Correlation matrix                                                   #
    # ------------------------------------------------------------------ #
    corr_matrix = means_df.corr(method=method)

    # ------------------------------------------------------------------ #
    # Optional clustering                                                  #
    # ------------------------------------------------------------------ #
    if cluster:
        dist_matrix = 1 - corr_matrix.abs().values
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        condensed      = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method='average')
        order          = leaves_list(linkage_matrix)
        channels_ordered = [corr_matrix.columns[i] for i in order]
        corr_matrix = corr_matrix.loc[channels_ordered, channels_ordered]

    # ------------------------------------------------------------------ #
    # Font sizes                                                           #
    # ------------------------------------------------------------------ #
    width          = plot_size[0]
    title_fontsize = int(np.clip(8 + width * 0.6, 10, 24))
    tick_fontsize  = int(np.clip(5 + width * 0.3,  7, 14))
    annot_fontsize = int(np.clip(4 + width * 0.4,  7, 16))

    # ------------------------------------------------------------------ #
    # Triangle mask                                                        #
    # ------------------------------------------------------------------ #
    channels    = list(corr_matrix.columns)
    n_ch        = len(channels)
    corr_values = corr_matrix.values.copy()  # copy to avoid view issues

    if triangle != 'full':
        for i in range(n_ch):
            for j in range(n_ch):
                if triangle == 'upper' and i > j:
                    corr_values[i, j] = np.nan
                elif triangle == 'lower' and i < j:
                    corr_values[i, j] = np.nan

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=plot_size)

    masked_values = np.ma.masked_invalid(corr_values)
    im = ax.imshow(masked_values, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='auto', zorder=1)

    # White background for masked (empty triangle) cells
    ax.set_facecolor('white')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')

    # ------------------------------------------------------------------ #
    # Annotations                                                          #
    # ------------------------------------------------------------------ #
    if annotate:
        for i in range(n_ch):
            for j in range(n_ch):
                val = corr_values[i, j]
                if np.isnan(val):
                    continue
                text_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=annot_fontsize, color=text_color,
                        fontweight='bold', zorder=10)

    # ------------------------------------------------------------------ #
    # Ticks — major first, then minor (order matters in matplotlib)       #
    # ------------------------------------------------------------------ #
    ax.set_xticks(range(n_ch))
    ax.set_yticks(range(n_ch))
    ax.set_xticklabels(channels, fontsize=tick_fontsize, rotation=45, ha='right')
    ax.set_yticklabels(channels, fontsize=tick_fontsize)

    # Minor ticks after major — for grid lines between cells
    ax.set_xticks(np.arange(n_ch + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_ch + 1) - 0.5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Grid — gray so visible on both colored and white cells
    ax.grid(which='minor', color='#cccccc', linewidth=1.0, zorder=5)

    # ------------------------------------------------------------------ #
    # Title                                                                #
    # ------------------------------------------------------------------ #
    cluster_label = ' · clustered' if cluster else ''
    triangle_label = f' · {triangle} triangle' if triangle != 'full' else ''
    ax.set_title(
        f"All fruits (n={n})  |  label: {label_info}  |  "
        f"{color_space.upper()} {method.capitalize()} correlation"
        f"{cluster_label}{triangle_label}",
        fontweight='bold', fontsize=title_fontsize, pad=12
    )

    plt.tight_layout()
    plt.show()

    return corr_matrix