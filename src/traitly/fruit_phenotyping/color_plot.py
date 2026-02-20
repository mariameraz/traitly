# traitly/fruit_phenotyping/color.py

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

# ============================================================================
# Create histogram plots for each color channels
# ============================================================================

def plot_color_histogram(
    df: pd.DataFrame,
    fruit_id: Optional[int] = None,
    color_space: Optional[str] = None,
    legend_position: str = 'top-right',
    legend_font_size: float = 9.5,
    xlabel_font_size: float = 11.0,
    ylabel_font_size: float = 10.0,
    axes_font_size: float = 9.0,
    overlay: bool = False,       
    alpha: float = 0.72,         
) -> None:
    """
    Plot pixel-value histograms.
    One row per color space.

    overlay=False  → one column per channel (default)
    overlay=True   → all channels of a space overlaid in a single plot
    """

    _LOC_MAP = {
        'top-right':    'upper right',
        'top-left':     'upper left',
        'bottom-right': 'lower right',
        'bottom-left':  'lower left',
        'none': 'none'
    }
    if legend_position not in _LOC_MAP:
        raise ValueError(f"legend_position: {legend_position} must be one of {list(_LOC_MAP)}. Got '{legend_position}'.")
    mpl_loc = _LOC_MAP[legend_position]
    
    if mpl_loc == 'none':
        mpl_loc = None
    
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    _sentinel    = {'rgb': 'R_0', 'lab': 'L_0', 'hsv': 'H_0', 'gray': 'Gray_0'}

    if color_space is None:
        spaces = {s for s, col in _sentinel.items() if col in df.columns}
        if not spaces:
            raise ValueError(
                "Could not detect any color space in the DataFrame. "
                "Expected at least one of these columns: " + str(list(_sentinel.values()))
            )
    else:
        color_space = color_space.lower().replace(' ', '')
        spaces = valid_spaces if color_space == 'all' else set(color_space.split(','))
        if not spaces.issubset(valid_spaces):
            raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    cs_label = (color_space or'auto').upper()

    bad_spaces = [s for s in spaces if _sentinel[s] not in df.columns]
    if bad_spaces:
        raise ValueError(
            f"DataFrame does not contain histogram columns for: {bad_spaces}. "
            f"Expected columns like {[_sentinel[s] for s in bad_spaces]}. "
            f"Run analyze_color() with get_color_histogram=True first."
        )

    if fruit_id is None:
        subset = df
        image_name  = df['image_name'].iloc[0] if 'image_name' in df.columns else ''
        title_fruit = f"All fruits (n={len(df)}) in {image_name}"
    else:
        subset = df[df['fruit_id'] == fruit_id]
        if subset.empty:
            raise ValueError(f"fruit_id {fruit_id} not found in DataFrame.")
        image_name  = subset['image_name'].iloc[0] if 'image_name' in subset.columns else ''
        title_fruit = f"Fruit ID {fruit_id} in {image_name}"

    space_channels = {}
    if 'rgb' in spaces:
        space_channels['RGB'] = [
            ('R', 256,   0,  255, 'r_channel',     'firebrick',     'R (0-255)'),
            ('G', 256,   0,  255, 'g_channel',     'forestgreen',   'G (0-255)'),
            ('B', 256,   0,  255, 'b_channel',     'steelblue',     'B (0-255)'),
        ]
    if 'lab' in spaces:
        space_channels['LAB'] = [
            ('L', 101,   0,  100, 'gray_channel',  'dimgray',       'L (0-100)'),
            ('a', 256, -128, 127, 'a_channel',     'palevioletred', 'a (-128-127)'),
            ('b', 256, -128, 127, 'b_lab_channel', 'goldenrod',     'b (-128-127)'),
        ]
    if 'hsv' in spaces:
        space_channels['HSV'] = [
            ('H', 360,   0,  359, 'hue_gradient',  'steelblue',     'Hue (0-360)'),
            ('S', 101,   0,  100, 'gray_channel',  'darkorange',    'Saturation (0-100%)'),
            ('V', 101,   0,  100, 'gray_channel',  'seagreen',      'Value (0-100%)'),
        ]
    if 'gray' in spaces:
        space_channels['Gray'] = [
            ('Gray', 256, 0, 255, 'gray_channel',  'gray',          'Gray (0-255)'),
        ]

    # Plot layout, default  1 col per channel. If overlay, 1 col per space for RGB ONLY.
    nrows = len(space_channels)
    ncols = 1 if overlay else max(len(chs) for chs in space_channels.values())

    # Helpers
    def _weighted_mean_std(vals, counts):
        total = counts.sum()
        if total == 0:
            return np.nan, np.nan
        mean = float(counts @ vals / total)
        std  = float(np.sqrt(counts @ (vals - mean) ** 2 / total))
        return mean, std

    def _circ_stats_from_hist(h_vals, h_counts):
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

    def _get_counts(ch, lo, hi):
        cols = [f'{ch}_{i}' for i in range(lo, hi + 1)]
        return subset[cols].sum(axis=0).values.astype(float)

    def _make_colorbar_img(bar_type, n):
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
            t = np.linspace(0, 1, n)
            img[0, :, 0] = t; img[0, :, 1] = t; img[0, :, 2] = 1 - t
        return np.clip(img, 0, 1)

    # Validate overlay before creating the figure
    if overlay:
        if spaces != {'rgb'} and 'rgb' not in spaces:
            raise ValueError(
                "overlay=True is only available for RGB. "
                "Run with color_space='rgb' or include 'rgb' in your color_space."
            )
        if not spaces.issubset({'rgb'}):
            non_rgb = spaces - {'rgb'}
            raise ValueError(
                f"overlay=True is only available for RGB, but you also requested: {non_rgb}. "
                f"Use color_space='rgb' when overlay=True."
            )
        if 'R_0' not in df.columns:
            raise ValueError(
                "overlay=True requires RGB histogram columns (e.g. 'R_0', 'G_0', 'B_0'). "
                "Run analyze_color() with get_color_histogram=True and color_space='rgb' first."
            )

    # Figure
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 5.5 * nrows),
        squeeze=False
    )

    fig.suptitle(
        f"{title_fruit} ({cs_label})",
        fontweight='bold', fontsize=13, y=1.01
    )

    for row_idx, (space_label, channels) in enumerate(space_channels.items()):


        if overlay:
            ax = axes[row_idx, 0]
            ax.set_title(space_label, fontsize=11, fontweight='bold',
                         loc='left', pad=6, color='#333333')



            for ch, n_bins, lo, hi, bar_type, plot_color, xlabel in channels:
                counts = _get_counts(ch, lo, hi)
                x_vals = np.arange(lo, hi + 1)

                if ch == 'H':
                    mean_val, std_val = _circ_stats_from_hist(x_vals, counts)
                else:
                    mean_val, std_val = _weighted_mean_std(x_vals, counts)

                total = counts.sum()
                freq  = counts / total if total > 0 else counts

                x_plot    = x_vals
                mean_plot = mean_val
                bar_width = 1.0

                mean_label = f'{ch}  mean: {mean_val:.1f}  std: {std_val:.1f}'
                ax.bar(x_plot, freq, width=bar_width, color=plot_color,
                       alpha=alpha, label=mean_label)
                ax.axvline(mean_plot, color=plot_color, linewidth=1.5, linestyle='--')
                ax.tick_params(labelsize=axes_font_size)
                if mpl_loc is not None:
                    ax.legend(fontsize=legend_font_size, loc=mpl_loc)
                ax.spines[['top', 'right']].set_visible(False)

        # Normal mode with one axis per channel
        else:
            for col_idx in range(ncols):
                ax = axes[row_idx, col_idx]

                if col_idx >= len(channels):
                    ax.set_visible(False)
                    continue

                ch, n_bins, lo, hi, bar_type, plot_color, xlabel = channels[col_idx]
                counts = _get_counts(ch, lo, hi)
                x_vals = np.arange(lo, hi + 1)

                if ch == 'H':
                    mean_val, std_val = _circ_stats_from_hist(x_vals, counts)
                    stat_label = f'Circ. mean: {mean_val:.1f}\nCirc. std:  {std_val:.1f}'
                else:
                    mean_val, std_val = _weighted_mean_std(x_vals, counts)
                    stat_label = f'Mean: {mean_val:.1f}\nStd:  {std_val:.1f}'

                total = counts.sum()
                freq  = counts / total if total > 0 else counts

                ax.bar(x_vals, freq, width=1.0, color=plot_color, alpha=alpha)
                ax.axvline(mean_val, color='red', linewidth=2.0, linestyle='--', label=stat_label)
                ax.set_xlim(lo, hi)
                ax.set_ylabel('Rel. frequency', fontsize=ylabel_font_size)
                ax.set_xlabel(xlabel, fontsize=xlabel_font_size, labelpad=32)
                ax.tick_params(labelsize=axes_font_size)
                if mpl_loc is not None:
                    ax.legend(fontsize=legend_font_size, loc=mpl_loc)
                ax.spines[['top', 'right']].set_visible(False)

                # Space label on the left-most column of each row
                if col_idx == 0:
                    ax.set_title(space_label, fontsize=11, fontweight='bold',
                                 loc='left', pad=6, color='#333333')

                cb_ax = ax.inset_axes([0.04, -0.14, 0.96, 0.07])
                cb_img = _make_colorbar_img(bar_type, n_bins)
                cb_ax.imshow(cb_img, aspect='auto', extent=[lo, hi, 0, 1])
                cb_ax.set_xlim(lo, hi)
                cb_ax.axis('off')

    plt.tight_layout()
    plt.show()

# ============================================================================
# Create scatter plots for color channels pairwise comparison
# ============================================================================

def get_fruit_color_samples(
    img: np.ndarray,
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_locule_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    color_space: str = 'hsv',
    sample_size: Optional[int] = 5000,
    renumber: bool = True,
    dark_threshold: int = 15,
    jitter: bool = False,
    jitter_strength: float = 0.5,
    erosion_px: int = 3,   
) -> pd.DataFrame:
    """
    Extract raw pixel values from total_pericarp for scatter plotting.

    Parameters
    ----------
    img, mask, contours, fruit_locule_map : same as get_fruit_color_histogram()
    fruit_id         : None → all fruits; int → single fruit
    color_space      : 'all' or comma-separated subset of {'rgb','lab','hsv','gray'}
    sample_size       : Max pixels to sample randomly (None = all)
    renumber         : Renumber fruit IDs from 1..n
    dark_threshold   : Exclude pixels with gray <= threshold
    jitter           : Add small gaussian noise to smooth discretization (for plots only)
    jitter_strength  : Scale of jitter noise
    edge_erosion_px  : Erode the mask by this many pixels before sampling to exclude
                       border artifacts caused by background bleed-in. Only affects
                       color extraction — does not modify the original mask.
                       Use 0 (default) to disable.

    Returns
    -------
    pd.DataFrame — one row per pixel:
        fruit_id, [R, G, B], [L, a, b], [H, S, V], [Gray]
        + R_norm, G_norm, B_norm (0-1, always present for point coloring)
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

    # Erode mask once globally if requested
    if erosion_px > 0:
        kernel       = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_px * 2 + 1, erosion_px * 2 + 1)
        )
        mask_sampled = cv2.erode(mask.copy(), kernel, iterations=1)
    else:
        mask_sampled = mask

    if renumber:
        iter_map, fruit_id_map = renumber_fruit_locule_map(fruit_locule_map)
    else:
        iter_map = fruit_locule_map
        fruit_id_map = {k: k for k in fruit_locule_map.keys()}

    if fruit_id is not None:
        if fruit_id not in iter_map:
            raise ValueError(f"fruit_id: {fruit_id} not found.")
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
        roi_img  = img          [y:y+h, x:x+w]
        roi_mask = mask_sampled [y:y+h, x:x+w]   # eroded mask for sampling
        roi_gray = gray_full    [y:y+h, x:x+w]

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

        n_valid    = int(valid.sum())
        bgr_pixels = roi_img[valid].astype(np.float32)

        record: Dict[str, np.ndarray] = {
            'fruit_id': np.full(n_valid, fid, dtype=np.int32),
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
            record['H'] = h_norm
            record['S'] = s_norm
            record['V'] = v_norm

        all_records.append(pd.DataFrame(record))

    if not all_records:
        return pd.DataFrame()

    df_pixels = pd.concat(all_records, ignore_index=True)

    if sample_size is not None and len(df_pixels) > sample_size:
        df_pixels = df_pixels.sample(n=sample_size, random_state=42).reset_index(drop=True)

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

def plot_color_scatter(
    img: np.ndarray,
    mask: np.ndarray,
    contours: List[np.ndarray],
    fruit_map: Dict[int, List[int]],
    fruit_id: Optional[int] = None,
    color_space: Optional[str] = None,
    sample_size: int = 10000,
    plot_size: Tuple[int, int] = (18, 5),
    renumber: bool = True,
    dark_threshold: int = 15,
    title_font_size: Optional[float] = None,
    xlabel_font_size: Optional[float] = None,
    ylabel_font_size: Optional[float] = None,
    axes_font_size: Optional[float] = None,
    img_name: Optional[str] = None,
    alpha: float = 0.6,
    erosion_px: int = 3
) -> None:
    """

    """
    # Determinate channel information
    _space_channels = {
        'rgb': [('R', 'R (0-255)'),      ('G', 'G (0-255)'),          ('B', 'B (0-255)')],
        'lab': [('L', 'L (0-100)'),      ('a', 'a (-128-127)'),       ('b', 'b (-128-127)')],
        'hsv': [('H', 'Hue (0-360)'),    ('S', 'Saturation (0-100)'), ('V', 'Value (0-100)')],
    }
    valid_spaces = set(_space_channels.keys())

    # Resolve which spaces to plot
    if color_space is None:
        # fixed order
        spaces_to_plot = ['rgb', 'lab', 'hsv']
    else:
        cs = color_space.lower().replace(' ', '')
        if ',' in cs or cs == 'all':
            raise ValueError(
                "plot_color_scatter() does not accept comma-separated spaces. "
                "Use color_space=None to plot all spaces."
            )
        if cs not in valid_spaces:
            raise ValueError(f"Invalid color_space: '{cs}'. Valid: {valid_spaces}")
        spaces_to_plot = [cs]

    # Extract pixels for all needed spaces in one call
    extract_cs = ','.join(spaces_to_plot)
    df_px = get_fruit_color_samples(
        img=img, mask=mask, contours=contours,
        fruit_locule_map=fruit_map,
        fruit_id=fruit_id,
        color_space=extract_cs,
        sample_size=sample_size,
        renumber=renumber,
        dark_threshold=dark_threshold,
        jitter=True,
        jitter_strength=0.4,
        erosion_px = erosion_px
    )

    if df_px.empty:
        print("No pixels found for the selected fruit(s).")
        return

    point_colors = np.clip(df_px[['R_norm', 'G_norm', 'B_norm']].values, 0.0, 1.0)

    def _circ_mean_std(vals):
        rad = np.deg2rad(vals)
        s, c = np.mean(np.sin(rad)), np.mean(np.cos(rad))
        mean_deg = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
        R = np.clip(np.sqrt(s**2 + c**2), 1e-12, 1.0)
        return mean_deg, np.rad2deg(np.sqrt(-2.0 * np.log(R)))

    def _stat_label(col):
        vals = df_px[col].values
        if col == 'H':
            m, s = _circ_mean_std(vals)
            return f'Circ. mean={m:.1f}  Circ. std={s:.1f}'
        return f'mean={np.mean(vals):.1f}  std={np.std(vals):.1f}'

    # Ensure 3 cols for 3 channel spaces
    n_cols = 3   
    n_rows = len(spaces_to_plot)

    # Font sizes (scale automatically if not provided)
    width     = plot_size[0]
    _title_fs = title_font_size  if title_font_size  is not None else int(np.clip(8 + width * 0.6, 10, 24))
    _label_fs = xlabel_font_size if xlabel_font_size is not None else int(np.clip(6 + width * 0.4,  8, 18))
    _ylbl_fs  = ylabel_font_size if ylabel_font_size is not None else _label_fs
    _tick_fs  = axes_font_size   if axes_font_size   is not None else int(np.clip(5 + width * 0.3,  7, 14))
    
    n_fruits = df_px['fruit_id'].nunique()
    cs_label = color_space.upper() if color_space else 'ALL'

    if img_name:
            title = (
                f"All fruits (n={n_fruits}) in {img_name} ({len(df_px):,} pixels)"
                if fruit_id is None else
                f"Fruit ID {fruit_id} in {img_name} ({len(df_px):,} pixels)"
            )
    else:
        title = (
            f"All fruits (n={n_fruits}) ({len(df_px):,} pixels)"
            if fruit_id is None else
            f"Fruit ID {fruit_id} ({len(df_px):,} pixels)"
        )

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(plot_size[0], plot_size[1] * n_rows),
        squeeze=False
    )

    fig.suptitle(title, fontweight='bold', fontsize=_title_fs, y=1.01)

    for row_idx, space in enumerate(spaces_to_plot):
        channels = _space_channels[space]
        pairs = [
            (channels[i], channels[j])
            for i in range(len(channels))
            for j in range(i + 1, len(channels))
        ]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(pairs):
                ax.set_visible(False)
                continue

            (cx, lx), (cy, ly) = pairs[col_idx]

            ax.scatter(
                df_px[cx].values, df_px[cy].values,
                c=point_colors, alpha=alpha, s=10,
                linewidths=0, rasterized=True
            )
            ax.set_xlabel(f'{lx}\n{_stat_label(cx)}', fontsize=_label_fs)
            ax.set_ylabel(f'{ly}\n{_stat_label(cy)}', fontsize=_ylbl_fs)
            ax.set_title(f'{cx} vs {cy}', fontweight='bold', fontsize=_title_fs)
            ax.tick_params(axis='both', labelsize=_tick_fs)
            ax.grid(alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)

            # Space label on the leftmost column of each row
            if col_idx == 0:
                ax.annotate(space.upper(), xy=(-0.25, 0.5),
                            xycoords='axes fraction', fontsize=_title_fs,
                            fontweight='bold', color='#333333',
                            va='center', ha='center', rotation=90)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Create correlation plots for color channels
# ============================================================================

def plot_color_correlation(
    df: pd.DataFrame,
    fruit_id: Optional[int] = None,
    color_space: Optional[str] = None, 
    method: str = 'pearson',
    plot_size: Tuple[int, int] = (8, 6),
    cluster: bool = False,
    annotate: bool = True,
    triangle: str = 'full',
    plot: bool = True,
    # Font sizes
    title_font_size: Optional[float] = None, 
    axes_font_size: Optional[float] = None,
    annot_font_size: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute and plot correlation matrix between color channels.

    """


    # Validate
    method = method.lower()
    if method not in {'pearson', 'spearman'}:
        raise ValueError(f"method: {method} must be 'pearson' or 'spearman'")

    if triangle not in {'full', 'upper', 'lower'}:
        raise ValueError(f"triangle: {triangle} must be 'full', 'upper', or 'lower'")

    # Determine valid color spaces
    valid_spaces = {'rgb', 'lab', 'hsv', 'gray'}
    _sentinel    = {'rgb': 'R_0', 'lab': 'L_0', 'hsv': 'H_0', 'gray': 'Gray_0'}

    if color_space is None:
        spaces = {s for s, col in _sentinel.items() if col in df.columns}
        if not spaces:
            raise ValueError(
                "Could not detect any color space in the DataFrame. "
                "Expected at least one of: " + str(list(_sentinel.values()))
            )
    else:
        cs = color_space.lower().replace(' ', '')
        spaces = valid_spaces if cs == 'all' else set(cs.split(','))
        if not spaces.issubset(valid_spaces):
            raise ValueError(f"Invalid color_space: {spaces - valid_spaces}. Valid: {valid_spaces}")

    # Select rows
    if fruit_id is not None:
        subset = df[df['fruit_id'] == fruit_id]
        if subset.empty:
            raise ValueError(f"fruit_id: {fruit_id} not found.")
        if len(subset) < 2:
            print("Warning: correlation needs more than 1 fruit. Showing all fruits instead.")
            subset = df
    else:
        subset = df

    image_name = subset['image_name'].iloc[0] if 'image_name' in subset.columns else ''
    n = len(subset)

    # Channel definitions
    channel_defs = []
    if 'rgb'  in spaces: channel_defs += [('R', 0, 255),    ('G', 0, 255),    ('B', 0, 255)]
    if 'lab'  in spaces: channel_defs += [('L', 0, 100),    ('a', -128, 127), ('b', -128, 127)]
    if 'hsv'  in spaces: channel_defs += [('H', 0, 359),    ('S', 0, 100),    ('V', 0, 100)]
    if 'gray' in spaces: channel_defs += [('Gray', 0, 255)]

    # Weighted mean per channel per fruit
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

    means_df    = pd.DataFrame(means_data)
    corr_matrix = means_df.corr(method=method)
   
    # Optional clustering
    if cluster:
        dist_matrix = 1 - corr_matrix.abs().values
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix    = (dist_matrix + dist_matrix.T) / 2
        condensed      = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method='average')
        order          = leaves_list(linkage_matrix)
        channels_ord   = [corr_matrix.columns[i] for i in order]
        corr_matrix    = corr_matrix.loc[channels_ord, channels_ord]

    # Font sizes (scale automatically if not provided)
    width      = plot_size[0]
    _title_fs  = title_font_size if title_font_size is not None else int(np.clip(8 + width * 0.6, 10, 24))
    _tick_fs   = axes_font_size  if axes_font_size  is not None else int(np.clip(5 + width * 0.3,  7, 14))
    _annot_fs  = annot_font_size if annot_font_size is not None else int(np.clip(4 + width * 0.4,  7, 16))
   
    # Triangle mask
    channels    = list(corr_matrix.columns)
    n_ch        = len(channels)
    corr_values = corr_matrix.values.copy()

    if triangle != 'full':
        for i in range(n_ch):
            for j in range(n_ch):
                if triangle == 'upper' and i > j:
                    corr_values[i, j] = np.nan
                elif triangle == 'lower' and i < j:
                    corr_values[i, j] = np.nan

    # Figure
    if plot:
        fig, ax = plt.subplots(figsize=plot_size)

        masked_values = np.ma.masked_invalid(corr_values)
        im = ax.imshow(masked_values, cmap='RdBu_r', vmin=-1, vmax=1,
                    aspect='auto', zorder=1)
        ax.set_facecolor('white')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"({method.capitalize()} correlation) ",)

        if annotate:
            for i in range(n_ch):
                for j in range(n_ch):
                    val = corr_values[i, j]
                    if np.isnan(val):
                        continue
                    text_color = 'white' if abs(val) > 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=_annot_fs, color=text_color,
                            fontweight='bold', zorder=10)

        ax.set_xticks(range(n_ch))
        ax.set_yticks(range(n_ch))
        ax.set_xticklabels(channels, fontsize=_tick_fs, rotation=45, ha='right')
        ax.set_yticklabels(channels, fontsize=_tick_fs)
        ax.set_xticks(np.arange(n_ch + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_ch + 1) - 0.5, minor=True)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.grid(which='minor', color='#cccccc', linewidth=1.0, zorder=5)

        ax.set_title(
            f"All fruits (n={n}) in {image_name} ",
            fontweight='bold', fontsize=_title_fs, pad=12
        )

        plt.tight_layout()
        plt.show()

    return corr_matrix