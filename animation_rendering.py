from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

# ====== UMAP Animation ======
class UMAPAnimationRenderer:
    """
    Renders a frame-by-frame animation of YouTube watch history using the
    circular category layout from clustering_video_types.py.

    Each category's centroid is anchored on the unit-circle perimeter.
    Points are coloured by subcluster, channel, or category, with optional
    shape-mode encoding using distinct matplotlib markers per shape value.

    Two rendering passes per frame:
      - Noise pass  : sub_label == -1, wide trailing window, dim grey, no glow
      - Cluster pass: labelled points, narrow window, layered glow bloom rings
                      In shape_mode, each unique marker value is a separate scatter call.
    """

    # Default glow ring (size, base_alpha) pairs.
    DEFAULT_GLOW_LAYERS = [
        (130, 0.004), (105, 0.007), (83, 0.011), (64, 0.018),
        (49,  0.030), (37,  0.050), (27,  0.080), (19,  0.130), (13, 0.200),
    ]

    # Per-marker size scale (matches _markers in clustering_video_types.py)
    MARKER_SIZE_SCALES = {
        'o': 1.0, 's': 0.9, '^': 1.2, 'D': 0.85, 'v': 1.2,
        'p': 0.9, 'h': 0.9, 'P': 0.85, '*': 3.0,  'X': 0.85,
    }

    def __init__(
        self,
        # ── Timing ────────────────────────────────────────────────────────────
        fps: int = 30,
        seconds_per_day: float = 0.2,
        # ── Trail window sizes ─────────────────────────────────────────────────
        window_size: int = 7,               # days cluster points remain visible
        noise_window_multiplier: int = 4,   # noise window = window_size × noise_window_multiplier
        # ── Figure layout ─────────────────────────────────────────────────────
        figsize: tuple = (19.2, 10.8),
        dpi: int = 300,
        # ── Circular chart viewport ───────────────────────────────────────────
        chart_limit: float = 1.35,          # axis range: [-chart_limit, chart_limit]
        show_circle: bool = True,           # draw faint unit-circle boundary
        show_labels: bool = True,           # draw category labels at perimeter
        show_legend: bool = False,          # draw category colour legend
        # ── Shape mode ────────────────────────────────────────────────────────
        shape_mode: bool = True,            # True → use shape_marker for each scatter pass
        # ── Glow effect ───────────────────────────────────────────────────────
        glow_size: float = 1.0,             # global scale multiplier for all glow sizes
        glow_layers: list = None,           # list of (size, alpha) tuples; None = DEFAULT_GLOW_LAYERS
        core_point_size: float = 5.5,      # solid dot drawn on top of glow rings
        # ── Cluster alpha decay ───────────────────────────────────────────────
        cluster_alpha_decay: float = 2.0,   # power > 1 → fast initial drop, flatter tail
        # ── Noise point styling ───────────────────────────────────────────────
        noise_point_size: float = 12.0,
        noise_base_alpha: float = 0.45,     # max alpha for noise points (dim background)
        noise_alpha_decay: float = 0.4,     # power < 1 → slow decay (noise lingers)
        noise_color: str = 'grey',
        noise_fallback_color: tuple = (0.6, 0.6, 0.6, 1.0),
        # ── Duration-based noise point scaling ────────────────────────────────
        scale_by_duration: bool = False,
        noise_duration_min_size: float = 4.0,
        noise_duration_max_size: float = 30.0,
    ):
        # ── Timing ────────────────────────────────────────────────────────────
        self.fps = fps
        self.seconds_per_day = seconds_per_day
        self.frames_per_day = max(1, round(fps * seconds_per_day))

        # ── Trail windows ─────────────────────────────────────────────────────
        self.window_size = window_size
        self.noise_window_multiplier = noise_window_multiplier
        self.noise_window = window_size * noise_window_multiplier

        # ── Figure layout ─────────────────────────────────────────────────────
        self.figsize = figsize
        self.dpi = dpi

        # ── Circular chart ────────────────────────────────────────────────────
        self.chart_limit = chart_limit
        self.show_circle = show_circle
        self.show_labels = show_labels
        self.show_legend = show_legend
        self.shape_mode  = shape_mode

        # ── Glow effect ───────────────────────────────────────────────────────
        self.glow_size = glow_size
        raw_layers = glow_layers if glow_layers is not None else self.DEFAULT_GLOW_LAYERS
        self._glow_layers = [(s * glow_size, a) for s, a in raw_layers]
        self._core_point_size = core_point_size * glow_size

        # ── Alpha decay ───────────────────────────────────────────────────────
        self.cluster_alpha_decay = cluster_alpha_decay

        # ── Noise styling ─────────────────────────────────────────────────────
        self.noise_point_size    = noise_point_size
        self.noise_base_alpha    = noise_base_alpha
        self.noise_alpha_decay   = noise_alpha_decay
        self.noise_color         = noise_color
        self.noise_fallback_color = noise_fallback_color
        self.scale_by_duration   = scale_by_duration
        self.noise_duration_min_size = noise_duration_min_size
        self.noise_duration_max_size = noise_duration_max_size

        # ── Internal state ────────────────────────────────────────────────────
        self.fig = None
        self.ax  = None
        self._days            = None  # list[Period]
        self._day_ordered_ids = None  # list[list[str]]
        self._day_umap_data   = None  # list[(coords_sub, colors, noise, sizes, markers)]
        self._shape_marker    = None  # dict: shape_value → marker char, set in _setup_figure

    # ──────────────────────────────────────────────────────────────────────────
    # Static helper: circular layout
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_circular_layout(umap_embeddings, umap_cats):
        """
        Transform raw UMAP coordinates into a circular layout where each
        category's centroid is anchored on the unit-circle perimeter.

        Replicates the layout logic from clustering_video_types.py lines 364–405.

        Parameters
        ----------
        umap_embeddings : DataFrame  (index=video_id, cols=[0, 1])
        umap_cats       : Series     (index=video_id) — category per video

        Returns
        -------
        coords_df     : DataFrame  (index=video_id, cols=[0, 1]) — circular coords
        cat_order     : list       — categories sorted by count (dense first)
        cat_angles    : dict       — category → angle in radians on unit circle
        raw_centroids : dict       — category → mean UMAP coord before transform
        """
        cats_series = umap_cats.loc[umap_embeddings.index]
        cat_order   = cats_series.value_counts().index.tolist()
        n_cats      = len(cat_order)
        cat_angles  = {cat: 2 * np.pi * i / n_cats for i, cat in enumerate(cat_order)}

        raw_centroids = {
            cat: umap_embeddings.loc[cats_series == cat].values.mean(axis=0)
            for cat in cat_order
        }

        coords_circ = np.empty_like(umap_embeddings.values, dtype=float)
        for cat in cat_order:
            idx   = cats_series[cats_series == cat].index
            pts   = umap_embeddings.loc[idx].values
            cent  = raw_centroids[cat]
            angle = cat_angles[cat]

            rel = pts - cent
            raw_angle = np.arctan2(cent[1], cent[0])
            rot = angle - raw_angle
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            rotated = np.column_stack([
                rel[:, 0] * cos_r - rel[:, 1] * sin_r,
                rel[:, 0] * sin_r + rel[:, 1] * cos_r,
            ])

            spread = np.abs(rel).max() or 1.0
            rotated /= spread * 2.0  # compress inward (target_r=1 → /spread * (2/1))

            tx = np.cos(angle)
            ty = np.sin(angle)
            row_mask = umap_embeddings.index.get_indexer(idx)
            coords_circ[row_mask] = rotated + np.array([tx, ty])

        coords_df = pd.DataFrame(coords_circ, index=umap_embeddings.index, columns=[0, 1])
        return coords_df, cat_order, cat_angles, raw_centroids

    # ──────────────────────────────────────────────────────────────────────────
    # Figure setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_figure(self, cat_colours=None, cat_angles=None, cat_order=None, shape_marker=None):
        """
        Create the figure, set fixed circular viewport, and draw static
        decorations (unit-circle ring, category labels, legend).

        Parameters
        ----------
        cat_colours  : dict|None  — category → RGB colour, for perimeter labels
        cat_angles   : dict|None  — category → angle (radians), for label placement
        cat_order    : list|None  — ordered category list
        shape_marker : dict|None  — shape_value → matplotlib marker char
        """
        self._shape_marker = shape_marker

        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.ax.set_facecolor('#0a0a0a')
        self.fig.patch.set_facecolor('#0a0a0a')
        self.fig.patch.set_alpha(1)
        self.ax.patch.set_alpha(0)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        lim = self.chart_limit
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)

        # ── Static decorations (drawn once; not cleared by _draw_frame) ────────
        if self.show_circle:
            circle = plt.Circle((0, 0), 1.0, color='white', fill=False, lw=0.5, alpha=0.15)
            self.ax.add_patch(circle)

        if self.show_labels and cat_colours and cat_angles and cat_order:
            for cat in cat_order:
                angle = cat_angles[cat]
                lx = 1.12 * np.cos(angle)
                ly = 1.12 * np.sin(angle)
                ha = ('left'  if np.cos(angle) >  0.1 else
                      'right' if np.cos(angle) < -0.1 else 'center')
                self.ax.text(lx, ly, cat, color=cat_colours[cat], fontsize=7,
                             ha=ha, va='center', fontweight='bold')

        if self.show_legend and cat_colours and cat_order:
            handles = [Patch(color=cat_colours[cat], label=cat)
                       for cat in cat_order if cat in cat_colours]
            self.ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                           ncol=4, frameon=False, labelcolor='white', fontsize=7)

    # ──────────────────────────────────────────────────────────────────────────
    # Data pre-caching
    # ──────────────────────────────────────────────────────────────────────────

    def _precache_days(self, df, coords_df, colors_series, is_noise_series,
                       marker_series=None, duration_series=None):
        """
        Pre-build per-day data arrays to avoid repeated lookups inside the
        hot render loop.

        Parameters
        ----------
        df               : DataFrame   — watch history with 'date' and 'id' columns
        coords_df        : DataFrame   — circular coords (index=video_id, cols=[0,1])
        colors_series    : Series      — per-video colour tuple (index=video_id)
        is_noise_series  : Series      — bool, True = noise / sub_label==-1
        marker_series    : Series|None — shape value per video (index=video_id)
        duration_series  : Series|None — duration in seconds (index=video_id)
        """
        self._days            = sorted(df['date'].dt.to_period('D').unique())
        self._day_ordered_ids = []
        self._day_umap_data   = []

        _dur_sizes = (self._compute_duration_sizes(duration_series)
                      if self.scale_by_duration and duration_series is not None else None)

        for day in tqdm(self._days, desc='Pre-caching days'):
            day_df = df[df['date'].dt.to_period('D') == day].sort_values('date')

            seen, ordered = set(), []
            for vid_id in day_df['id']:
                if vid_id not in seen and vid_id in coords_df.index:
                    seen.add(vid_id)
                    ordered.append(vid_id)
            self._day_ordered_ids.append(ordered)

            if ordered:
                coords_sub = coords_df.loc[ordered]
                noise      = is_noise_series.reindex(ordered).fillna(True).values.astype(bool)
                colors     = np.array([colors_series[v] for v in ordered], dtype=float)
                markers    = marker_series.reindex(ordered).values if marker_series is not None else None
                sizes      = (_dur_sizes.reindex(ordered).fillna(self.noise_point_size).values
                              if _dur_sizes is not None else None)
            else:
                coords_sub = coords_df.iloc[:0]
                noise      = np.empty(0, dtype=bool)
                colors     = np.empty((0, 3), dtype=float)
                markers    = np.empty(0, dtype=object) if marker_series is not None else None
                sizes      = np.empty(0) if _dur_sizes is not None else None

            self._day_umap_data.append((coords_sub, colors, noise, sizes, markers))

    # ──────────────────────────────────────────────────────────────────────────
    # Alpha decay functions
    # ──────────────────────────────────────────────────────────────────────────

    def _get_alpha(self, offset: int) -> float:
        """Alpha for clustered points: power-law decay over window_size days."""
        if offset == 0:
            return 1.0
        if 1 <= offset < self.window_size:
            return (1.0 - offset / self.window_size) ** self.cluster_alpha_decay
        return 0.0

    def _get_noise_alpha(self, offset: int) -> float:
        """Alpha for noise points: slow decay over the wider noise_window."""
        if offset == 0:
            return self.noise_base_alpha
        if 1 <= offset < self.noise_window:
            return self.noise_base_alpha * (1.0 - offset / self.noise_window) ** self.noise_alpha_decay
        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Duration size scaling
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_duration_sizes(self, duration_series):
        """
        Map video durations to point sizes via a sigmoid on log-duration.

        Parameters
        ----------
        duration_series : Series  (index=video_id, values=duration in seconds)
        """
        dur = pd.to_numeric(duration_series, errors='coerce')
        median = dur.median()
        if np.isnan(median) or median <= 0:
            median = 1.0
        dur_filled = dur.fillna(median).clip(lower=1)

        log_dur = np.log(dur_filled.values)
        log_med = np.log(median)
        log_std = log_dur.std()
        if log_std == 0:
            log_std = 1.0

        k   = 0.8 / log_std
        sig = 1.0 / (1.0 + np.exp(-k * (log_dur - log_med)))
        sizes = self.noise_duration_min_size + (self.noise_duration_max_size - self.noise_duration_min_size) * sig
        return pd.Series(sizes, index=duration_series.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Frame drawing
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_frame(self, day_idx: int, n_visible: int):
        """
        Clear and redraw all scatter layers for one animation frame.

        Two-pass rendering:
          Pass 1 — Noise (sub_label == -1): dim grey over a wide trailing window; no glow.
          Pass 2 — Clusters: full-colour points with layered glow bloom rings over a
                   shorter window. In shape_mode each marker value gets its own scatter
                   call so markers are rendered correctly.

        Parameters
        ----------
        day_idx   : int  — index of the current day in self._days
        n_visible : int  — how many of today's watches to reveal
        """
        # Remove all scatter artists from the previous frame
        while self.ax.collections:
            self.ax.collections[-1].remove()

        # ── Pass 1: noise points (background) ─────────────────────────────────
        for d_idx in range(max(0, day_idx - self.noise_window + 1), day_idx + 1):
            offset = day_idx - d_idx
            alpha  = self._get_noise_alpha(offset)
            if alpha <= 0:
                continue

            coords_sub, colors, noise, sizes, _ = self._day_umap_data[d_idx]
            n = n_visible if d_idx == day_idx else len(coords_sub)
            if n == 0 or not noise[:n].any():
                continue

            nm = noise[:n]
            noise_s = sizes[:n][nm] if (self.scale_by_duration and sizes is not None) else self.noise_point_size
            self.ax.scatter(
                coords_sub[0].values[:n][nm],
                coords_sub[1].values[:n][nm],
                s=noise_s, alpha=alpha,
                color=self.noise_color, linewidths=0, zorder=1,
            )

        # ── Pass 2: clustered points with glow bloom ───────────────────────────
        for d_idx in range(max(0, day_idx - self.window_size + 1), day_idx + 1):
            offset = day_idx - d_idx
            alpha  = self._get_alpha(offset)
            if alpha <= 0:
                continue

            coords_sub, colors, noise, sizes, markers = self._day_umap_data[d_idx]
            n = n_visible if d_idx == day_idx else len(coords_sub)
            non_noise = ~noise[:n]
            if n == 0 or not non_noise.any():
                continue

            xu = coords_sub[0].values[:n][non_noise]
            yu = coords_sub[1].values[:n][non_noise]
            pc = colors[:n][non_noise]

            cluster_s = (sizes[:n][non_noise]
                         if (self.scale_by_duration and sizes is not None)
                         else self._core_point_size)
            scale = cluster_s / self._core_point_size

            if self.shape_mode and markers is not None and self._shape_marker is not None:
                # Per-marker scatter so each shape is rendered with the correct glyph
                mvals = markers[:n][non_noise]
                for mval in dict.fromkeys(mvals):  # preserve encounter order
                    marker_char = self._shape_marker.get(mval, 'o')
                    ms_scale    = self.MARKER_SIZE_SCALES.get(marker_char, 1.0)
                    mm = mvals == mval
                    cs = (cluster_s[mm] if np.ndim(cluster_s) > 0 else cluster_s) * ms_scale
                    sc = cs / self._core_point_size

                    # Glow rings (outermost → innermost)
                    for size, ring_alpha in self._glow_layers:
                        self.ax.scatter(
                            xu[mm], yu[mm], s=size * sc, alpha=ring_alpha * alpha,
                            color=pc[mm], linewidths=1, zorder=3, marker=marker_char,
                        )
                    # Solid core dot
                    self.ax.scatter(
                        xu[mm], yu[mm], s=cs, alpha=alpha,
                        color=pc[mm], linewidths=0, zorder=4, marker=marker_char,
                    )
            else:
                # Single scatter pass (no shape mode)
                for size, ring_alpha in self._glow_layers:
                    self.ax.scatter(
                        xu, yu, s=size * scale, alpha=ring_alpha * alpha,
                        color=pc, linewidths=1, zorder=3,
                    )
                self.ax.scatter(
                    xu, yu, s=cluster_s, alpha=alpha,
                    color=pc, linewidths=0, zorder=4,
                )

        # Re-enforce viewport
        lim = self.chart_limit
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def render(
        self,
        df,
        coords_df,
        colors_series,
        is_noise_series,
        output_dir,
        marker_series=None,
        shape_marker=None,
        duration_series=None,
        cat_colours=None,
        cat_angles=None,
        cat_order=None,
    ):
        """
        Run the full render loop and write one SVG per frame to output_dir.

        Parameters
        ----------
        df               : DataFrame   — watch history with 'date' (datetime) and 'id' columns
        coords_df        : DataFrame   — circular layout coords (index=video_id, cols=[0,1]);
                                         compute with UMAPAnimationRenderer.compute_circular_layout()
        colors_series    : Series      — per-video colour tuple (index=video_id)
        is_noise_series  : Series      — bool, True = noise / sub_label==-1 (index=video_id)
        output_dir       : str         — destination dir; cleared and recreated each run
        marker_series    : Series|None — shape value per video (e.g. sub_label); index=video_id
        shape_marker     : dict|None   — maps shape_value → matplotlib marker char
        duration_series  : Series|None — duration in seconds per video (for scale_by_duration)
        cat_colours      : dict|None   — category → colour, for perimeter labels
        cat_angles       : dict|None   — category → angle (radians), for label placement
        cat_order        : list|None   — ordered list of categories
        """
        print(f'frames_per_day={self.frames_per_day}  window_size={self.window_size}  noise_window={self.noise_window}')
        print(f'Activities with embeddings: {len(df[df["id"].isin(coords_df.index)])}')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        self._setup_figure(cat_colours, cat_angles, cat_order, shape_marker)
        self._precache_days(df, coords_df, colors_series, is_noise_series,
                            marker_series, duration_series)

        frame_idx = 0
        for day_idx, day in tqdm(enumerate(self._days), total=len(self._days), desc='Rendering'):
            n_today = len(self._day_ordered_ids[day_idx])
            for f in range(self.frames_per_day):
                n_visible = round((f + 1) / self.frames_per_day * n_today)
                self._draw_frame(day_idx, n_visible)
                self.fig.savefig(f'{output_dir}/frame_{frame_idx:05d}.svg', transparent=True)
                frame_idx += 1
            print(f'Rendered {day} ({frame_idx} frames total)')

        plt.close(self.fig)
        print(f'Done — {frame_idx} frames at {self.fps}fps → {frame_idx / self.fps:.1f}s saved to {output_dir}/')

    def sample_frame(
        self,
        df,
        coords_df,
        colors_series,
        is_noise_series,
        day_idx: int = 0,
        n_visible: int = None,
        show: bool = True,
        save_path: str = None,
        preview_dpi: int = 300,
        marker_series=None,
        shape_marker=None,
        duration_series=None,
        cat_colours=None,
        cat_angles=None,
        cat_order=None,
    ):
        """
        Render and display a single frame for quick visual iteration.

        Lazily initialises figure and cache on first call; subsequent calls
        reuse the cache so parameter tweaks (glow, alpha, etc.) are fast.
        Call reset() first if you want to re-initialise with different data.

        Parameters
        ----------
        df               : DataFrame
        coords_df        : DataFrame   — circular layout coords (index=video_id, cols=[0,1])
        colors_series    : Series      — per-video colour tuple (index=video_id)
        is_noise_series  : Series      — bool, True = noise (index=video_id)
        day_idx          : int         — which day to render (0-indexed)
        n_visible        : int|None    — how many watches to show; defaults to all
        show             : bool        — whether to call plt.show()
        save_path        : str|None    — if given, saves the frame to this path
        preview_dpi      : int         — DPI for plt.show() preview (lower = faster)
        marker_series    : Series|None
        shape_marker     : dict|None
        duration_series  : Series|None
        cat_colours      : dict|None
        cat_angles       : dict|None
        cat_order        : list|None
        """
        if self._days is None:
            original_dpi = self.dpi
            self.dpi = preview_dpi
            self._setup_figure(cat_colours, cat_angles, cat_order, shape_marker)
            self.dpi = original_dpi
            self._precache_days(df, coords_df, colors_series, is_noise_series,
                                marker_series, duration_series)

        if not self._days:
            raise RuntimeError('No data available — df has no dates overlapping with coords_df index.')

        day_idx   = min(max(day_idx, 0), len(self._days) - 1)
        n_visible = n_visible if n_visible is not None else len(self._day_ordered_ids[day_idx])

        print(f'Sample frame: day={self._days[day_idx]}  day_idx={day_idx}  n_visible={n_visible}')
        self._draw_frame(day_idx, n_visible)

        if save_path:
            self.fig.savefig(save_path, transparent=True, dpi=self.dpi)
            print(f'Saved to {save_path}')

        if show:
            plt.show()

    def frame_to_date(self, frame: int):
        """
        Return the calendar date that a given frame number corresponds to.

        Parameters
        ----------
        frame : int  — absolute frame index (0-based), as used in output filenames
        """
        if self._days is None:
            raise RuntimeError('No data cached — call render() or sample_frame() first.')
        day_idx = min(frame // self.frames_per_day, len(self._days) - 1)
        return self._days[day_idx].to_timestamp().date()

    def reset(self):
        """
        Clear all cached data and close the figure.

        Call this before re-using the renderer with a different dataset or if
        you want sample_frame to re-run setup with updated parameters.
        """
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = self.ax = None
        self._days = self._day_ordered_ids = self._day_umap_data = None
        self._shape_marker = None
