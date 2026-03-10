from matplotlib import pyplot as plt
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

# ====== UMAP Animation ======
class UMAPAnimationRenderer:
    """
    Renders a frame-by-frame animation of UMAP-embedded YouTube watch history.

    Each calendar day is spread across `frames_per_day` frames, with videos
    revealed in chronological watch order within that day. A rolling glow trail
    fades older days using configurable alpha decay curves.

    Two separate rendering passes per frame:
      - Noise pass  : HDBSCAN label == -1, wide window, dim grey, no glow
      - Cluster pass: labelled points, narrow window, layered glow bloom
    """

    # Default glow ring (size, base_alpha) pairs.
    # Ring sizes decrease toward the core; alphas increase outward — outer rings
    # are barely visible, creating a soft gaussian-like bloom rather than hard rings.
    DEFAULT_GLOW_LAYERS = [
        (130, 0.004), (105, 0.007), (83, 0.011), (64, 0.018),
        (49,  0.030), (37,  0.050), (27,  0.080), (19,  0.130), (13, 0.200),
    ]

    def __init__(
        self,
        # ── Timing ────────────────────────────────────────────────────────────
        fps: int = 30,
        seconds_per_day: float = 0.2,
        # ── Trail window sizes ─────────────────────────────────────────────────
        window_size: int = 7,               # days cluster points remain visible
        noise_window_multiplier: int = 4,   # noise window = window_size × noise_window_multiplier
        # ── Figure layout ─────────────────────────────────────────────────────
        figsize: tuple = (19.2, 10.8),           # inches; (16,9) @ dpi=300 → 4800×2700px
        dpi: int = 150,
        # ── Axis viewport ─────────────────────────────────────────────────────
        axis_padding: float = 0.5,          # blank margin around embedding extent
        axis_zoom: float = 1.35,            # crop factor > 1 zooms in slightly
        # ── Glow effect ───────────────────────────────────────────────────────
        glow_size: float = 1.0,             # global scale multiplier for all glow sizes
        glow_layers: list = None,           # list of (size, alpha) tuples; None = DEFAULT_GLOW_LAYERS
        core_point_size: float = 20.0,      # solid dot drawn on top of glow rings
        # ── Cluster alpha decay ───────────────────────────────────────────────
        cluster_alpha_decay: float = 2.0,   # power > 1 → fast initial drop, flatter tail
        # ── Noise point styling ───────────────────────────────────────────────
        noise_point_size: float = 12.0,
        noise_base_alpha: float = 0.45,     # max alpha for noise points (dim background)
        noise_alpha_decay: float = 0.4,     # power < 1 → slow decay (noise lingers)
        noise_color: str = 'grey',
        noise_fallback_color: tuple = (0.6, 0.6, 0.6, 1.0),  # color for unlabelled clusters
        # ── Duration-based noise point scaling ────────────────────────────────
        scale_by_duration: bool = False,  # enable sigmoid size scaling for noise points
        noise_duration_min_size: float = 4.0,   # size for shortest videos
        noise_duration_max_size: float = 30.0,  # size for longest videos
    ):
        # ── Timing ────────────────────────────────────────────────────────────
        self.fps = fps
        self.seconds_per_day = seconds_per_day
        # frames_per_day: how many PNG frames map to one calendar day
        self.frames_per_day = max(1, round(fps * seconds_per_day))

        # ── Trail windows ─────────────────────────────────────────────────────
        self.window_size = window_size
        self.noise_window_multiplier = noise_window_multiplier
        # Noise points use a much wider window to provide persistent background texture
        self.noise_window = window_size * noise_window_multiplier

        # ── Figure layout ─────────────────────────────────────────────────────
        self.figsize = figsize
        self.dpi = dpi

        # ── Axis viewport ─────────────────────────────────────────────────────
        self.axis_padding = axis_padding
        self.axis_zoom = axis_zoom

        # ── Glow effect ───────────────────────────────────────────────────────
        self.glow_size = glow_size
        raw_layers = glow_layers if glow_layers is not None else self.DEFAULT_GLOW_LAYERS
        # Pre-scale ring sizes once here so _draw_frame doesn't multiply every call
        self._glow_layers = [(s * glow_size, a) for s, a in raw_layers]
        self._core_point_size = core_point_size * glow_size

        # ── Alpha decay ───────────────────────────────────────────────────────
        self.cluster_alpha_decay = cluster_alpha_decay

        # ── Noise styling ─────────────────────────────────────────────────────
        self.noise_point_size = noise_point_size
        self.noise_base_alpha = noise_base_alpha
        self.noise_alpha_decay = noise_alpha_decay
        self.noise_color = noise_color
        self.noise_fallback_color = noise_fallback_color
        self.scale_by_duration = scale_by_duration
        self.noise_duration_min_size = noise_duration_min_size
        self.noise_duration_max_size = noise_duration_max_size

        # ── Internal state (populated lazily during render / sample_frame) ────
        self.fig = None
        self.ax  = None
        self._days            = None  # list[Period] — one per calendar day
        self._day_ordered_ids = None  # list[list[str]] — chronological IDs per day
        self._day_umap_data   = None  # list[(umap_df, colors_arr, is_noise_arr)]
        # Axis limit helpers, set by _setup_figure
        self._x_c = self._x_h = self._y_c = self._y_h = None

    # ──────────────────────────────────────────────────────────────────────────
    # Figure setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_figure(self, umapped):
        """
        Create a clean matplotlib figure and lock axis limits to the full
        embedding extent. Limits are computed once so every frame uses the
        same coordinate space regardless of which points are visible.

        Parameters
        ----------
        umapped : DataFrame
            DataFrame with columns [0, 1] (UMAP x/y). Used only for limits.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Opaque figure background; transparent axes patch for compositing
        self.fig.patch.set_alpha(1)
        self.ax.patch.set_alpha(0)

        # Strip all decorations for a clean dark-canvas look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # ── Compute fixed viewport ────────────────────────────────────────────
        # 1. Expand by axis_padding to add breathing room around the point cloud
        x_min = umapped[0].min() - self.axis_padding
        x_max = umapped[0].max() + self.axis_padding
        y_min = umapped[1].min() - self.axis_padding
        y_max = umapped[1].max() + self.axis_padding

        # 2. Find the centre, then shrink half-widths by axis_zoom to crop inward
        self._x_c = (x_min + x_max) / 2
        self._y_c = (y_min + y_max) / 2
        self._x_h = (x_max - x_min) / 2 / self.axis_zoom
        self._y_h = (y_max - y_min) / 2 / self.axis_zoom

        self.ax.set_xlim(self._x_c - self._x_h, self._x_c + self._x_h)
        self.ax.set_ylim(self._y_c - self._y_h, self._y_c + self._y_h)
        # Prevent scatter calls inside _draw_frame from auto-expanding limits
        self.ax.set_autoscale_on(False)

    # ──────────────────────────────────────────────────────────────────────────
    # Data pre-caching
    # ──────────────────────────────────────────────────────────────────────────

    def _precache_days(self, df, combined_embeddings, hdbscan_colours):
        """
        Pre-build per-day data arrays to avoid repeated DataFrame lookups
        inside the hot render loop.

        For each calendar day this builds:
          - An ordered list of unique video IDs (first-watch order, embedding-only)
          - The corresponding UMAP coords, HDBSCAN colours, and noise mask as
            numpy arrays ready for ax.scatter()

        Parameters
        ----------
        df : DataFrame
            Watch activity data with 'date' (datetime) and 'id' columns.
        combined_embeddings : DataFrame
            Clustered embeddings; columns [0, 1, 'hdbscan_label'].
        hdbscan_colours : dict
            Maps cluster label (int) → RGBA tuple.
        """
        self._days = sorted(df['date'].dt.to_period('D').unique())
        self._day_ordered_ids = []
        self._day_umap_data   = []

        # Pre-compute sigmoid duration sizes for every video in one pass
        _dur_sizes = self._compute_duration_sizes(combined_embeddings) if self.scale_by_duration else None

        for day in tqdm(self._days, desc='Pre-caching days'):
            # All watches for this day, sorted oldest-first
            day_df = df[df['date'].dt.to_period('D') == day].sort_values('date')

            # Deduplicate while preserving first-watch order; skip missing embeddings
            seen, ordered = set(), []
            for vid_id in day_df['id']:
                if vid_id not in seen and vid_id in combined_embeddings.index:
                    seen.add(vid_id)
                    ordered.append(vid_id)
            self._day_ordered_ids.append(ordered)

            if ordered:
                umap_sub = combined_embeddings.loc[ordered]
                labels   = umap_sub['hdbscan_label'].values
                # Map each label to its colour; fall back to noise_fallback_color
                colors   = np.array([
                    hdbscan_colours.get(l, self.noise_fallback_color) for l in labels
                ])
                is_noise = (labels == -1)
                sizes    = _dur_sizes.reindex(umap_sub.index).fillna(self.noise_point_size).values if _dur_sizes is not None else None
            else:
                # Empty arrays so _draw_frame can safely slice without branching
                umap_sub = combined_embeddings.iloc[:0]
                colors   = np.empty((0, 4))
                is_noise = np.empty(0, dtype=bool)
                sizes    = np.empty(0) if _dur_sizes is not None else None

            self._day_umap_data.append((umap_sub, colors, is_noise, sizes))
            # print(f'  {day}: {len(ordered)} embeddable watches')

    # ──────────────────────────────────────────────────────────────────────────
    # Alpha decay functions
    # ──────────────────────────────────────────────────────────────────────────

    def _get_alpha(self, offset: int) -> float:
        """
        Alpha for regular (clustered) points by how many days in the past they are.

        offset=0 → current day → fully opaque.
        offset in [1, window_size) → power-law decay: fast initial drop then a
            flatter tail, controlled by cluster_alpha_decay.
        offset >= window_size → invisible (0.0).
        """
        if offset == 0:
            return 1.0
        if 1 <= offset < self.window_size:
            return (1.0 - offset / self.window_size) ** self.cluster_alpha_decay
        return 0.0

    def _get_noise_alpha(self, offset: int) -> float:
        """
        Alpha for noise (unclustered) points by how many days in the past they are.

        Noise points use a much wider window and slower decay so they linger as
        a dim background, showing the full embedding topology at all times.
        """
        if offset == 0:
            return self.noise_base_alpha
        if 1 <= offset < self.noise_window:
            return self.noise_base_alpha * (1.0 - offset / self.noise_window) ** self.noise_alpha_decay
        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Duration size scaling
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_duration_sizes(self, combined_embeddings):
        """
        Map each video's duration to a point size via a sigmoid on log-duration.

        Log scale is used because durations are heavy-tailed (most videos are
        short; a few are very long). The sigmoid is centred at the log-median
        with a gentle slope (0.8 / log_std) so the size difference between
        the extremes (very short / very long) is compressed — both ends stay
        near noise_duration_min_size / noise_duration_max_size rather than
        spreading linearly across the full range.
        """
        dur = pd.to_numeric(combined_embeddings['duration'], errors='coerce')
        median = dur.median()
        if np.isnan(median) or median <= 0:
            median = 1.0
        dur_filled = dur.fillna(median).clip(lower=1)

        log_dur = np.log(dur_filled.values)
        log_med = np.log(median)
        log_std = log_dur.std()
        if log_std == 0:
            log_std = 1.0

        # Gentle sigmoid: 0.8 stddev per unit keeps extremes compressed
        k = 0.8 / log_std
        sig = 1.0 / (1.0 + np.exp(-k * (log_dur - log_med)))

        sizes = self.noise_duration_min_size + (self.noise_duration_max_size - self.noise_duration_min_size) * sig
        return pd.Series(sizes, index=combined_embeddings.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Frame drawing
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_frame(self, day_idx: int, n_visible: int):
        """
        Clear and redraw all scatter layers for one animation frame.

        Two-pass rendering:
          Pass 1 — Noise: dim grey points over a wide trailing window; no glow.
          Pass 2 — Clusters: full-colour points with layered glow bloom rings,
                   over a shorter trailing window.

        Parameters
        ----------
        day_idx   : int
            Index of the 'current' day in self._days.
        n_visible : int
            How many of today's chronologically-ordered watches to show.
            Incremented across frames_per_day frames to animate the reveal.
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

            umap_sub, colors, is_noise, sizes = self._day_umap_data[d_idx]
            # Today: show only n_visible; past days: show everything
            n = n_visible if d_idx == day_idx else len(umap_sub)
            if n == 0 or not is_noise[:n].any():
                continue

            noise_mask = is_noise[:n]
            noise_s = sizes[:n][noise_mask] if (self.scale_by_duration and sizes is not None) else self.noise_point_size
            self.ax.scatter(
                umap_sub[0].values[:n][noise_mask],
                umap_sub[1].values[:n][noise_mask],
                s=noise_s, alpha=alpha,
                color=self.noise_color, linewidths=0, zorder=1,
            )

        # ── Pass 2: clustered points with glow bloom ───────────────────────────
        for d_idx in range(max(0, day_idx - self.window_size + 1), day_idx + 1):
            offset = day_idx - d_idx
            alpha  = self._get_alpha(offset)
            if alpha <= 0:
                continue

            umap_sub, colors, is_noise, sizes = self._day_umap_data[d_idx]
            n = n_visible if d_idx == day_idx else len(umap_sub)
            non_noise = ~is_noise[:n]
            if n == 0 or not non_noise.any():
                continue

            xu = umap_sub[0].values[:n][non_noise]
            yu = umap_sub[1].values[:n][non_noise]
            pc = colors[:n][non_noise]

            # Per-point core size; scale factor used to proportionally resize glow rings
            cluster_s = sizes[:n][non_noise] if (self.scale_by_duration and sizes is not None) else self._core_point_size
            scale = cluster_s / self._core_point_size  # scalar 1.0 or per-point array

            # Render glow rings from outermost (large, faint) to innermost (small, bright)
            for size, ring_alpha in self._glow_layers:
                self.ax.scatter(
                    xu, yu, s=size * scale, alpha=ring_alpha * alpha,
                    color=pc, linewidths=1, zorder=3,
                )

            # Solid core dot on top of the glow stack
            self.ax.scatter(
                xu, yu, s=cluster_s, alpha=alpha,
                color=pc, linewidths=0, zorder=4,
            )

        # Re-enforce viewport — savefig can silently expand limits despite autoscale_on=False
        self.ax.set_xlim(self._x_c - self._x_h, self._x_c + self._x_h)
        self.ax.set_ylim(self._y_c - self._y_h, self._y_c + self._y_h)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, df, combined_embeddings, hdbscan_colours, output_dir, umapped=None):
        """
        Run the full render loop and write one PNG per frame to output_dir.

        Parameters
        ----------
        df : DataFrame
            Watch activity data with 'date' (datetime) and 'id' columns.
        combined_embeddings : DataFrame
            Clustered embeddings with columns [0, 1, 'hdbscan_label'].
        hdbscan_colours : dict
            Maps cluster label (int) → RGBA colour tuple.
        output_dir : str
            Destination directory. Cleared and recreated at the start of each run.
        umapped : DataFrame, optional
            Used only to compute axis limits. Defaults to combined_embeddings.
        """
        if umapped is None:
            umapped = combined_embeddings

        print(f'frames_per_day={self.frames_per_day}  window_size={self.window_size}  noise_window={self.noise_window}')
        print(f'Activities with embeddings: {len(df[df["id"].isin(combined_embeddings.index)])}')

        # Clear output directory so stale frames don't contaminate the new render
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        self._setup_figure(umapped)
        self._precache_days(df, combined_embeddings, hdbscan_colours)

        frame_idx = 0
        for day_idx, day in tqdm(enumerate(self._days), total=len(self._days), desc='Rendering'):
            n_today = len(self._day_ordered_ids[day_idx])
            for f in range(self.frames_per_day):
                # Reveal watches proportionally: frame 1 shows 1/frames_per_day,
                # the last frame shows all of today's watches
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
        combined_embeddings,
        hdbscan_colours,
        day_idx: int = 0,
        n_visible: int = None,
        umapped=None,
        show: bool = True,
        save_path: str = None,
        preview_dpi: int = 300,
    ):
        """
        Render and display a single frame for quick visual iteration.

        Sets up the figure and pre-caches days on first call; subsequent calls
        reuse the cache so parameter tweaks (glow, alpha, etc.) are fast.
        Call reset() first if you want to re-initialise with different data.

        Parameters
        ----------
        df : DataFrame
            Watch activity data.
        combined_embeddings : DataFrame
            Clustered embeddings with columns [0, 1, 'hdbscan_label'].
        hdbscan_colours : dict
            Maps cluster label (int) → RGBA colour tuple.
        day_idx : int
            Which day to render (0-indexed into the sorted unique days).
        n_visible : int, optional
            How many of that day's watches to show. Defaults to all of them.
        umapped : DataFrame, optional
            For axis limits. Defaults to combined_embeddings.
        show : bool
            Whether to call plt.show() after drawing.
        save_path : str, optional
            If given, saves the frame PNG to this path at the renderer's dpi.
        preview_dpi : int
            DPI used for plt.show() preview (lower = faster screen render).
            Has no effect on save_path output, which always uses self.dpi.
        """
        if umapped is None:
            umapped = combined_embeddings

        # Lazily initialise — skip if already cached from a previous call
        if self._days is None:
            # Use a lower dpi for the preview figure so screen rendering is fast
            original_dpi = self.dpi
            self.dpi = preview_dpi
            self._setup_figure(umapped)
            self.dpi = original_dpi
            self._precache_days(df, combined_embeddings, hdbscan_colours)

        # Clamp to valid range
        day_idx = min(max(day_idx, 0), len(self._days) - 1)

        # Default: reveal all watches for that day
        if n_visible is None:
            n_visible = len(self._day_ordered_ids[day_idx])

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
        frame : int
            Absolute frame index (0-based), as used in the output filenames.

        Returns
        -------
        datetime.date
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
        self._x_c = self._x_h = self._y_c = self._y_h = None