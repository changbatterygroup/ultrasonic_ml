from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class AcousticsViewer:
    """Base viewer for raw and analyzed waveforms on a single axes."""

    def __init__(self, db, waveforms=None, analyses=None, *, x_unit="ns", y_unit="V", figsize=(10, 5)):
        self.db = db
        self.waveforms = waveforms or db.waveform_columns
        self.analyses = analyses or []
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.figsize = figsize

        self.n = db.get_acquisition_count()
        self.row = 0

        self.time = np.asarray(db.fetch_time(0))
        self.x_limits = (float(self.time.min()), float(self.time.max()))

        self.absolute_max = {
            waveform: self._get_maximum(waveform)
            for waveform in self.waveforms
        }

        self.fig = None
        self.ax = None
        self.slider = None
        self.raw_lines = {}
        self.analysis_lines = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_maximum(self, waveform):
        if hasattr(self.db, "absolute_max") and waveform in self.db.absolute_max:
            return float(self.db.absolute_max[waveform])

        maximum = 0.0

        for row in range(self.n):
            data = np.asarray(self.db.fetch_waveform(waveform, row))
            maximum = max(maximum, float(np.max(np.abs(data))))

        return maximum

    def fetch_waveform(self, waveform, row=None):
        row = self.row if row is None else row
        return np.asarray(self.db.fetch_waveform(waveform, row))

    def fetch_analysis(self, analysis, row=None):
        row = self.row if row is None else row
        return np.asarray([self.db.fetch_analysis_value(row, waveform, analysis, result_name: str, reference_id: int | None = None))

    def fetch_time(self, row=None):
        row = self.row if row is None else row
        return np.asarray(self.db.fetch_time(row))

    def get_collection_index(self, row=None):
        row = self.row if row is None else row
        return self.db.get_acquisition_index(row)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def build(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        self._setup_axes()
        self._create_slider()
        self._connect_events()
        self.update()

        # self.fig.tight_layout(rect=(0, 0.08, 1, 0.96))

        return self.fig, self.ax

    def _setup_axes(self):
        self.ax.set_xlim(*self.x_limits)

        maximum = max(self.absolute_max.values())
        self.ax.set_ylim(-maximum, maximum)

        self.ax.set_xlabel(f"Time ({self.x_unit})")
        self.ax.set_ylabel(f"Amplitude ({self.y_unit})")
        self.ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_raw(self):
        """Plot raw waveforms."""
        for waveform in self.waveforms:
            data = self.fetch_waveform(waveform)
            n = min(len(self.time), len(data))

            if waveform not in self.raw_lines:
                self.raw_lines[waveform], = self.ax.plot(
                    self.time[:n],
                    data[:n],
                    lw=1,
                    label=waveform,
                )
            else:
                self.raw_lines[waveform].set_data(
                    self.time[:n],
                    data[:n],
                )

    def plot_analyzed(self):
        """Plot analyzed waveforms."""
        for analysis in self.analyses:
            data = self.fetch_analysis(analysis)
            n = min(len(self.time), len(data))

            if analysis not in self.analysis_lines:
                self.analysis_lines[analysis], = self.ax.plot(
                    self.time[:n],
                    data[:n],
                    lw=1,
                    label=analysis,
                )
            else:
                self.analysis_lines[analysis].set_data(
                    self.time[:n],
                    data[:n],
                )

    def update(self):
        self.plot_raw()
        self.plot_analyzed()

        self.ax.legend()

        self.fig.suptitle(
            f"Collection index: {self.get_collection_index()}",
            fontsize=12,
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _create_slider(self):
        slider_ax = self.fig.add_axes((0.15, 0.02, 0.7, 0.03))

        self.slider = Slider(
            slider_ax,
            "Collection",
            0,
            self.n - 1,
            valinit=0,
            valstep=1,
        )

        self.slider.on_changed(self._on_slider)

    def _on_slider(self, value):
        self.row = int(value)
        self.update()

    def next(self):
        self.goto(self.row + 1)

    def previous(self):
        self.goto(self.row - 1)

    def goto(self, row):
        row = int(np.clip(row, 0, self.n - 1))
        self.slider.set_val(row)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes is self.ax:
            self.on_click(event)

    def on_click(self, event):
        pass

    def _on_key(self, event):
        if event.key in ("right", "down"):
            self.next()
        elif event.key in ("left", "up"):
            self.previous()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def refresh(self):
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def show(self):
        if self.fig is None:
            self.build()

        plt.show()
        
        
        
        


# from __future__ import annotations

# import io
# import sqlite3
# from dataclasses import dataclass
# from typing import Any

# import numpy as np
# from matplotlib import pyplot as plt


# @dataclass
# class _WaveformStore:
#     collection_index: np.ndarray
#     x_mm: np.ndarray
#     z_mm: np.ndarray
#     time_ns: list[np.ndarray]
#     voltage: list[np.ndarray]


# class AcousticsSpectralSpatialViewer:
#     """Clickable spatial/spectral viewer for an open Loaded_Database object.

#     Expected input:
#         database = db.Loaded_Database(sqlite_file)
#     where database.connection is an open sqlite3.Connection.
#     """

#     def __init__(
#         self,
#         database: Any,
#         *,
#         table: str = "acoustics",
#         index_column: str = "collection_index",
#         x_column: str = "X",
#         z_column: str = "Z",
#         time_column: str = "time",
#     ) -> None:
#         self.connection = self._resolve_connection(database)
#         self.table = table
#         self.index_column = index_column
#         self.x_column = x_column
#         self.z_column = z_column
#         self.time_column = time_column
#         self.voltage_columns = database.voltage_keys

#         self.data = self._load_acoustics_rows()
#         self._prepare_fft_cache()

#         self._fig: plt.Figure | None = None
#         self._ax_images: list[plt.Axes | None] = None
#         self._ax_spectrums: list[plt.Axes | None] = None
#         self._selected_marker = None
#         self._freq_line = None

#         self.cmap='viridis'
        
#         self.abs_max_voltage = max([np.max(np.abs(v)) for v in self.data.voltage])
        
#     @staticmethod
#     def _resolve_connection(database: Any) -> sqlite3.Connection:
#         if isinstance(database, sqlite3.Connection):
#             return database
#         connection = getattr(database, "connection", None)
#         if isinstance(connection, sqlite3.Connection):
#             return connection
#         raise TypeError("database must be a Loaded_Database-like object with .connection")

#     def _table_columns(self) -> list[str]:
#         rows = self.connection.execute(f"PRAGMA table_info({self.table})").fetchall()
#         return [row[1] for row in rows]

#     @staticmethod
#     def _to_array(value: Any) -> np.ndarray:
#         if isinstance(value, np.ndarray):
#             return value
#         if isinstance(value, (bytes, bytearray, memoryview)):
#             return np.load(io.BytesIO(value), allow_pickle=True)
#         return np.asarray(value)

#     def _load_acoustics_rows(self) -> _WaveformStore:
#         query = f"""
#             SELECT {self.index_column}, {self.x_column}, {self.z_column},
#                    {self.time_column}, {self.voltage_column}
#             FROM {self.table}
#             ORDER BY {self.index_column}
#         """
#         rows = self.connection.execute(query).fetchall()
#         if not rows:
#             raise ValueError("No rows found in acoustics table")

#         collection_index = np.array([int(r[0]) for r in rows], dtype=int)
#         x_mm = np.array([float(r[1]) for r in rows], dtype=float)
#         z_mm = np.array([float(r[2]) for r in rows], dtype=float)
#         time_ns = [self._to_array(r[3]).astype(float).ravel() for r in rows]
#         voltage = [self._to_array(r[4]).astype(float).ravel() for r in rows]

#         return _WaveformStore(
#             collection_index=collection_index,
#             x_mm=x_mm,
#             z_mm=z_mm,
#             time_ns=time_ns,
#             voltage=voltage,
#         )

#     def _prepare_fft_cache(self) -> None:
#         wave_lengths = np.array([len(v) for v in self.data.voltage], dtype=int)
#         time_lengths = np.array([len(t) for t in self.data.time_ns], dtype=int)
#         n = int(min(wave_lengths.min(), time_lengths.min()))
#         if n < 8:
#             raise ValueError("Waveforms are too short for spectral analysis")

#         # Trim to shortest length so heterogeneous rows are still usable.
#         self._wave_matrix = np.vstack([v[:n] for v in self.data.voltage])
#         self._time_matrix_ns = np.vstack([t[:n] for t in self.data.time_ns])

#         dt_ns = float(np.median(np.diff(self._time_matrix_ns[0])))
#         if dt_ns <= 0:
#             raise ValueError("Invalid time axis (dt <= 0)")

#         dt_s = dt_ns * 1e-9
#         self.freq_hz = np.fft.rfftfreq(n, d=dt_s)
#         self.spectrum_mag = np.abs(np.fft.rfft(self._wave_matrix, axis=1))

#     def _nearest_row_for_position(self, x_mm: float, z_mm: float) -> int:
#         d2 = (self.data.x_mm - x_mm) ** 2 + (self.data.z_mm - z_mm) ** 2
#         return int(np.argmin(d2))

#     def _nearest_freq_index(self, freq_hz: float) -> int:
#         return int(np.argmin(np.abs(self.freq_hz - float(freq_hz))))

#     def _grid_view(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
#         x_unique = np.unique(self.data.x_mm)
#         z_unique = np.unique(self.data.z_mm)
#         if x_unique.size * z_unique.size != values.size:
#             return None

#         x_to_i = {v: i for i, v in enumerate(x_unique)}
#         z_to_i = {v: i for i, v in enumerate(z_unique)}
#         grid = np.full((z_unique.size, x_unique.size), np.nan)

#         for x_val, z_val, v in zip(self.data.x_mm, self.data.z_mm, values):
#             zi = z_to_i.get(z_val)
#             xi = x_to_i.get(x_val)
#             if zi is None or xi is None:
#                 return None
#             if not np.isnan(grid[zi, xi]):
#                 return None
#             grid[zi, xi] = v

#         if np.isnan(grid).any():
#             return None
#         return x_unique, z_unique, grid

#     def plot_image_at_frequency(
#         self,
#         freq_hz: float,
#         *,
#         ax: plt.Axes | None = None,
#         cmap: str = "viridis",
#     ) -> plt.Axes:
#         """Plot spatial image using spectral magnitude at selected frequency."""
#         if ax is None: _, ax = plt.subplots()

#         fi = self._nearest_freq_index(freq_hz)
#         f_sel = float(self.freq_hz[fi])
#         vals = self.spectrum_mag[:, fi]

#         grid_data = self._grid_view(vals)
#         if grid_data is not None:
#             x_unique, z_unique, grid = grid_data
#             im = ax.pcolormesh(x_unique, z_unique, grid, shading="auto", cmap=cmap)
#         else:
#             im = ax.scatter(self.data.x_mm, self.data.z_mm, c=vals, cmap=cmap, s=28, linewidths=0)

#         ax.set_xlabel("x (mm)")
#         ax.set_ylabel("z (mm)")
#         ax.set_title(f"Spatial magnitude at {f_sel/1e6:.3f} MHz")
#         ax.set_aspect("equal", adjustable="datalim")

#         if ax.figure is not None:
#             if len(ax.figure.axes) == 0 or ax.figure.axes[-1] is not ax:
#                 pass
#             plt.colorbar(im, ax=ax, label="|FFT|", fraction=0.046, pad=0.04)

#         return ax

#     def plot_spectrum_at_position(
#         self,
#         x_mm: float,
#         z_mm: float,
#         *,
#         ax: plt.Axes | None = None,
#         max_hz: float | None = None,
#     ) -> plt.Axes:
#         """Plot a spectrum at the nearest measured (x, z) spatial position."""
#         if ax is None:
#             _, ax = plt.subplots()

#         row = self._nearest_row_for_position(x_mm, z_mm)
#         x_sel = float(self.data.x_mm[row])
#         z_sel = float(self.data.z_mm[row])

#         freq = self.freq_hz
#         mag = self.spectrum_mag[row]

#         if max_hz is not None:
#             mask = freq <= float(max_hz)
#             freq = freq[mask]
#             mag = mag[mask]

#         ax.plot(freq, y, color="tab:blue", lw=1.2)
#         ax.set_xlabel("Frequency (Hz)")
#         ax.set_ylabel(ylabel)
#         ax.set_title(
#             f"Spectrum near x={x_sel:.2f} mm, z={z_sel:.2f} mm "
#             f"(collection_index={self.data.collection_index[row]})"
#         )

#         return ax

#     def plot_image_at_clicked_spectral_position(self, event, *, ax: plt.Axes | None = None) -> plt.Axes:
#         """Callback-friendly method: event.xdata is treated as frequency in Hz."""
#         if event is None or event.xdata is None:
#             return ax if ax is not None else self._ax_image
#         target_ax = ax if ax is not None else self._ax_image
#         if target_ax is None:
#             raise ValueError("No image axis available")
#         target_ax.clear()
#         return self.plot_image_at_frequency(float(event.xdata), ax=target_ax)

#     def plot_spectrum_at_clicked_spatial_position(self, event, *, ax: plt.Axes | None = None) -> plt.Axes:
#         """Callback-friendly method: event.(xdata, ydata) is treated as spatial (x, z)."""
#         if event is None or event.xdata is None or event.ydata is None:
#             return ax if ax is not None else self._ax_spectrum
#         target_ax = ax if ax is not None else self._ax_spectrum
#         if target_ax is None:
#             raise ValueError("No spectrum axis available")
#         target_ax.clear()
#         return self.plot_spectrum_at_position(float(event.xdata), float(event.ydata), ax=target_ax)

#     def build_clickable_raw_figure(
#         self,
#         *,
#         initial_x: float | None = None,
#         initial_z: float | None = None,
#         initial_freq_hz: float | None = None,
#         db_scale: bool = True,
#         max_hz: float | None = None,
#     ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
#         """Create linked spatial/spectral axes with click interactions enabled."""
#         fig, axs = plt.subplots(len(self.voltage_columns)+1, 1, figsize=(10,10), constrained_layout=True)
#         axs = axs.flatten()
#         ax_images = axs[:-1]
#         ax_spectrum = axs[-1]
#         self._fig = fig
#         self._ax_images = ax_images
#         self._ax_spectrum = ax_spectrum

#         if initial_x is None or initial_z is None:
#             row0 = 0
#             initial_x = float(self.data.x_mm[row0])
#             initial_z = float(self.data.z_mm[row0])

#         if initial_freq_hz is None:
#             row0 = self._nearest_row_for_position(float(initial_x), float(initial_z))
#             local_spec = self.spectrum_mag[row0]
#             initial_freq_hz = float(self.freq_hz[np.argmax(local_spec)])

#         self.plot_image_at_frequency(float(initial_freq_hz), axs=ax_images, cmap=self.cmap)
        
#         self.plot_spectrum_at_position(float(initial_x), float(initial_z), ax=ax_spectrum, 
#                                        db_scale=db_scale, max_hz=max_hz, 
#                                        )

#         row_sel = self._nearest_row_for_position(float(initial_x), float(initial_z))
#         self._selected_marker = ax_image.scatter(
#             [self.data.x_mm[row_sel]],
#             [self.data.z_mm[row_sel]],
#             s=90,
#             facecolors="none",
#             edgecolors="red",
#             linewidths=1.5,
#         )

#         freq_idx = self._nearest_freq_index(float(initial_freq_hz))
#         freq_sel = float(self.freq_hz[freq_idx])
#         self._freq_line = ax_spectrum.axvline(freq_sel, color="red", lw=1.0, ls="--")

#         def _on_click(event) -> None:
#             if event.inaxes is ax_image and event.xdata is not None and event.ydata is not None:
#                 x_click = float(event.xdata)
#                 z_click = float(event.ydata)
#                 row = self._nearest_row_for_position(x_click, z_click)

#                 ax_spectrum.clear()
#                 self.plot_spectrum_at_position(
#                     float(self.data.x_mm[row]),
#                     float(self.data.z_mm[row]),
#                     ax=ax_spectrum,
#                     db_scale=db_scale,
#                     max_hz=max_hz,
#                 )

#                 if self._selected_marker is not None:
#                     self._selected_marker.remove()
#                 self._selected_marker = ax_image.scatter(
#                     [self.data.x_mm[row]],
#                     [self.data.z_mm[row]],
#                     s=90,
#                     facecolors="none",
#                     edgecolors="red",
#                     linewidths=1.5,
#                 )

#                 if self._freq_line is not None:
#                     f_now = self._freq_line.get_xdata()[0]
#                     self._freq_line = ax_spectrum.axvline(f_now, color="red", lw=1.0, ls="--")

#                 fig.canvas.draw_idle()

#             elif event.inaxes is ax_spectrum and event.xdata is not None:
#                 f_click = float(event.xdata)
#                 ax_image.clear()
#                 self.plot_image_at_frequency(f_click, ax=ax_image, cmap=cmap)

#                 if self._selected_marker is not None:
#                     row = self._nearest_row_for_position(
#                         float(self._selected_marker.get_offsets()[0][0]),
#                         float(self._selected_marker.get_offsets()[0][1]),
#                     )
#                     self._selected_marker = ax_image.scatter(
#                         [self.data.x_mm[row]],
#                         [self.data.z_mm[row]],
#                         s=90,
#                         facecolors="none",
#                         edgecolors="red",
#                         linewidths=1.5,
#                     )

#                 if self._freq_line is not None:
#                     self._freq_line.remove()
#                 self._freq_line = ax_spectrum.axvline(f_click, color="red", lw=1.0, ls="--")

#                 fig.canvas.draw_idle()

#         fig.canvas.mpl_connect("button_press_event", _on_click)
#         return fig, (ax_image, ax_spectrum)
