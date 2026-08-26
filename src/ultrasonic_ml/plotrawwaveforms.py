from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


DEFAULT_TIME_SCALE = 1e-3  # stored ns -> us


@dataclass
class AcousticsData:
    collection_index: np.ndarray
    x_mm: np.ndarray
    z_mm: np.ndarray
    time_raw: list[np.ndarray]
    voltage: list[np.ndarray]


class AcousticsPlotter:
    """Efficient plotting utilities for spatial and spectral acoustics views.

    Input must be an open database object created as:
        database = db.Loaded_Database(sqlite_file)
    The object must expose `database.connection` as a sqlite3.Connection.
    """

    def __init__(
        self,
        database: Any,
        *,
        voltage_column: str | None = None,
        time_column: str = "time",
        x_column: str = "X",
        z_column: str = "Z",
        index_column: str = "collection_index",
        table: str = "acoustics",
    ) -> None:
        self.connection = self._resolve_connection(database)
        self.table = table
        self.time_column = time_column
        self.x_column = x_column
        self.z_column = z_column
        self.index_column = index_column
        self.voltage_column = voltage_column or self._auto_voltage_column()

        self.data = self._load_table()
        self._matrix_voltage: np.ndarray | None = None
        self._matrix_time: np.ndarray | None = None
        self._freq_hz: np.ndarray | None = None
        self._mag_spectrum: np.ndarray | None = None

    @staticmethod
    def _resolve_connection(database: Any) -> sqlite3.Connection:
        if isinstance(database, sqlite3.Connection):
            return database
        connection = getattr(database, "connection", None)
        if isinstance(connection, sqlite3.Connection):
            return connection
        raise TypeError(
            "Expected an open database object with `.connection` as sqlite3.Connection."
        )

    def _table_columns(self) -> list[str]:
        rows = self.connection.execute(f"PRAGMA table_info({self.table})").fetchall()
        return [row[1] for row in rows]

    def _auto_voltage_column(self) -> str:
        columns = self._table_columns()
        for candidate in ("voltage_preprocessed", "voltage"):
            if candidate in columns:
                return candidate

        voltage_like = [c for c in columns if c.startswith("voltage")]
        if voltage_like:
            return voltage_like[0]
        raise ValueError("No voltage-like column found in acoustics table.")

    @staticmethod
    def _blob_to_array(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.load(io.BytesIO(value), allow_pickle=True)
        return np.asarray(value)

    def _load_table(self) -> AcousticsData:
        query = f"""
            SELECT {self.index_column}, {self.x_column}, {self.z_column},
                   {self.time_column}, {self.voltage_column}
            FROM {self.table}
            ORDER BY {self.index_column}
        """
        rows = self.connection.execute(query).fetchall()
        if not rows:
            raise ValueError(f"{self.table} table is empty")

        collection_index = np.array([int(r[0]) for r in rows], dtype=int)
        x_mm = np.array([float(r[1]) for r in rows], dtype=float)
        z_mm = np.array([float(r[2]) for r in rows], dtype=float)
        time_raw = [self._blob_to_array(r[3]).astype(float) for r in rows]
        voltage = [self._blob_to_array(r[4]).astype(float) for r in rows]

        return AcousticsData(
            collection_index=collection_index,
            x_mm=x_mm,
            z_mm=z_mm,
            time_raw=time_raw,
            voltage=voltage,
        )

    def _build_matrices(self) -> None:
        if self._matrix_voltage is not None:
            return

        lengths = np.array([len(v) for v in self.data.voltage], dtype=int)
        if np.any(lengths != lengths[0]):
            raise ValueError(
                "Waveforms are not equal length; spectral matrix operations require equal-length signals."
            )

        time_lengths = np.array([len(t) for t in self.data.time_raw], dtype=int)
        if np.any(time_lengths != time_lengths[0]):
            raise ValueError("Time arrays are not equal length.")

        self._matrix_voltage = np.vstack(self.data.voltage)
        self._matrix_time = np.vstack(self.data.time_raw)

    def _build_spectra(self) -> None:
        if self._mag_spectrum is not None:
            return

        self._build_matrices()
        assert self._matrix_time is not None
        assert self._matrix_voltage is not None

        dt_raw = float(np.median(np.diff(self._matrix_time[0])))
        if dt_raw <= 0:
            raise ValueError("Invalid time axis: non-positive dt")

        dt_s = dt_raw * 1e-9  # stored ns -> s
        self._freq_hz = np.fft.rfftfreq(self._matrix_voltage.shape[1], d=dt_s)
        self._mag_spectrum = np.abs(np.fft.rfft(self._matrix_voltage, axis=1))

    def _row_from_collection_index(self, collection_index: int) -> int:
        matches = np.where(self.data.collection_index == int(collection_index))[0]
        if len(matches) == 0:
            raise IndexError(f"collection_index {collection_index} not found")
        return int(matches[0])

    @property
    def freq_hz(self) -> np.ndarray:
        self._build_spectra()
        assert self._freq_hz is not None
        return self._freq_hz

    @property
    def mag_spectrum(self) -> np.ndarray:
        self._build_spectra()
        assert self._mag_spectrum is not None
        return self._mag_spectrum

    def metric_rms(self) -> np.ndarray:
        self._build_matrices()
        assert self._matrix_voltage is not None
        return np.sqrt(np.mean(self._matrix_voltage**2, axis=1))

    def metric_peak_to_peak(self) -> np.ndarray:
        self._build_matrices()
        assert self._matrix_voltage is not None
        return np.ptp(self._matrix_voltage, axis=1)

    def metric_spectral_peak_hz(self, min_hz: float = 0.0, max_hz: float | None = None) -> np.ndarray:
        f = self.freq_hz
        s = self.mag_spectrum

        fmax = float(max_hz) if max_hz is not None else float(f[-1])
        mask = (f >= float(min_hz)) & (f <= fmax)
        if not np.any(mask):
            raise ValueError("No frequencies selected by min_hz/max_hz")

        local = s[:, mask]
        idx = np.argmax(local, axis=1)
        return f[mask][idx]

    def metric_band_energy_ratio(self, band_hz: tuple[float, float]) -> np.ndarray:
        low, high = float(band_hz[0]), float(band_hz[1])
        if low >= high:
            raise ValueError("band_hz must be (low, high) with low < high")

        f = self.freq_hz
        s2 = self.mag_spectrum**2
        band_mask = (f >= low) & (f <= high)
        if not np.any(band_mask):
            raise ValueError("Band does not overlap frequency axis")

        band_energy = np.trapezoid(s2[:, band_mask], x=f[band_mask], axis=1)
        total_energy = np.trapezoid(s2, x=f, axis=1)
        eps = np.finfo(float).eps
        return band_energy / np.maximum(total_energy, eps)

    def _spatial_scatter(
        self,
        values: np.ndarray,
        *,
        ax: plt.Axes,
        title: str,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        cbar_label: str | None = None,
    ) -> None:
        norm = Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(
            self.data.x_mm,
            self.data.z_mm,
            c=values,
            cmap=cmap,
            norm=norm,
            s=24,
            linewidths=0,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        cbar = plt.colorbar(sc, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    def plot_spatial_metric(
        self,
        metric: str = "rms",
        *,
        band_hz: tuple[float, float] = (2.0e6, 2.5e6),
        min_hz: float = 0.0,
        max_hz: float | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        if metric == "rms":
            vals = self.metric_rms()
            self._spatial_scatter(vals, ax=ax, title="Spatial RMS", cbar_label="RMS")
        elif metric == "ptp":
            vals = self.metric_peak_to_peak()
            self._spatial_scatter(vals, ax=ax, title="Spatial Peak-to-Peak", cbar_label="Peak-to-peak")
        elif metric == "peak_freq":
            vals = self.metric_spectral_peak_hz(min_hz=min_hz, max_hz=max_hz)
            self._spatial_scatter(
                vals,
                ax=ax,
                title="Spatial Spectral Peak Frequency",
                cbar_label="Frequency (Hz)",
            )
        elif metric == "band_ratio":
            vals = self.metric_band_energy_ratio(band_hz)
            self._spatial_scatter(
                vals,
                ax=ax,
                title=f"Spatial Band Energy Ratio ({band_hz[0]:.2e}-{band_hz[1]:.2e} Hz)",
                cbar_label="Energy ratio",
            )
        else:
            raise ValueError("metric must be one of: rms, ptp, peak_freq, band_ratio")

        return ax

    def plot_waveform(
        self,
        *,
        collection_index: int | None = None,
        row: int = 0,
        time_scale: float = DEFAULT_TIME_SCALE,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        i = self._row_from_collection_index(collection_index) if collection_index is not None else int(row)
        t = self.data.time_raw[i] * time_scale
        v = self.data.voltage[i]

        ax.plot(t, v, lw=1.25)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage")
        ax.set_title(
            f"Waveform | collection_index={self.data.collection_index[i]} | "
            f"x={self.data.x_mm[i]:.2f} mm | z={self.data.z_mm[i]:.2f} mm"
        )
        return ax

    def plot_spectrum(
        self,
        *,
        collection_index: int | None = None,
        row: int = 0,
        max_hz: float | None = None,
        db_scale: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        i = self._row_from_collection_index(collection_index) if collection_index is not None else int(row)
        f = self.freq_hz
        s = self.mag_spectrum[i]

        if max_hz is not None:
            mask = f <= float(max_hz)
            f_plot = f[mask]
            s_plot = s[mask]
        else:
            f_plot = f
            s_plot = s

        if db_scale:
            s_plot = 20.0 * np.log10(np.maximum(s_plot, np.finfo(float).eps))
            ax.set_ylabel("Magnitude (dB)")
        else:
            ax.set_ylabel("Magnitude")

        ax.plot(f_plot, s_plot, lw=1.25)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"Spectrum | collection_index={self.data.collection_index[i]}")
        return ax

    def plot_spatial_spectral_dashboard(
        self,
        *,
        collection_index: int | None = None,
        row: int = 0,
        band_hz: tuple[float, float] = (2.0e6, 2.5e6),
        max_hz: float | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        i = self._row_from_collection_index(collection_index) if collection_index is not None else int(row)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

        self.plot_spatial_metric("peak_freq", ax=axes[0, 0])
        axes[0, 0].scatter(
            [self.data.x_mm[i]],
            [self.data.z_mm[i]],
            s=90,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
        )

        self.plot_waveform(row=i, ax=axes[0, 1])
        self.plot_spatial_metric("band_ratio", band_hz=band_hz, ax=axes[1, 0])
        axes[1, 0].scatter(
            [self.data.x_mm[i]],
            [self.data.z_mm[i]],
            s=90,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
        )
        self.plot_spectrum(row=i, max_hz=max_hz, ax=axes[1, 1])

        return fig, axes


# Convenience wrappers for notebook usage

def plot_spatial_metric(database: Any, metric: str = "rms", **kwargs: Any) -> plt.Axes:
    return AcousticsPlotter(database).plot_spatial_metric(metric=metric, **kwargs)


def plot_waveform(database: Any, **kwargs: Any) -> plt.Axes:
    return AcousticsPlotter(database).plot_waveform(**kwargs)


def plot_spectrum(database: Any, **kwargs: Any) -> plt.Axes:
    return AcousticsPlotter(database).plot_spectrum(**kwargs)


def plot_spatial_spectral_dashboard(database: Any, **kwargs: Any) -> tuple[plt.Figure, np.ndarray]:
    return AcousticsPlotter(database).plot_spatial_spectral_dashboard(**kwargs)
