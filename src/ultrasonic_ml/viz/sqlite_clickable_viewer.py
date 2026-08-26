from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class AcousticsViewer:
    """Interactive viewer for raw and analyzed waveforms."""

    def __init__(
        self,
        db,
        waveforms=None,
        analyses=None,
        *,
        figsize=(20, 10),
        x_limits=None,
        y_limits=None,
    ):
        self.db = db
        self.waveforms = waveforms or db.waveform_columns
        self.analyses = analyses or []
        self.figsize = figsize

        self.row = 0
        self.n = db.get_acquisition_count()

        self.x_limits = x_limits or self._get_x_limits()
        self.y_limits = y_limits or self._get_y_limits()

        self.fig = None
        self.ax = None
        self.slider = None

        self.raw_lines = {}
        self.analysis_lines = {}
        self.mode="raw"

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def _get_x_limits(self):
        time = self.db.fetch_time(0)
        return float(np.min(time)), float(np.max(time))

    def _get_y_limits(self):
        maxima = []

        for waveform in self.waveforms:
            if waveform in self.db.absolute_max:
                maxima.append(self.db.absolute_max[waveform])

        if not maxima:
            return -1.0, 1.0

        maximum = max(maxima)
        return -maximum, maximum

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def fetch_raw(self, waveform, row=None):
        row = self.row if row is None else row
        return self.db.fetch_waveform(waveform, row)

    def fetch_analyzed(self, waveform, analysis_name, result_name, row=None, reference_id=None):
        row = self.row if row is None else row
        collection_index = self.db.get_acquisition_index(row)

        return self.db.fetch_analysis_result(
            collection_index,
            waveform,
            analysis_name,
            result_name,
            reference_id,
        )

    def fetch_time(self, row=None):
        row = self.row if row is None else row
        return self.db.fetch_time(row)

    def fetch_analyzed_labels(self, row=None):
        row = self.row if row is None else row
        collection_index = self.db.get_acquisition_index(row)
        
        if self.mode=="raw":
            return 'Time', '(ns)', 'Voltage', '(mV)'
        
        if self.mode=="analysis":
            # collection_index,waveform,analysis_name,result_name,x_axis,x_unit,y_axis,y_unit,reference_id
            return self.db.list_analysis_results(collection_index, self.analyses[0]['waveform'], self.analyses[0]['analysis_name'], self.analyses[0]['result_name'])
            
        return self.db.fetch_x_labels(row)
    
    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def build(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)


        self._setup_axes()
        self._create_slider()
        self._connect_events()
        self.update()

        self.fig.tight_layout(rect=(0.15, 0.15, 0.85, 0.85))
        return self.fig, self.ax

    def _setup_axes(self):
        self.ax.set_xlim(*self.x_limits)
        self.ax.set_ylim(*self.y_limits)

        labels = self.fetch_analyzed_labels()[0]
        
        self.ax.set_xlabel(f"{labels['x_axis']} {labels['x_unit']}")
        self.ax.set_ylabel(f"{labels['y_axis']} {labels['y_unit']}")
        self.ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_raw(self):
        """Plot all selected raw waveforms."""
        time = self.fetch_time()

        for waveform in self.waveforms:
            data = np.asarray(self.fetch_raw(waveform))
            n = min(len(time), len(data))

            if waveform not in self.raw_lines:
                self.raw_lines[waveform], = self.ax.plot(
                    time[:n],
                    data[:n],
                    lw=1,
                    label=waveform,
                )
            else:
                self.raw_lines[waveform].set_data(
                    time[:n],
                    data[:n],
                )

    def plot_analyzed(self):
        """Plot all selected analysis results."""
        time = self.fetch_time()

        for analysis in self.analyses:
            waveform = analysis["waveform"]
            analysis_name = analysis["analysis_name"]
            result_name = analysis["result_name"]
            reference_id = analysis.get("reference_id")

            data = np.asarray(
                self.fetch_analyzed(
                    waveform,
                    analysis_name,
                    result_name,
                    reference_id=reference_id,
                )
            )

            n = min(len(time), len(data))
            key = (
                waveform,
                analysis_name,
                result_name,
                reference_id,
            )

            if key not in self.analysis_lines:
                self.analysis_lines[key], = self.ax.plot(
                    time[:n],
                    data[:n],
                    lw=1,
                    label=f"{waveform}: {result_name}",
                )
            else:
                self.analysis_lines[key].set_data(
                    time[:n],
                    data[:n],
                )
    
    def update(self):
        self.ax.clear()
        self.raw_lines.clear()
        self.analysis_lines.clear()

        if self.mode == "raw":
            self.plot_raw()
        elif self.mode == "analysis":
            self.plot_analyzed()

        self._setup_axes()
        self.ax.legend()

        self.ax.set_title(
            f"Collection index: {self.db.get_acquisition_index(self.row)}"
        )

        self.refresh()
        
    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _create_slider(self):
        slider_ax = self.fig.add_axes(
            (0.25, 0.02, 0.55, 0.02)
        )

        self.slider = Slider(
            slider_ax,
            "Collection ",
            0,
            self.n - 1,
            valinit=0,
            valstep=1,
        )

        self.slider.on_changed(self._on_slider)

    def _on_slider(self, value):
        self.row = int(value)
        self.update()

    def goto(self, row):
        row = int(np.clip(row, 0, self.n - 1))
        self.slider.set_val(row)

    def next(self):
        self.goto(self.row + 1)

    def previous(self):
        self.goto(self.row - 1)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _connect_events(self):
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self._on_click,
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )

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
    
    def show_raw(self):
        self.mode = "raw"
        if self.fig is None:
            self.build()
        else:
            self.update()

        plt.show()

    def show_preprocessed(self):
        self.mode = "analysis"
        self.analyses = [ {
                "waveform": waveform,
                "analysis_name": "preprocessed",
                "result_name": "waveform",}
            for waveform in self.waveforms]

        self.raw_lines.clear()
        self.analysis_lines.clear()

        if self.fig is None: self.build()
        else: self.update()

        plt.show()