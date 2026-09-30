from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import hilbert
from ultrasonic_ml.data.sqlite_acoustic_data import *

plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'figure.dpi': 200,
        'savefig.dpi': 200, })

def get_max_fig_dims(margin=0.9, fallback=(1400, 900)):
    """
    Determine usable screen dimensions in pixels.
    current screen's resolution, so tk windows never exceed screen bounds.

    Parameters
    ----------
    margin : float
        Fraction of screen dimensions to use, leaving room for window chrome/taskbar.
    fallback : tuple[int, int]
        Screen dimensions in pixels if detection fails.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # don't actually show a blank window
        screen_width_px = root.winfo_screenwidth()
        screen_height_px = root.winfo_screenheight()
        root.destroy()
        return screen_width_px * margin, screen_height_px * margin
    except Exception:
        return fallback

max_figwidth, max_figheight = get_max_fig_dims(margin=1.0)
print(f'tk max_dimensions: {max_figwidth:.0f} x {max_figheight:.0f} px')

# TODO: with lag, try using functools.lru_cache


class AcousticsViewer:
    """Simple waveform viewer."""

    def __init__(self, db, figsize: tuple[float, float] | None = None) -> None:
        self.db = db
        self.waveforms = db.waveform_columns
        self.collection_idx = 0
        self.n = db.get_acquisition_count()
        dpi = plt.rcParams.get("figure.dpi", 100)
        self.figsize = figsize or (0.8 * max_figwidth / dpi, 0.8 * max_figheight / dpi)
        self.fig = None
        self.line_ax = None
        self.slider = None
        self.lines = {}

    ################################################################
    # initialization
    ################################################################

    def build_slider(self, parent, on_change, maximum=None, label="Collection"):
        """Build a Tk slider."""
        slider = tk.Scale(parent, label=label, from_=0, to=self.n - 1 if maximum is None else maximum, orient="horizontal", command=lambda value: on_change(int(float(value))))
        slider.set(0)
        slider.pack(fill="x")
        self.slider = slider
        return slider

    def format_toolbar(self):
        """Increase toolbar control size for Tk displays."""
        self.toolbar.configure(height=38)
        for button in self.toolbar._buttons.values():
            button.configure(width=24, height=24, padx=2, pady=2)
            button.pack_configure(padx=1, pady=1)
        self.toolbar._message_label.configure(font=("TkDefaultFont", 12))

    def reset_view_state(self):
        """Clear artists belonging to a previously closed view."""
        self.lines.clear()

    def create_tk_view(self, title, fig):
        """Create a Tk window containing a Matplotlib canvas and controls."""
        self.reset_view_state()
        root = tk.Tk()
        root.title(title)

        controls = tk.Frame(root, height=52)
        controls.pack(side="bottom", fill="x", pady=(0, 12))
        controls.pack_propagate(False)

        toolbar_frame = tk.Frame(root, height=45)
        toolbar_frame.pack(side="bottom", fill="x")
        toolbar_frame.pack_propagate(False)

        canvas = FigureCanvasTkAgg(fig, master=root)
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        self.toolbar.update()
        self.format_toolbar()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return root, controls, canvas

    def close_tk_view(self, root, canvas):
        """Cancel Tk callbacks before destroying a viewer window."""
        for callback in ("_idle_draw_id", "_event_loop_id"):
            callback_id = getattr(canvas, callback, None)
            if callback_id is not None:
                try:
                    root.after_cancel(callback_id)
                except tk.TclError:
                    pass
                setattr(canvas, callback, None)
        plt.close(self.fig)
        root.quit()
        root.destroy()

    ################################################################
    # plotting
    ################################################################
    
    def plot_raw_waveform(self, ax, row: int | None = None):
        """Plot raw waveforms and Hilbert envelopes for one acquisition.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        row : int, optional
            The row index of the acquisition to plot. If None, uses self.collection_idx.
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plotted waveforms.
        """
        row = self.collection_idx if row is None else row
        time = self.db.fetch_time(row)
        for waveform in self.waveforms:
            data = self.db.fetch_waveform(waveform, row)
            envelope = f"hilbert_envelope_{waveform}"
            if waveform not in self.lines:
                self.lines[waveform], = ax.plot(time, data, lw=1, label=waveform)
                self.lines[envelope], = ax.plot(time, np.abs(hilbert(data)), lw=0.5, linestyle="--", color=self.lines[waveform].get_color())
            else:
                self.lines[waveform].set_data(time, data)
                self.lines[envelope].set_data(time, np.abs(hilbert(data)))

        return ax

    def plot_preprocessed(self, ax, row: int | None = None):
        """Plot preprocessed waveforms and Hilbert envelopes for one acquisition."""
        row = self.collection_idx if row is None else row
        time = self.db.fetch_time(row)
        for waveform in self.waveforms:
            data = self.db.fetch_preprocessed_waveform(waveform, row)
            envelope = f"hilbert_envelope_{waveform}"
            if waveform not in self.lines:
                self.lines[waveform], = ax.plot(time, data, lw=1, label=waveform)
                self.lines[envelope], = ax.plot(time, np.abs(hilbert(data)), lw=0.5, linestyle="--", color=self.lines[waveform].get_color())
            else:
                self.lines[waveform].set_data(time, data)
                self.lines[envelope].set_data(time, np.abs(hilbert(data)))
        return ax

    ################################################################
    # formatting
    ################################################################

    def format_waveform_ax(self, ax, keys_=None):
        """Format a waveform axis and apply the database amplitude limits."""
        maximum = max(self.db.raw_maxs_.values()) if keys_ is None else max(self.db.analysis_maxs_[keys_].values())
        ax.set_ylim(-maximum, maximum)
        ax.set_autoscale_on(False)
        ax.set_ylabel("Voltage (mV)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e-3:g}"))
        ax.set_xlabel("Time (us)")
        ax.legend()

    ################################################################
    # standalone viewers
    ################################################################
    
    def show_raw(self) -> None:
        """Show raw waveforms with a collection slider."""
        self.fig, self.line_ax = plt.subplots(1, 1, figsize=self.figsize)
        root, controls, canvas = self.create_tk_view("Raw waveforms", self.fig)
        self.plot_raw_waveform(self.line_ax)
        self.format_waveform_ax(self.line_ax)
        self.line_ax.set_title(f"Collection index: {self.collection_idx}")

        def update(val):
            self.collection_idx = int(val)
            self.plot_raw_waveform(self.line_ax, self.collection_idx)
            self.line_ax.set_title(f"Collection index: {self.collection_idx}")
            canvas.draw_idle()

        self.build_slider(controls, update)
        self.fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.2)
        canvas.draw()
        root.protocol("WM_DELETE_WINDOW", lambda: self.close_tk_view(root, canvas))
        root.mainloop()

    def show_preprocessed(self) -> None:
        """Show preprocessed waveforms with a collection slider."""
        self.fig, self.line_ax = plt.subplots(1, 1, figsize=self.figsize)
        root, controls, canvas = self.create_tk_view("Preprocessed waveforms", self.fig)
        self.plot_preprocessed(self.line_ax)
        self.format_waveform_ax(self.line_ax, keys_=('hilbert_window','preprocessed_waveform', 2))
        self.line_ax.set_title(f"Collection index: {self.collection_idx}")

        def update(val):
            self.collection_idx = int(val)
            self.plot_preprocessed(self.line_ax, self.collection_idx)
            self.line_ax.set_title(f"Collection index: {self.collection_idx}")
            canvas.draw_idle()

        self.build_slider(controls, update)
        self.fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.2)
        canvas.draw()
        root.protocol("WM_DELETE_WINDOW", lambda: self.close_tk_view(root, canvas))
        root.mainloop()


class AcousticsScanViewer(AcousticsViewer):
    """Tk viewer for scan data shaped as (x, z, sample)."""

    def __init__(self, db, figsize: tuple[float, float] | None = None) -> None:
        AcousticsViewer.__init__(self, db, figsize)
        self.img_ax = None
        self.line_ax = None
        self.scan_img = None
        self.scan_colorbar = None
        self.cursor = None
        self.line_cursor = None
        self.scan_waveform_line = None
        self.scan_hilbert_line = None
        self.grid = None
        self.grid_waveform = None
        self.grid_reference_id = None
        self.loaded_reference_id = None
        self.data_mode = "raw"
        self.waveform = self.waveforms[0] if self.waveforms else None
        self.x = 0
        self.z = 0
        self.y = 0
        self.time_value = 0.0

    ################################################################
    # state and data
    ################################################################

    def reset_view_state(self):
        """Reset artists and cached references for a new Tk window."""
        AcousticsViewer.reset_view_state(self)
        self.scan_img = None
        self.scan_colorbar = None
        self.cursor = None
        self.line_cursor = None
        self.scan_waveform_line = None
        self.scan_hilbert_line = None
        self.grid = None
        self.grid_waveform = None
        self.grid_reference_id = None
        self.loaded_reference_id = None
        self.time_value = 0.0

    def _get_processed_reference(self):
        return self.db.fetch_latest_analysis_reference(
            self.waveform,
            "preprocessed",
            "waveform",
        )

    def _load_grid(self):
        """Load the active raw or processed grid only when its key changes."""
        if self.waveform is None:
            return None

        if self.data_mode == "hilbert":
            key_changed = (
                self.grid_waveform != self.waveform
                or self.loaded_reference_id != self.grid_reference_id
            )
            if key_changed or self.grid is None:
                self.grid = self.db.fetch_hilbert_grid(
                    self.waveform,
                    self.grid_reference_id,
                )
        elif self.data_mode == "processed":
            key_changed = (
                self.grid_waveform != self.waveform
                or self.loaded_reference_id != self.grid_reference_id
            )
            if key_changed or self.grid is None:
                self.grid = self.db.fetch_analysis_grid(
                    self.waveform,
                    "preprocessed",
                    "waveform",
                    self.grid_reference_id,
                )
        elif self.grid_waveform != self.waveform or self.grid is None:
            self.grid = self.db.fetch_waveform_grid(self.waveform)

        self.grid_waveform = self.waveform
        self.loaded_reference_id = self.grid_reference_id
        return self.grid

    ################################################################
    # plotting
    ################################################################

    def plot_scan_image(self, ax):
        """Update the image slice for the active sample index."""
        grid = self._load_grid()
        if grid is None:
            return ax

        self.y = int(np.clip(self.y, 0, grid.shape[-1] - 1))
        image = grid[:, :, self.y] if self.data_mode != "raw" else grid[:, :, self.y].T
        if self.scan_img is None:
            self.scan_img = ax.imshow(
                image,
                aspect="auto",
                origin="upper",
                extent=(0, self.db.X_, -self.db.Z_, 0),
                cmap="viridis",
            )
        else:
            self.scan_img.set_data(image)

        if self.cursor is None:
            self.cursor = ax.scatter(self.x, -self.z, color="red", s=10)
        else:
            self.cursor.set_offsets([(self.x, -self.z)])
        return ax

    def plot_scan_waveform(self, ax):
        """Update the selected waveform at the active image coordinate."""
        grid = self._load_grid()
        if grid is None:
            return ax

        row = int(self.db.fetch_collection_index(self.x, self.z))
        time = self.db.fetch_time(row)
        if self.data_mode == "raw":
            data = grid[self.x, self.z, :]
            envelope = np.abs(hilbert(data))
        elif self.data_mode == "hilbert":
            processed_grid = self.db.fetch_analysis_grid(
                self.waveform,
                "preprocessed",
                "waveform",
                self.grid_reference_id,
            )
            data = processed_grid[self.z, self.x, :]
            envelope = grid[self.z, self.x, :]
        else:
            data = grid[self.z, self.x, :]
            envelope = np.abs(hilbert(data))
        self.y = int(np.clip(self.y, 0, len(time) - 1))
        self.time_value = float(time[self.y])

        label = f"{self.waveform} @ ({self.x}, {self.z})"
        if self.scan_waveform_line is None:
            self.scan_waveform_line, = ax.plot(time, data, lw=1, label=label)
        else:
            self.scan_waveform_line.set_data(time, data)
            self.scan_waveform_line.set_label(label)

        if self.scan_hilbert_line is None:
            self.scan_hilbert_line, = ax.plot(
                time,
                envelope,
                color=self.scan_waveform_line.get_color(),
                lw=0.75,
                linestyle=":",
            )
        else:
            self.scan_hilbert_line.set_data(time, envelope)

        maximum = max(
            float(np.max(np.abs(data))),
            float(np.max(np.abs(envelope))),
            np.finfo(float).eps,
        )
        ax.set_ylim(-maximum, maximum)
        ax.set_autoscale_on(False)
        cursor_time = time[self.y]
        if self.line_cursor is None:
            self.line_cursor = ax.axvline(
                cursor_time, color="red", lw=0.5, linestyle="--"
            )
        else:
            self.line_cursor.set_xdata([cursor_time, cursor_time])
        return ax

    def _get_scan_coordinates(self):
        """Convert scan indices to physical coordinates from database spacing."""
        x_step = abs(float(self.db.parameters["primaryAxisStep"]))
        z_step = abs(float(self.db.parameters["secondaryAxisStep"]))
        return self.x * x_step, -self.z * z_step

    def format_scan_ax(self, ax):
        """Format the scan image axis and its color scale."""
        if self.scan_img is None:
            return ax
        if self.data_mode != "raw":
            maximum = float(np.max(np.abs(self.grid[:, :, self.y])))
        else:
            maximum = max(self.db.raw_maxs_.values())
        self.scan_img.set_clim(-maximum, maximum)
        ax.set_autoscale_on(False)
        if self.scan_colorbar is None:
            self.scan_colorbar = plt.colorbar(self.scan_img, ax=ax, label="Amplitude")
        ax.set_xlabel("X")
        ax.set_xlim(0, self.db.X_)
        ax.set_ylabel("Z")
        ax.set_ylim(-self.db.Z_, 0)
        return ax

    def format_scan_waveform_ax(self, ax):
        """Format the scan waveform using the displayed trace amplitude."""
        data = self.scan_waveform_line.get_ydata()
        maximum = float(np.max(np.abs(data)))
        ax.set_ylim(-maximum, maximum)
        ax.set_autoscale_on(False)
        ax.set_ylabel("Voltage (mV)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e-3:g}"))
        ax.set_xlabel("Time (us)")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    def _refresh_view(self):
        """Update both panes and the shared title."""
        self.plot_scan_image(self.img_ax)
        self.plot_scan_waveform(self.line_ax)
        x_coordinate, z_coordinate = self._get_scan_coordinates()
        self.fig.suptitle(
            f"{self.waveform} @ t={self.time_value * 1e-3:g} us, "
            f"(x, z)=({x_coordinate:g}, {z_coordinate:g})"
        )
        self.fig.canvas.draw_idle()

    ################################################################
    # controls
    ################################################################

    def build_dropdown(self, parent):
        dropdown = ttk.Combobox(parent, values=self.waveforms, state="readonly")
        dropdown.set(self.waveform)
        dropdown.bind("<<ComboboxSelected>>", lambda event: self._set_waveform(dropdown.get()))
        dropdown.pack(side="left", fill="x", expand=True)
        return dropdown

    def build_click(self, ax, axis):
        """Connect image x/z or waveform time clicks to the viewer state."""
        def _click(event):
            if self.toolbar.mode or event.inaxes is not ax or event.xdata is None:
                return
            if axis == "image" and event.ydata is not None:
                self._set_position(event.xdata, event.ydata)
            elif axis == "waveform":
                self._set_sample(event.xdata)
        return ax.figure.canvas.mpl_connect("button_press_event", _click)

    def _set_waveform(self, name):
        self.waveform = name
        if self.data_mode != "raw":
            self.grid_reference_id = self._get_processed_reference()
        self.grid = None
        self._refresh_view()

    def _set_sample(self, time_value):
        _, self.y = self.db.fetch_time_index(float(time_value))
        self._refresh_view()

    def _set_position(self, x, displayed_z):
        self.x = int(np.clip(round(x), 0, self.db.X_ - 1))
        self.z = int(np.clip(round(-displayed_z), 0, self.db.Z_ - 1))
        self._refresh_view()

    ################################################################
    # windows
    ################################################################

    def _show_scan(self, data_mode):
        self.data_mode = data_mode
        self.y = 0
        self.fig, (self.img_ax, self.line_ax) = plt.subplots(2, 1, figsize=self.figsize)
        root, controls, canvas = self.create_tk_view(
            "Hilbert scan viewer" if data_mode == "hilbert"
            else "Preprocessed scan viewer" if data_mode == "processed"
            else "Scan viewer",
            self.fig,
        )
        self.grid_reference_id = (
            self._get_processed_reference() if data_mode != "raw" else None
        )

        self._refresh_view()
        self.format_scan_ax(self.img_ax)
        self.format_scan_waveform_ax(self.line_ax)
        self.dropdown = self.build_dropdown(controls)
        self.img_click = self.build_click(self.img_ax, "image")
        self.line_click = self.build_click(self.line_ax, "waveform")
        self.fig.tight_layout()
        canvas.draw()
        root.protocol("WM_DELETE_WINDOW", lambda: self.close_tk_view(root, canvas))
        root.mainloop()

    def show_raw_scan(self):
        """Show raw image slices and waveforms for 3D scan data."""
        self._show_scan("raw")

    def show_processed_scan(self):
        """Show preprocessed image slices and waveforms for 3D scan data."""
        self._show_scan("processed")

    def show_hilbert_scan(self):
        """Show Hilbert envelopes of preprocessed scan data."""
        self._show_scan("hilbert")
        

# class AcousticsScanViewer_old(AcousticsViewer):
#     """Tk viewer for scan data shaped as (x, z, sample)."""

#     def __init__(self, db, figsize: tuple[float, float] | None = None) -> None:
#         AcousticsViewer.__init__(self, db, figsize)
#         self.img_ax = None
#         self.line_ax = None
#         self.scan_img = None
#         self.scan_colorbar = None
#         self.cursor = None
#         self.line_cursor = None
#         self.scan_waveform_line = None
#         self.grid = None
#         self.grid_waveform = None
#         self.grid_reference_id = None
#         self.loaded_reference_id = None
#         self.processed = False
#         self.waveform = self.waveforms[0] if self.waveforms else None
#         self.x = 0
#         self.z = 0
#         self.y = 0
#         self.time_value = 0.0

#     ################################################################
#     # state and data
#     ################################################################

#     def reset_view_state(self):
#         """Reset artists and cached references for a new Tk window."""
#         AcousticsViewer.reset_view_state(self)
#         self.scan_img = None
#         self.scan_colorbar = None
#         self.cursor = None
#         self.line_cursor = None
#         self.scan_waveform_line = None
#         self.grid = None
#         self.grid_waveform = None
#         self.grid_reference_id = None
#         self.loaded_reference_id = None
#         self.time_value = 0.0

#     def _get_processed_reference(self):
#         return self.db.fetch_latest_analysis_reference(
#             self.waveform,
#             "preprocessed",
#             "waveform",
#         )

#     def _load_grid(self):
#         """Load the active raw or processed grid only when its key changes."""
#         if self.waveform is None:
#             return None

#         if self.processed:
#             key_changed = (
#                 self.grid_waveform != self.waveform
#                 or self.loaded_reference_id != self.grid_reference_id
#             )
#             if key_changed or self.grid is None:
#                 self.grid = self.db.fetch_analysis_grid(
#                     self.waveform,
#                     "preprocessed",
#                     "waveform",
#                     self.grid_reference_id,
#                 )
#         elif self.grid_waveform != self.waveform or self.grid is None:
#             self.grid = self.db.fetch_waveform_grid(self.waveform)

#         self.grid_waveform = self.waveform
#         self.loaded_reference_id = self.grid_reference_id
#         return self.grid

#     ################################################################
#     # plotting
#     ################################################################

#     def plot_scan_image(self, ax):
#         """Update the image slice for the active sample index."""
#         grid = self._load_grid()
#         if grid is None:
#             return ax

#         self.y = int(np.clip(self.y, 0, grid.shape[-1] - 1))
#         image = grid[:, :, self.y] if self.processed else grid[:, :, self.y].T
#         if self.scan_img is None:
#             self.scan_img = ax.imshow(
#                 image,
#                 aspect="auto",
#                 origin="upper",
#                 extent=(0, self.db.X_, -self.db.Z_, 0),
#                 cmap="viridis",
#             )
#         else:
#             self.scan_img.set_data(image)

#         if self.cursor is None:
#             self.cursor = ax.scatter(self.x, -self.z, color="red", s=10)
#         else:
#             self.cursor.set_offsets([(self.x, -self.z)])
#         return ax

#     def plot_scan_waveform(self, ax):
#         """Update the selected waveform at the active image coordinate."""
#         grid = self._load_grid()
#         if grid is None:
#             return ax

#         row = int(self.db.fetch_collection_index(self.x, self.z))
#         time = self.db.fetch_time(row)
#         data = grid[self.z, self.x, :] if self.processed else grid[self.x, self.z, :]
#         self.y = int(np.clip(self.y, 0, len(time) - 1))
#         self.time_value = float(time[self.y])

#         label = f"{self.waveform} @ ({self.x}, {self.z})"
#         if self.scan_waveform_line is None:
#             self.scan_waveform_line, = ax.plot(time, data, lw=1, label=label)
#         else:
#             self.scan_waveform_line.set_data(time, data)
#             self.scan_waveform_line.set_label(label)

#         maximum = max(float(np.max(np.abs(data))), np.finfo(float).eps)
#         ax.set_ylim(-maximum, maximum)
#         ax.set_autoscale_on(False)
#         cursor_time = time[self.y]
#         if self.line_cursor is None:
#             self.line_cursor = ax.axvline(
#                 cursor_time, color="red", lw=0.5, linestyle="--"
#             )
#         else:
#             self.line_cursor.set_xdata([cursor_time, cursor_time])
#         return ax

#     def _get_scan_coordinates(self):
#         """Convert scan indices to physical coordinates from database spacing."""
#         x_step = abs(float(self.db.parameters["primaryAxisStep"]))
#         z_step = abs(float(self.db.parameters["secondaryAxisStep"]))
#         return self.x * x_step, -self.z * z_step

#     def format_scan_ax(self, ax):
#         """Format the scan image axis and its color scale."""
#         if self.scan_img is None:
#             return ax
#         if self.processed:
#             maximum = float(np.max(np.abs(self.grid[:, :, self.y])))
#         else:
#             maximum = max(self.db.raw_maxs_.values())
#         self.scan_img.set_clim(-maximum, maximum)
#         ax.set_autoscale_on(False)
#         if self.scan_colorbar is None:
#             self.scan_colorbar = plt.colorbar(self.scan_img, ax=ax, label="Amplitude")
#         ax.set_xlabel("X")
#         ax.set_xlim(0, self.db.X_)
#         ax.set_ylabel("Z")
#         ax.set_ylim(-self.db.Z_, 0)
#         return ax

#     def format_scan_waveform_ax(self, ax):
#         """Format the scan waveform using the displayed trace amplitude."""
#         data = self.scan_waveform_line.get_ydata()
#         maximum = float(np.max(np.abs(data)))
#         ax.set_ylim(-maximum, maximum)
#         ax.set_autoscale_on(False)
#         ax.set_ylabel("Voltage (mV)")
#         ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e-3:g}"))
#         ax.set_xlabel("Time (us)")
#         legend = ax.get_legend()
#         if legend is not None:
#             legend.remove()

#     def _refresh_view(self):
#         """Update both panes and the shared title."""
#         self.plot_scan_image(self.img_ax)
#         self.plot_scan_waveform(self.line_ax)
#         x_coordinate, z_coordinate = self._get_scan_coordinates()
#         self.fig.suptitle(
#             f"{self.waveform} @ t={self.time_value * 1e-3:g} us, "
#             f"(x, z)=({x_coordinate:g}, {z_coordinate:g})"
#         )
#         self.fig.canvas.draw_idle()

#     ################################################################
#     # controls
#     ################################################################

#     def build_dropdown(self, parent):
#         dropdown = ttk.Combobox(parent, values=self.waveforms, state="readonly")
#         dropdown.set(self.waveform)
#         dropdown.bind("<<ComboboxSelected>>", lambda event: self._set_waveform(dropdown.get()))
#         dropdown.pack(side="left", fill="x", expand=True)
#         return dropdown

#     def build_click(self, ax, axis):
#         """Connect image x/z or waveform time clicks to the viewer state."""
#         def _click(event):
#             if self.toolbar.mode or event.inaxes is not ax or event.xdata is None:
#                 return
#             if axis == "image" and event.ydata is not None:
#                 self._set_position(event.xdata, event.ydata)
#             elif axis == "waveform":
#                 self._set_sample(event.xdata)
#         return ax.figure.canvas.mpl_connect("button_press_event", _click)

#     def _set_waveform(self, name):
#         self.waveform = name
#         if self.processed:
#             self.grid_reference_id = self._get_processed_reference()
#         self.grid = None
#         self._refresh_view()

#     def _set_sample(self, time_value):
#         _, self.y = self.db.fetch_time_index(float(time_value))
#         self._refresh_view()

#     def _set_position(self, x, displayed_z):
#         self.x = int(np.clip(round(x), 0, self.db.X_ - 1))
#         self.z = int(np.clip(round(-displayed_z), 0, self.db.Z_ - 1))
#         self._refresh_view()

#     ################################################################
#     # windows
#     ################################################################

#     def _show_scan(self, processed):
#         self.processed = processed
#         self.y = 0
#         self.fig, (self.img_ax, self.line_ax) = plt.subplots(2, 1, figsize=self.figsize)
#         root, controls, canvas = self.create_tk_view(
#             "Preprocessed scan viewer" if processed else "Scan viewer",
#             self.fig,
#         )
#         self.grid_reference_id = self._get_processed_reference() if processed else None

#         self._refresh_view()
#         self.format_scan_ax(self.img_ax)
#         self.format_scan_waveform_ax(self.line_ax)
#         self.dropdown = self.build_dropdown(controls)
#         self.img_click = self.build_click(self.img_ax, "image")
#         self.line_click = self.build_click(self.line_ax, "waveform")
#         self.fig.tight_layout()
#         canvas.draw()
#         root.protocol("WM_DELETE_WINDOW", lambda: self.close_tk_view(root, canvas))
#         root.mainloop()

#     def show_raw_scan(self):
#         """Show raw image slices and waveforms for 3D scan data."""
#         self._show_scan(processed=False)

#     def show_preprocessed_scan(self):
#         """Show preprocessed image slices and waveforms for 3D scan data."""
#         self._show_scan(processed=True)
        
