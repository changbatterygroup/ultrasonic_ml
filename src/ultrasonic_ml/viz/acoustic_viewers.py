import h5py
import numpy as np
from functools import lru_cache

import param
import panel as pn
import holoviews as hv
from holoviews import streams

from bokeh.models import CustomJSTickFormatter



class AcousticViewer(param.Parameterized):
    # put tracked parameters here

    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------
    
    def __init__(self, db, from_group='preprocessed_data', **params):
        super().__init__(**params)
        self.db = db
        self.from_group = from_group
        self.labels = db.parameters['waveform_labels']
        self.h5 = None
        self._is_open = False
        
        self.open()
        
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------
    
    def open(self):
        if self._is_open: return

        self.h5 = h5py.File(self.db.h5_name, 'r')
        self._open_datasets()
        self._is_open = True

    def close(self):
        if not self._is_open: return
        self._close_datasets()
        self.clear_cache()
        if self.h5 is not None: self.h5.close()
        self.h5 = None
        self._is_open = False
        print("AcousticViewer closed and cache cleared.")

    def clear_cache(self): pass # call .cache_clear() on all methods with header @lrucache

    def _open_datasets(self): raise NotImplementedError # methods for setting self.datasets, self.shape, self.time,

    def _close_datasets(self): pass
    
    # -------------------------------------------------------------------------
    # Data Fetchers/ Interactive grid components
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------------
    
    def create_dashboard(self): raise NotImplementedError

    @classmethod
    def launch(cls, **kwargs):
        with cls(**kwargs) as visualizer:
            server = pn.serve( visualizer.create_dashboard(), show=True, threaded=True )
            try: input("Use the viewer, then press Enter here to close it...")
            finally: server.stop()
            
            
class AcousticScanViewer(AcousticViewer):
    current_x = param.Integer(default=0)
    current_z = param.Integer(default=0)
    current_y = param.Integer(default=0)
    
    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------
    
    def __init__(self, db, from_group='preprocessed_data', **params):
        super().__init__(db, from_group, **params)
        
        self.aspect_ratio = 10
        self._spatial_tap_streams = {}
        self._waveform_tap_stream = None
        
        self.current_x = db.shape[1] // 2
        self.current_z = db.shape[2] // 2
        self.current_y = db.shape[3] // 2
        
    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------

    def clear_cache(self): self._fetch_1d_waveform.cache_clear()    
    
    def _open_datasets(self):
        group = f"/{self.from_group}"
        self.acoustic_ds = self.h5[f"{group}/waveforms"]
        self.hilbert_ds = self.h5[f"{group}/hilbert"]
        
        self.shape = self.h5[f"{group}/waveforms"].shape
        self.time = self.acoustic_ds.attrs['time_array']

    def _close_datasets(self):
        self.acoustic_ds = None
        self.hilbert_ds = None
    
    # -------------------------------------------------------------------------
    # Data Fetchers
    # -------------------------------------------------------------------------

    def get_2d_slice(self, w: int, y: int, dataset: str = 'hilbert') -> np.ndarray:
        if dataset == 'hilbert': source = self.hilbert_ds
        elif dataset == 'acoustic': source = self.acoustic_ds
        else: raise ValueError("dataset must be 'hilbert' or 'acoustic'")
        return source[int(w), :, :, int(y)]

    @lru_cache(maxsize=128)
    def _fetch_1d_waveform(self, w: int, x: int, z: int): return self.acoustic_ds[w, x, z, :], self.hilbert_ds[w, x, z, :]

    def get_1d_waveform(self, w: int, x: int, z: int) -> tuple: return self._fetch_1d_waveform(int(w), int(x), -int(z))
    
    # -------------------------------------------------------------------------
    # 2D Image Plot 
    # -------------------------------------------------------------------------
    
    #TODO: get rid of prints
    def build_spatial_grid(self, direction_idx: int, dataset: str = 'hilbert'):
        """
        Generates a 2D spatial scan grid for a specific waveform layout plane.
        Includes an active crosshair tracking where you click.
        """
        tap_stream = streams.Tap(x=self.current_x, y=self.current_z)

        def spatial_renderer(current_y, current_x, current_z, x, y):
            # Fetch slice matrix
            grid_data = self.get_2d_slice(direction_idx, current_y, dataset)
            
            # Create a HoloViews Image object for the 2D grid
            img = hv.Image(
                grid_data.T, 
                kdims=['x', 'z'], 
                vdims=['Intensity'],
                bounds=(0, 0, self.shape[1], self.shape[2]) #(x0, y0, x1, y1) # changing has no effect
            ).opts(
                title=f"{self.labels[direction_idx]} {dataset.title()} Grid (Time Y={current_y})",
                cmap='Viridis', colorbar=True, width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio,
                tools=['tap', 'hover'],
            )
            
            # Crosshair Overlay lines mapping active interaction coordinates
            crosshair = hv.VLine(current_x).opts(color='white', line_dash='dashed', line_width=1.5) * \
                        hv.HLine(current_z).opts(color='white', line_dash='dashed', line_width=1.5)
            
            return img * crosshair
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_y', 'current_x', 'current_z'],
        )
        dynamic_grid = hv.DynamicMap(
            spatial_renderer,
            streams=[state_stream, tap_stream],
        )

        def update_coordinates(x, y):
            if x is not None and y is not None:
                self.current_x = int(np.clip(x, 0, self.shape[1] - 1))
                self.current_z = int(np.clip(y, 0, self.shape[2] - 1))

        tap_stream.add_subscriber(update_coordinates)
        self._spatial_tap_streams[direction_idx] = tap_stream
        return dynamic_grid

    # -------------------------------------------------------------------------
    # 1D Waveform Plot
    # -------------------------------------------------------------------------
    
    def build_overlay_waveform_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """
        # Distinguishable color palette for tracking multiple waveform variations
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        time_formatter = CustomJSTickFormatter( 
            args={'times': self.time.tolist()},
            code="""
                const i = Math.round(tick);
                return i >= 0 && i < times.length
                    ? `${times[i]} ns`
                    : "";
                """ )
        line_tap = streams.Tap(x=self.current_y)

        def waveform_renderer(current_x, current_z, current_y, x, y):
            time_axis = np.arange(self.shape[3])
            overlay_elements = []
            
            for w_idx in range(self.shape[0]):
                acoustic_wave, hilbert_win = self.get_1d_waveform(w_idx, current_x, current_z)
                color = palette[w_idx % len(palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (time_axis, acoustic_wave), kdims=['Time (Y)'], vdims=['Amplitude'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8)
                
                # Matching Hilbert trace (uses same color with a distinct dash pattern)
                envelope = hv.Curve(
                    (time_axis, hilbert_win), kdims=['Time (Y)'], vdims=['Amplitude'], 
                    label=f"{label_str} Hilbert"
                ).opts(color=color, line_dash='dotdash', alpha=0.5)
                
                overlay_elements.extend([curve, envelope])
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=time_formatter,

            )
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_x', 'current_z', 'current_y'],
        )
        waveform_map = hv.DynamicMap(
            waveform_renderer,
            streams=[state_stream, line_tap],
        )

        def update_time(x, y):
            if x is not None: self.current_y = int(np.clip(x, 0, self.shape[3] - 1))

        line_tap.add_subscriber(update_time)
        self._waveform_tap_stream = line_tap
        return waveform_map

    # -------------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------------
    def create_dashboard(self):
        """Assembles dynamically computed items into an integrated control dashboard."""
        # Create an individual 2D plot panel row matching each waveform array slice index
        grid_plots = [self.build_spatial_grid(w) for w in range(self.shape[0])]

        # Generate the shared, synchronized 1D timeline trace
        waveform_plot = self.build_overlay_waveform_plot()

        # Format elements cleanly into columns & rows
        dashboard_layout = pn.Column(
            pn.pane.Markdown(f"## 📊 Scanned waveform visualization, ({self.shape[0]} channels)"),
            pn.pane.Markdown("*Click an image to center the spatial crosshair. Click the line plot to select a specific time slice.*"),
            pn.Column(
                pn.Column(*grid_plots), # Stacks image dimensions vertically based on array count
                pn.Card(waveform_plot, title="Synchronized Waveform Array View", margin=(0,10))
            )
        )
        return dashboard_layout



