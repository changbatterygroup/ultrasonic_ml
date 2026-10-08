from __future__ import annotations
import io
import json
import ast
import sqlite3
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Iterator    
import math
import numpy as np  
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from tqdm import tqdm

from ..utils import profile, NumpyEncoder 
import time

import h5py
import matplotlib.pyplot as plt
import pprint

class AcousticSQlite: 
    """Base class for SQLite acoustics data and analysis."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the AcousticsDatabase.

        Parameters
        ----------
        path : str | Path
            The path to the SQLite database file.
        """
        
        print("Establishing SQLite connection to database...")
        self.types_dict = {
            "REAL": float,
            "TEXT": str,
            "INTEGER": int,
            "BLOB": bytes,
            "NULL": type(None),
        }
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.acoustics_table = "acoustics"
        self.parameters_table = "parameters"

        self.connect()
        # print('setting parameters')
        self.waveform_columns = self.get_waveform_columns()
        self.parameters = self.get_sqlite_parameters()
        self.initialize_frequency_parameters()

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def __enter__(self) -> AcousticsSQlite:
        """Enter the context manager.

        Returns
        -------
        AcousticsDatabase
            The AcousticsDatabase instance.
        """
        
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager and close the database connection.

        Parameters
        ----------
        *args : Any
            The exception type, value, and traceback.
        """
        self.close()

    def __len__(self) -> int:
        """Get the number of acquisitions in the acoustics table.

        Returns
        -------
        int
            The number of acquisitions.
        """
        return self.get_acquisition_count()

    def print_schema(self) -> None:
        """Pretty-print the database's table and column structure.
        """
        tables = self.get_tables()
        for i, table in enumerate(tables):
            is_last_table = i == len(tables) - 1
            print(f"{'└── ' if is_last_table else '├── '}{table}")

            columns = self.get_columns(table)
            prefix = "    " if is_last_table else "│   "
            for j, column in enumerate(columns):
                is_last_column = j == len(columns) - 1
                print(f"{prefix}{'└── ' if is_last_column else '├── '}{column}")

    def connect(self) -> None:
        """Connect to the SQLite database.
        """
        self.connection = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def close(self) -> None:
        """Close the SQLite database connection.
        """
        if self.connection:
            self.connection.close()
        self.connection = self.cursor = None

    def get_tables(self) -> list[str]:
        """Get the list of tables in the SQLite database.

        Returns
        -------
        list[str]
            A list of table names.
        """
        rows = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r[0] for r in rows]

    def get_columns(self, table: str) -> list[str]:
        """Get the list of columns in a table.

        Parameters
        ----------
        table : str
            The name of the table.

        Returns
        -------
        list[str]
            A list of column names.
        """
        rows = self.connection.execute(f"PRAGMA table_info({self._quote(table)})").fetchall()
        return [r[1] for r in rows]

    def fetch_column(self, column: str, table: str = "acoustics") -> list[Any]:
        """Fetch all values from a specific column in a table.

        Parameters
        ----------
        column : str
            The name of the column.
        table : str, optional
            The name of the table, by default "acoustics".

        Returns
        -------
        list[Any]
            A list of values from the specified column.
        """
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)}"
        return [r[0] for r in self.connection.execute(query)]

    def fetch_value(self, column: str, row: int, table: str = "acoustics") -> Any:
        """Fetch a single value from a specific column and row in a table.

        Parameters
        ----------
        column : str
            The name of the column.
        row : int
            The row index.
        table : str, optional
            The name of the table, by default "acoustics".

        Returns
        -------
        Any
            The value from the specified column and row.
        """
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)} LIMIT 1 OFFSET ?"
        result = self.connection.execute(query, (row,)).fetchone()
        if result is None:
            raise IndexError(f"Row {row} does not exist.")
        return result[0]

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def get_sqlite_parameters(self) -> dict[str, Any]:
        """Get the parameters from the parameters table.

        Returns
        -------
        dict[str, Any]
            A dictionary of parameters.
        """
        columns = self.get_columns(self.parameters_table)
        if not columns:
            return {}
        query = f"SELECT {','.join(map(self._quote, columns))} FROM {self._quote(self.parameters_table)} LIMIT 1"
        row = self.connection.execute(query).fetchone()
        return dict(zip(columns, row)) if row else {}

    def initialize_frequency_parameters(self) -> None:
        """Initialize frequency-related parameters from the database.
        """
        self.f = self.parameters.get("transducerFrequency")
        self.f = float(self.f) * 1e6 if self.f is not None else None

        try:
            time = self.fetch_time()
            self.dt = float(np.median(np.diff(time))) * 1e-9
            self.f_s = 1 / self.dt
        except (KeyError, ValueError, IndexError):
            self.dt = self.f_s = None

        self.omega_scaled = self.f * self.dt if self.f is not None and self.dt is not None else None

        values = {"omega_scaled": self.omega_scaled, "dt": self.dt, "f_s": self.f_s}
        for key, value in values.items():
            if value is not None:
                self._add_column(key, self.parameters_table)
                self._write_parameter(key, value)

        self.parameters.update({k: v for k, v in values.items() if v is not None})

    # -------------------------------------------------------------------------
    # Raw Waveforms
    # -------------------------------------------------------------------------

    def store_acoustics_values(self, collection_index: int, column: str, value: Any) -> None:
        """Store a single value in the acoustics table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        column : str
            The name of the column to store the value in.
        value : Any
            The value to store.
        """
        self.store_acoustics_values_batch([collection_index], column, [value])

    def store_acoustics_values_batch(self, collection_index: int, column: str, value: Any) -> None:
        """Store values in the analysis table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        """
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)

        query = f"""
            INSERT INTO {self._quote(self.acoustics_table)}
            (collection_index,{self._quote(column)})
            VALUES (?,?)
            ON CONFLICT(collection_index) DO UPDATE SET
                {self._quote(column)} = excluded.{self._quote(column)}
        """
        self.connection.executemany(query, zip(collection_index, value))
        self.connection.commit()

    def fetch_waveform(self, waveform: str, row: int) -> np.ndarray:
        """Fetch a waveform from the acoustics table.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        row : int
            The row index of the acquisition.

        Returns
        -------
        np.ndarray
            The waveform as a NumPy array.
        """
        
        self._check_column('acoustics', waveform)
        return self.deserialize_array(self.fetch_value(waveform, row))

    def fetch_waveform_batch(self, waveform: str, rows: Any) -> list[np.ndarray]:
        """Fetch a batch of waveforms from the acoustics table.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        rows : Any
            An iterable of row indices.

        Returns
        -------
        list[np.ndarray]
            A list of waveforms as NumPy arrays.
        """
        return [self.fetch_waveform(waveform, int(row)) for row in rows]

    def get_waveform_columns(self) -> list[str]:
        """Get the list of waveform columns in the acoustics table.

        Returns
        -------
        list[str]
            A list of waveform column names.
        """
        excluded = {"collection_index", "X", "Z", "time", "time_collected", "frequency"}
        return [c for c in self.get_columns(self.acoustics_table) if c.startswith("voltage") and c not in excluded]

    def fetch_time(self, row: int = 0) -> np.ndarray:
        """Fetch the time array from the acoustics table.

        Parameters
        ----------
        row : int, optional
            The row index of the acquisition, by default 0.

        Returns
        -------
        np.ndarray
            The time array as a NumPy array.
        """
        return self.deserialize_array(self.fetch_value("time", row))

    def get_acquisition_count(self) -> int:
        """Get the total number of acquisitions in the acoustics table.

        Returns
        -------
        int
            The total number of acquisitions.
        """
        
        query = f"SELECT COUNT(*) FROM {self._quote(self.acoustics_table)}"
        return int(self.connection.execute(query).fetchone()[0])

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    @staticmethod
    def serialize_array(array: np.ndarray) -> bytes:
        """Serialize a NumPy array to bytes for storage in the SQLite database.

        Parameters
        ----------
        array : np.ndarray
            The NumPy array to serialize.

        Returns
        -------
        bytes
            The serialized byte representation of the array.
        """
            
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def deserialize_array(blob: Any) -> np.ndarray:
        """Deserialize bytes from the SQLite database back into a NumPy array.

        Parameters
        ----------
        blob : Any
            The byte representation of the array.

        Returns
        -------
        np.ndarray
            The deserialized NumPy array.
        """
            
        if isinstance(blob, np.ndarray):
            return blob
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        if isinstance(blob, bytearray):
            blob = bytes(blob)
        if not isinstance(blob, bytes):
            return np.asarray(blob)

        try:
            return np.load(io.BytesIO(blob), allow_pickle=False)
        except (ValueError, EOFError):
            return np.frombuffer(blob, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _delete_column(self, column: str, table: str) -> None:
        """Delete a column from a table in the SQLite database.

        Parameters
        ----------
        column : str
            The name of the column to delete.
        table : str
            The name of the table from which to delete the column.
        """
            
        if column in self.get_columns(table):
            self.connection.execute(f"ALTER TABLE {self._quote(table)} DROP COLUMN {self._quote(column)}")

    def _add_column(self, column: str, table: str, type: str='REAL') -> None:
        """Add a new column to the parameters table if it does not already exist.

        Parameters
        ----------
        column : str
            The name of the column to add.
        """
            
        if column not in self.get_columns(table):
            self.connection.execute(f"ALTER TABLE {self._quote(table)} ADD COLUMN {self._quote(column)} {type}")

    def _write_parameter(self, column: str, value: Any) -> None:
        """Write a parameter value to the parameters table.

        Parameters
        ----------
        column : str
            The name of the parameter column.
        value : Any
            The value to write to the parameter column.
        """
            
        self.connection.execute(f"UPDATE {self._quote(self.parameters_table)} SET {self._quote(column)}=?", (value,))
        self.connection.commit()

    def _check_column(self, table: str, column: str) -> None:
        """Check if a column exists in a table.

        Parameters
        ----------
        table : str
            The name of the table.
        column : str
            The name of the column.

        Raises
        ------
        ValueError
            If the column does not exist in the table.
        """
            
        if column not in self.get_columns(table): raise ValueError(f"Column '{column}' does not exist in '{table}'")

    @staticmethod
    def _quote(identifier: str) -> str:
        """Replace quotes with double quotes in an identifier (table or column name) for use in SQL queries.

        Parameters
        ----------
        identifier : str
            The identifier to quote.

        Returns
        -------
        str
            The quoted identifier.
        """
            
        return '"' + identifier.replace('"', '""') + '"'


# TODO: AcousticScanDatabase base class. This does not have real functionality
class AcousticDatabase:
    def __init__(self, save_folder: str, h5_name: str, sqlite_path: str | None = None):
        self.save_folder = save_folder
        self.h5_name = f'{self.save_folder}/{h5_name}'
        self.sqlite_database = AcousticSQlite(sqlite_path) if sqlite_path else None
        self.shape = None
        self.len = None
        self.parameters = {}
        
        try: 
            self.load_parameters()
        except KeyError as e: 
            print(e)
            print('Run parameter setup functions')
            pass
    
    __len__ = lambda self: self.len if self.len else len(self.sqlite_database) if self.sqlite_database else 0
    
    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
        
    def open_h5(self, mode='a'): return h5py.File(self.h5_name, mode)
    
    def get_check_group(self, h5, group_name: str):
        '''Check if the match_first_break group exists and return the waveforms and hilbert data.

        Parameters
        ----------
        h5 : h5py.File
            The open h5py file.

        Returns
        -------
        tuple
            A tuple containing the waveforms and hilbert data.
        '''
        try: return h5[group_name]['waveforms'], h5[group_name]['hilbert']
        except KeyError as e: print(e, f'Run setup for {group_name} first')

    def _print_schema(self, h5, tab: str = ''):
        for name, obj in h5.items():
            print(f"{tab}|--{obj}")
            if isinstance(obj, h5py.Group):
                self._print_schema(obj, tab + '|  ')
                
    def print_schema(self): 
        with self.open_h5() as h5:
            print(f"{h5}")
            self._print_schema(h5)
    
    def print_sqlite_schema(self): self.sqlite_database.print_schema()
    
    def sqlite_to_h5(self): pass # specific to the child class

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
        
    def print_parameters(self):
        with np.printoptions(threshold=10, edgeitems=2,):
            # Adjust 'indent' to change tab sizing if needed
            pprint.pprint(self.parameters, indent=4, compact=True)
            
    def get_sqlite_parameters(self): # TODO: finish filling these out. Make template a part of docs page. 
        # with units and descriptions
        parameters = {
            'transducerFrequency': {
                'value': self.sqlite_database.parameters['transducerFrequency'], 
                'unit': 'MHz', 'description': 'Transducer center frequency'}, 
            'measureTime': {
                'value': self.sqlite_database.parameters['measureTime'], 
                'unit': 'us', 'description': 'Measurement time'},
            'measureDelay': {
                'value': self.sqlite_database.parameters['measureDelay'], 
                'unit': 'us', 'description': 'Measurement delay'},
            'voltageRange': {
                'value': self.sqlite_database.parameters['voltageRange'], 
                'unit': 'V', 
                'description': 'Voltage range'},
            'dt': {
                'value': self.sqlite_database.parameters['dt'], 
                'unit': 's', 
                'description': 'Sampling interval'},
            'f_s': {
                'value': self.sqlite_database.parameters['f_s'], 
                'unit': 'Hz', 
                'description': 'Sampling frequency'},
            'experiment': {
                'value': self.sqlite_database.parameters['experiment'],
                'description': ''''move': move the transducers/ 'single pulse': perform a single test pulse/ 'repeat pulse': repeat a pulse at a single location for a given time and frequency/ 'single scan': perform a single 2D scan/ 'multi scan': repeat a 2D scan with a set frequency/ 'sweep': repeat pulse changing through parameters given in sweepParameters'''},
            'pulserPort': {
                'value': self.sqlite_database.parameters['pulserPort'],
                'description': 'COM port for the pulser'},
            'collectionMode': {
                'value': self.sqlite_database.parameters['collectionMode'],
                'description': 'transmission/echo/both'},
            'pulserType': {
                'value': self.sqlite_database.parameters['pulserType'],
                'description': ''' 'standard': single wave CompactPulser. 'tone burst': USBUT350 tone burst pulser'''},
            'multiplexer': {
                'value': self.sqlite_database.parameters['multiplexer'],
                'description': 'True/False: whether a multiplexer is used'},
            'collectionDirection': {
                'value': self.sqlite_database.parameters['collectionDirection'],
                'description': 'forward/reverse/both'},
            'autoRange': {
                'value': self.sqlite_database.parameters['autoRange'],
                'description': ''},
            'autoRangeEcho': {
                'value': self.sqlite_database.parameters['autoRangeEcho'],
                'description': ''},
            'waves': {
                'value': self.sqlite_database.parameters['waves'],
                'description': ''},
            'samples': {
                'value': self.sqlite_database.parameters['samples'],
                'description': ''},
            'picoModule': {
                'value': self.sqlite_database.parameters['picoModule'],
                'description': ''},
            'pulseModule': {
                'value': self.sqlite_database.parameters['pulseModule'],
                'description': ''},
        }
        
        # scan specific parameters
        if parameters['experiment']['value'] == 'single scan' or parameters['experiment']['value'] == 'multi scan':
            for k in  ['primaryAxis','secondaryAxis']: 
                parameters[k] = {
                    'value': self.sqlite_database.parameters[k]}
            for k in ['primaryAxisRange', 'primaryAxisStep', 'secondaryAxisRange', 'secondaryAxisStep']: 
                parameters[k] = {
                    'value': self.sqlite_database.parameters[k], 
                    'unit': 'mm'}
        if parameters['experiment']['value'] == 'multi scan': pass # experimentBaseName, 
        if parameters['experiment']['value'] == 'repeat pulse': pass #     'pulseInterval' : Minimum time between pulse collection, in seconds, 'experimentTime' : 5,                           # For repeat pulse: Time to collect data, in seconds

        if parameters['collectionMode']['value'] == 'transmission': pass # parameters['voltage_offset'] = {'value': self.sqlite_database.parameters['voltage_offset'], 'unit': 'V'}
        if parameters['collectionMode']['value'] == 'echo': pass # parameters['voltage_offset'] = {'value': self.sqlite_database.parameters['voltage_offset'], 'unit': 'V'}
        if parameters['collectionMode']['value'] == 'both':     
            parameters['voltage_gain'] = {
                'value':{
                    'forward':self.sqlite_database.parameters['gainForward'], 
                    'reverse': self.sqlite_database.parameters['gainReverse']} }
            
        if parameters['pulserType']['value'] == 'tone burst': 
            parameters['halfCycles'] = self.sqlite_database.parameters['halfCycles']

        if parameters['multiplexer']['value']:
            for k in  ['multiplexerPort','rfSwitch', 't0PulseSwitch', 't0ReceiveSwitch', 't1PulseSwitch', 't1ReceiveSwitch']:
                parameters[k] = self.sqlite_database.parameters[k]
        
        ####################################################
        # To format in the parameters table by index
        ####################################################
        if parameters['autoRange']['value']: pass
        if parameters['autoRangeEcho']['value']: pass
        
        ####################################################
        # for plotting
        ####################################################        
        parameters['time_array'] = self.sqlite_database.fetch_time()
        parameters['waveform_labels'] = self.sqlite_database.waveform_columns
        
        return parameters
                  
    def save_parameters(self):
        '''Save the parameters dictionary to the HDF5 file as a JSON string. 
        Create dataset config_metadata in the root of the HDF5 file if it does not exist.'''
        with self.open_h5() as h5:
            json_str = json.dumps(self.parameters, cls=NumpyEncoder) # Convert dictionary to a JSON string
            dt = h5py.special_dtype(vlen=str) # save as special h5 type
            try: del h5['config_metadata']  
            except: pass
            h5.require_dataset('config_metadata', (), data=json_str, dtype=dt)

    def load_parameters(self):
        print('Loading h5 parameters...')
        with self.open_h5() as h5:
            loaded_json = h5['config_metadata'][()]
            self.parameters = json.loads(loaded_json)
            self.parameters['time_array'] = np.array(self.parameters['time_array'])
            self.shape = tuple(self.parameters['shape'])
            self.len = self.parameters['len']
        print('\tFinished.')

    def setup_parameters(self): pass # specific to the child class. must also set time_array, shape, and len
    
    # -------------------------------------------------------------------------
    # Raw data
    # -------------------------------------------------------------------------
        
    def setup_raw_dataset(self):
        '''Write raw data and parameters to the HDF5 file.
        
        Parameters
        ----------
        data : np.ndarray
            The raw data to write.
        time_array : float
            The time array.
        parameters : dict
            The parameters to write.
        '''

        with self.open_h5() as h5:
            grp = h5.require_group('raw_data')
            waveforms_dset = grp.require_dataset('waveforms', shape=self.shape, dtype='f4', track_order=True)
            grp.require_dataset('hilbert', shape=self.shape, dtype='f4', track_order=True)
            waveforms_dset.attrs['time_array'] = self.parameters['time_array']
            
    def fill_raw_dataset(self, idx_data : Iterator[tuple[tuple, np.ndarray]]):
        '''Fill a series of specific locations in the raw data dataset.
        
        Parameters
        ----------
        idx_data : Iterator[tuple[tuple, np.ndarray]]
            An iterator of tuples, where each tuple contains an index (tuple) and the corresponding data (np.ndarray) to fill at that index.
            The index should fill axis until the last dimension, and the data should be a 1D array of shape (time_length,).
        '''
        with self.open_h5() as h5: # TODO: calculate the hilbert window and maxes. Look at preprocessed.
            try: dset = h5['raw_data']['waveforms']
            except KeyError as e: print(e, 'Run setup_raw_dataset first'); return
            
            for idx, data in tqdm(idx_data, desc="Filling raw data", total=math.prod(self.shape[:-1])):
                dset[idx] = data

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------
    
    def write_indexed_parameter(self, keys, values, shapes): pass # specific to the child class
       
    def load_shape(self): pass # specific to the child class
    
    def idx_sqlite_data_generator(self): pass # specific to the child class
    
    # fill _ with some combination of whatever axis are in the dataset (w_x_z, x_z, w_y, etc.)
    def _generator(self): pass # specific to the child class

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _absolute_max(waveform): return float(np.max(np.abs(waveform)))

    @staticmethod
    def _hilbert_window(waveform): return np.abs(hilbert(waveform))

    @staticmethod
    def _undo_gain(waveform, gain, offset): return (waveform - offset) / gain
    
    def _build_filter_sos(self, lower_fs_coeff: float, upper_fs_coeff: float, filter_order: int) -> Any:
        """Build Butterworth bandpass filter coefficients.

        Parameters
        ----------
        lower_fs_coeff : float
            Highpass filter lower limit of C*f_s.
        upper_fs_coeff : float
            Lowpass filter upper limit of C*f_s.
        filter_order : int
            The order of the filter.

        Returns
        -------
        Any
            The second-order sections of the filter.
        """
        if self.parameters['f_s']['value'] is None: raise ValueError("Sampling frequency is unavailable")
        f_s = self.parameters['f_s']['value']
        return butter(filter_order, [f_s * lower_fs_coeff, f_s * upper_fs_coeff], btype="bandpass", fs=f_s, output="sos")

    def setup_preprocessed_data(self, apply_ungain: bool = False, apply_filter: bool = False, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, filter_order: int = 3):
        '''Setup preprocessed group, dataset, and attributes
        
        Parameters
        ----------
        apply_ungain : bool
            Whether to apply ungain to the data.
        apply_filter : bool
            Whether to apply a bandpass filter to the data.
        lower_fs_coeff : float
            Highpass filter lower limit of C*f_s.
        upper_fs_coeff : float
            Lowpass filter upper limit of C*f_s.
        filter_order : int
            The order of the filter.
        '''
        metadata = {}
        with self.open_h5() as h5:
            grp = h5.require_group('preprocessed_data')
            dset = grp.require_dataset('waveforms', shape=self.shape, dtype='f4', track_order=True)
            grp.require_dataset('hilbert', shape=self.shape, dtype='f4', track_order=True)
            
            if apply_ungain: pass # TODO: later, get gain columns and indices/values from paramters collection mode
            if apply_filter: 
                self.sos = self._build_filter_sos(lower_fs_coeff, upper_fs_coeff, filter_order) if apply_filter else None
                metadata.update({"lower_fs_coeff": lower_fs_coeff, "upper_fs_coeff": upper_fs_coeff, "filter_order": filter_order})
            
            dset.attrs.update(metadata)
            dset.attrs['time_array'] = self.parameters['time_array']
    
    def fill_preprocessed_data(self, idxs: Iterator[tuple]) -> None:   
        '''Fill a series of specific locations in the raw data dataset.
                
        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples, where each tuple contains an index.
            The index should fill axis until the last dimension.
        '''
        with self.open_h5() as h5:
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'preprocessed_data')
            
            for idx in tqdm(idxs, desc="Writing preprocessed data", total=math.prod(self.shape[:-1])):
                data = h5['raw_data']['waveforms'][*idx]
                # if apply_ungain: waveform = self._undo_gain(waveform, gain, offset)
                try: data = sosfiltfilt(self.sos, data)
                except: continue

                waveform_dset[*idx] = data
                hilbert_dset[*idx] = self._hilbert_window(data)

    # -------------------------------------------------------------------------
    # alignment
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_rollback(waveform):
        '''
        Calculate the average rollback value fo each of the mean waveforms
        
        Parameters:
        -----------
        waveform: np.ndarray
            The waveform data.
            
        Returns:
        --------
        np.ndarray
            The mean rollback values for each waveform profile.
        '''
        mean_profile = waveform.mean(axis=tuple(i for i in range(1,len(waveform.shape)-1)))
        peak_distances = np.array([ find_peaks(profile, height=0.25*profile.max())[0][0] for profile in mean_profile ])
        rollback = peak_distances-min(peak_distances)
        return rollback
    
    def setup_first_break_aligned_data(self, from_group='preprocessed_data'):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        '''
        if len(self.parameters['waveform_labels'])<2: print('Only one waveform. No need to align peaks'); return
        if self.parameters['collectionDirection']['value'] != 'both': print('Collection mode is one direction. No need to align peaks'); return
        
        with self.open_h5() as h5:
            self.get_check_group(h5, from_group)
    
            rollbacks = self._calculate_rollback(h5[from_group]['hilbert'][:])
            max_ = rollbacks.max()
            grp = h5.require_group('first_break_aligned')
            
            new_shape = self.shape[:-1] + (self.shape[-1]-max_,)
            dset = grp.require_dataset('waveforms', shape=new_shape, dtype='f4', track_order=True)
            grp.require_dataset('hilbert', shape=new_shape, dtype='f4', track_order=True)
            
            dset.attrs['from_group'] = str(h5[from_group])
            dset.attrs['rollbacks'] = rollbacks
            dset.attrs['time_array'] = self.parameters['time_array'][:-max_]
  
    def fill_first_break_aligned_data(self, idxs: Iterator[tuple], from_group='preprocessed_data') -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        """          
        with self.open_h5() as h5:
            waveform_dset_unrolled, hilbert_dset_unrolled = self.get_check_group(h5, from_group)
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'first_break_aligned')
            new_shape = hilbert_dset.shape
            rollbacks = waveform_dset.attrs['rollbacks']

            for idx in tqdm(idxs, desc="Aligning first break", total=math.prod(self.shape[1:-1])):
                waveform_dset[*idx] = waveform_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]
                hilbert_dset[*idx] = hilbert_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]
    
    # -------------------------------------------------------------------------
    # Basic viz
    # -------------------------------------------------------------------------  
                  
    def view_mean_raw(self, idx):
        fig, ax = plt.subplots()
        with self.open_h5() as h5:
            data = h5['raw_data']['waveforms']
            ax.plot(data.attrs['time_array'], data[idx])
        fig.show()
  

# Naming convention: AcousticDatabase__ is a subclass of AcousticDatabase that handles __ analysis.   
class AcousticDatabaseFFT(AcousticDatabase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # -------------------------------------------------------------------------
    # alignment
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_rollback(waveform):
        '''
        Calculate the average rollback value fo each of the mean waveforms
        
        Parameters:
        -----------
        waveform: np.ndarray
            The waveform data.
            
        Returns:
        --------
        np.ndarray
            The mean rollback values for each waveform profile.
        '''
        mean_profile = waveform.mean(axis=tuple(i for i in range(1,len(waveform.shape)-1)))
        peak_distances = np.array([ find_peaks(profile, height=0.25*profile.max())[0][0] for profile in mean_profile ])
        rollback = peak_distances-min(peak_distances)
        return rollback
    
    def setup_first_break_aligned_data(self, from_group='preprocessed_data'):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        '''
        if len(self.parameters['waveform_labels'])<2: print('Only one waveform. No need to align peaks'); return
        if self.parameters['collectionDirection']['value'] != 'both': print('Collection mode is one direction. No need to align peaks'); return
        
        with self.open_h5() as h5:
            self.get_check_group(h5, from_group)
    
            rollbacks = self._calculate_rollback(h5[from_group]['hilbert'][:])
            max_ = rollbacks.max()
            grp = h5.require_group('first_break_aligned')
            
            new_shape = self.shape[:-1] + (self.shape[-1]-max_,)
            dset = grp.require_dataset('waveforms', shape=new_shape, dtype='f4', track_order=True)
            grp.require_dataset('hilbert', shape=new_shape, dtype='f4', track_order=True)
            
            dset.attrs['from_group'] = str(h5[from_group])
            dset.attrs['rollbacks'] = rollbacks
            dset.attrs['time_array'] = self.parameters['time_array'][:-max_]
  
    def fill_first_break_aligned_data(self, idxs: Iterator[tuple], from_group='preprocessed_data') -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        """          
        with self.open_h5() as h5:
            waveform_dset_unrolled, hilbert_dset_unrolled = self.get_check_group(h5, from_group)
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'first_break_aligned')
            new_shape = hilbert_dset.shape
            rollbacks = waveform_dset.attrs['rollbacks']

            for idx in tqdm(idxs, desc="Aligning first break", total=math.prod(self.shape[1:-1])):
                waveform_dset[*idx] = waveform_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]
                hilbert_dset[*idx] = hilbert_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]

    


# Naming convention: Acoustic__Database is a subclass of AcousticDatabase that handles __ data.
class AcousticScanDatabase(AcousticDatabase):
    """Acoustics database subclass for 2D scan experiments (collection_index <-> X,Z coordinates).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # -------------------------------------------------------------------------
    # setup and indexing functions
    # -------------------------------------------------------------------------
    
    def setup_parameters(self):
        """Setup the parameters for the acoustic data."""
        self.parameters = self.get_sqlite_parameters()
        self.parameters['shape'] = (
            len(self.parameters["waveform_labels"]),
            math.ceil(self.parameters["primaryAxisRange"]['value'] / abs(self.parameters["primaryAxisStep"]['value'])) + 1,
            math.ceil(self.parameters["secondaryAxisRange"]['value'] / abs(self.parameters["secondaryAxisStep"]['value'])) + 1,
            len(self.parameters["time_array"]))
        self.parameters['len'] = self.parameters['shape'][1]*self.parameters['shape'][2]
        
        self.shape = self.parameters['shape']
        self.len = self.parameters['len']
        
        self.save_parameters()

    def sqlite_to_h5(self): # TODO: test
        """Convert the SQLite database to an HDF5 file."""
        self.setup_parameters()
        
        keys = ['collection_idx', 'time_collected']
        values = (self.sqlite_database.fetch_column(column) for column in ('collection_index', 'time_collected'))
        shapes = [self.shape[1:3], self.shape[1:3]]
        # TODO: add cases for if there is offset, gain and autorange
        
        self.setup_raw_dataset()
        self.fill_raw_dataset(self.idx_sqlite_data_generator())
        
        self.write_indexed_parameter(keys, values, shapes)
        
        print(f'\tFinished')

    def idx_sqlite_data_generator(self):
        for w,waveform in enumerate(self.parameters['waveform_labels']):
            for x in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (w,x,z), self.sqlite_database.fetch_waveform(waveform, x+z*self.shape[1])

    def w_x_z_generator(self):
        for w in range(len(self.parameters['waveform_labels'])):
            for x in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (w,x,z)

    def x_z_generator(self):
        for x in range(self.shape[1]):
            for z in range(self.shape[2]):
                yield (x,z)

    # -------------------------------------------------------------------------
    # preprocessing
    # -------------------------------------------------------------------------
    
    def write_preprocessed_data(self, **kwargs): # TODO: implement shift/crop waveform based on first arrival time
        '''Write preprocessed data to the HDF5 file.

        Parameters
        ----------
        **kwargs
            Parameters passed to :meth:`setup_preprocessed_data`. See that
            method for the available preprocessing options.
        '''
        self.setup_preprocessed_data(**kwargs)
        self.fill_preprocessed_data(self.w_x_z_generator() )
        print('\tFinished')
    
    # -------------------------------------------------------------------------
    # alignment
    # -------------------------------------------------------------------------
        
    def write_first_break_aligned_data(self, **kwargs): 
        """Align the first break of the waveforms in the dataset and write to the HDF5 file.

        Parameters
        ----------
        **kwargs
            Parameters passed to :meth:`setup_first_break`. See that
            method for the available alignment options.
        """
        self.setup_first_break_aligned_data(**kwargs)
        self.fill_first_break_aligned_data(self.w_x_z_generator(), **kwargs)
        print('\tFinished')
    
    # -------------------------------------------------------------------------
    # basic viz
    # -------------------------------------------------------------------------

    def view_mean_img_plot(self, from_group = 'preprocessed_data'): #TODO: check
        fig, ax = plt.subplots(1+self.shape[0], 1)
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['waveforms']
            for i, dat in enumerate(data):
                a = ax[i].imshow(abs(dat).max(axis=(2)).T,)
                plt.colorbar(a, ax=ax[i], label='Ampl. (mV)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[-1].plot(data.attrs['time_array'], dat.mean(axis=(0,1)), label=self.parameters['waveform_labels'][i])
            ax[-1].legend()
        fig.suptitle('pixel wise abs. max. of image and spectrum of processed data')
        fig.tight_layout()
        fig.show()


      
# TODO
class AcousticsSweepDatabase(AcousticDatabase):
    """Acoustics database subclass for sweep experiments (collection_index <-> single swept parameter)."""

    # -------------------------------------------------------------------------
    # collection indexing
    # -------------------------------------------------------------------------

    def create_collection_index_mapping(self) -> None:
        """Map collection_index to the swept parameter's values.
        """
        if self.parameters.get("experiment") != "sweep":
            raise ValueError("Parameters do not describe a sweep experiment")

        sweep_params = ast.literal_eval(self.parameters["sweepParams"])
        idx = 0
        for key, vals in sweep_params.items():
            mapping = {}
            self._add_column(key, self.acoustics_table, type=self.types_dict.get(type(vals[0]).__name__, "REAL"))
            for val in vals:
                mapping[idx] = val
                idx += 1
            self.store_acoustics_value(collection_index=list(mapping.keys()), column=key, value=list(mapping.values()))

    def get_sweep_column(self) -> str:
        """Get the name of the swept parameter column.

        Returns
        -------
        str
            The name of the swept parameter.
        """
        sweep_params = ast.literal_eval(self.parameters["sweepParams"])
        return next(iter(sweep_params))

    def get_sweep_values(self) -> list[Any]:
        """Get the swept parameter's values, ordered by collection_index.

        Returns
        -------
        list[Any]
            The swept parameter values.
        """
        column = self.get_sweep_column()
        query = f"SELECT {self._quote(column)} FROM {self._quote(self.acoustics_table)} ORDER BY collection_index"
        return [r[0] for r in self.connection.execute(query)]

    def fetch_collection_index_by_value(self, value: Any) -> int:
        """Fetch the collection_index matching a swept parameter value.

        Parameters
        ----------
        value : Any
            The swept parameter value to look up.

        Returns
        -------
        int
            The matching collection_index.
        """
        column = self.get_sweep_column()
        query = f"SELECT collection_index FROM {self._quote(self.acoustics_table)} WHERE {self._quote(column)}=?"
        row = self.connection.execute(query, (value,)).fetchone()
        if row is None:
            raise ValueError(f"No collection_index found for {column}={value}")
        return int(row[0])


# TODO: build out and integrate with visualization classes    
class AcousticScanFrequencyDomainDatabase(AcousticDatabase):
    """A class for handling frequency domain data in an acoustics database."""
     
class AcousticDatabaseCWT(AcousticDatabaseFrequencyDomain):
    """A class for handling continuous wavelet transform data in an acoustics database.

    This class extends the AcousticsDatabaseFrequencyDomain class to provide additional functionality
    for working with continuous wavelet transform data.
    """

class AcousticDatabaseTMM(AcousticDatabaseFrequencyDomain):
    """A class for handling time-frequency map data in an acoustics database.

    This class extends the AcousticsDatabase class to provide additional functionality
    for working with time-frequency map data.
    """
