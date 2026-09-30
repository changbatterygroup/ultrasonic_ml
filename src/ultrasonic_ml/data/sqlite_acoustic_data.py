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
from scipy.signal import butter, sosfiltfilt, hilbert
from tqdm import tqdm

from ..utils import profile 
import time


# Done
class AcousticsDatabase: 
    #TODO: future consideration, deserializing blob is bottleneck. 
    # consider hdf5 with sqlite as metadata.
    # TODO: calculate maxes using hilbert window instead of our minmax
    """Base class for SQLite acoustics data and analysis."""

    def __init__(self, database_path: str | Path) -> None:
        """Initialize the AcousticsDatabase.

        Parameters
        ----------
        database_path : str | Path
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
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.acoustics_table = "acoustics"
        self.parameters_table = "parameters"
        self.analysis_table = "analysis"
        self.reference_table = "analysis_reference"
        self.index_column = "collection_index"
        self.parameters = {}
        self.waveform_columns = []
        self.dt = None
        self.f_s = None
        self.f = None
        self.omega_scaled = None
        self.absolute_max = {}
        self.absolute_max_overall = None
        self._cache_enabled = True
        self._cache: dict[tuple[Any, ...], Any] = {}

        self.connect()
        print('setting paramters')
        self.waveform_columns = self.get_waveform_columns()
        self.parameters = self.get_parameters()
        self.initialize_frequency_parameters()
        print('creating analysis and reference tables if they don\'t exist')
        self.create_analysis_table()
        self.create_reference_table()
        print('computing maxes')
        self.raw_maxs_ = {}
        self.analysis_maxs_ = {}
        self._compute_raw_maxs()
        # self._compute_analysis_maxs()
        # self.calculate_hilbert_raw()
        # TODO: add methods for deleting columns, tables, and reindexing.

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def __enter__(self) -> AcousticsDatabase:
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
        self.connection = sqlite3.connect(self.database_path, detect_types=sqlite3.PARSE_DECLTYPES)
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

    def get_parameters(self) -> dict[str, Any]:
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

    def get_datetime(self, row: int = 0) -> datetime:
        """Get the datetime of a specific acquisition.

        Parameters
        ----------
        row : int, optional
            The row index of the acquisition, by default 0.

        Returns
        -------
        datetime
            The datetime of the acquisition.
        """
        value = self.fetch_value("time_collected", row)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if isinstance(value, (int, float, np.number)):
            return datetime.fromtimestamp(float(value))
        raise TypeError(f"Unsupported time_collected type: {type(value)}")

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

    def get_waveform_columns(self) -> list[str]:
        """Get the list of waveform columns in the acoustics table.

        Returns
        -------
        list[str]
            A list of waveform column names.
        """
        excluded = {"collection_index", "X", "Z", "time", "time_collected", "frequency"}
        return [c for c in self.get_columns(self.acoustics_table) if c.startswith("voltage") and c not in excluded]

    def has_waveform(self, waveform: str) -> bool:
        """Check if a waveform column exists in the acoustics table.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.

        Returns
        -------
        bool
            True if the waveform column exists, False otherwise.
        """
        return waveform in self.waveform_columns

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
        
        if not self.has_waveform(waveform):
            raise ValueError(f"Unknown waveform: {waveform}")
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

    def fetch_waveform_value(self, waveform: str, index: int, row: int) -> float:
        """Fetch a specific value from a waveform at a given index and row.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        index : int
            The index of the value in the waveform array.
        row : int
            The row index of the acquisition.

        Returns
        -------
        float
            The value from the waveform at the specified index and row.
        """
        return float(self.fetch_waveform(waveform, row)[index])

    def fetch_index_across_acquisitions(self, waveform: str, index: int) -> np.ndarray:
        """Fetch a specific index value from a waveform across all acquisitions.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        index : int
            The index of the value in the waveform array.

        Returns
        -------
        np.ndarray
            A NumPy array of values from the waveform at the specified index across all acquisitions.
        """
        query = f"SELECT {self._quote(waveform)} FROM {self._quote(self.acoustics_table)}"
        return np.asarray([self.deserialize_array(r[0])[index] for r in self.connection.execute(query)])

    def get_acquisition_count(self) -> int:
        """Get the total number of acquisitions in the acoustics table.

        Returns
        -------
        int
            The total number of acquisitions.
        """
        
        query = f"SELECT COUNT(*) FROM {self._quote(self.acoustics_table)}"
        return int(self.connection.execute(query).fetchone()[0])

    def get_acquisition_index(self, row: int) -> int:
        """Get the collection index of a specific acquisition.

        Parameters
        ----------
        row : int
            The row index of the acquisition.

        Returns
        -------
        int
            The collection index of the acquisition.
        """
        return int(self.fetch_value(self.index_column, row))
    
    def _compute_raw_maxs(self) -> None:
        for waveform in self.waveform_columns:
            column = f"max_{waveform}"
            if column in self.get_columns(self.acoustics_table):
                print(f"Skipping {column} as it already exists.")
                continue
            
            print(f"Computing maxs for {waveform}...")
            self._add_column(column, self.acoustics_table)

            query = f"SELECT collection_index,{self._quote(waveform)} FROM {self._quote(self.acoustics_table)}"
            rows = self.connection.execute(query).fetchall()

            update_query = f"""
                UPDATE {self._quote(self.acoustics_table)}
                SET {self._quote(column)}=?
                WHERE collection_index=?
            """
            params = [(self._absolute_max(self.deserialize_array(blob)), collection_index) for collection_index, blob in rows]
            self.connection.executemany(update_query, params)

        self.connection.commit()
        self.set_global_raw_maxes()
    
    @profile
    def set_global_raw_maxes(self) -> None: # test
        print('Setting global raw maxes...')
        for waveform in self.waveform_columns:        
            self.raw_maxs_[waveform] = self.connection.execute(f"""
                SELECT MAX(max_{waveform}) FROM {self._quote(self.acoustics_table)}
                """).fetchone()[0]
            
    # -------------------------------------------------------------------------
    # Preprocessing functions
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _absolute_max(waveform: np.ndarray) -> float:
        """Calculate the absolute maximum of a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            The waveform data.

        Returns
        -------
        float
            The absolute maximum of the waveform.
        """
        return float(np.max(np.abs(waveform)))

    @staticmethod
    def _hilbert_window(waveform: np.ndarray) -> float:
        """Calculate the Hilbert window of a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            The waveform data.

        Returns
        -------
        np.float64
            The Hilbert window of the waveform.
        """
        hilbert_transform = hilbert(waveform)
        return float(np.max(np.abs(hilbert_transform)))
    
    def _undo_gain(self, waveform: np.ndarray, gain: float, offset: float = 0.0) -> np.ndarray:
        """Undo the gain and offset applied to a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            The waveform data.
        gain : float
            The gain value to undo.
        offset : float, optional
            The offset value to undo, by default 0.0.

        Returns
        -------
        np.ndarray
            The waveform with the gain and offset undone.
        """
        return (waveform - offset) / gain

    def _butterworth_filter(self, waveform: np.ndarray, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, order: int = 3, sos: Any | None = None) -> np.ndarray:
        """Apply a Butterworth bandpass filter to the waveform.

        Parameters
        ----------
        waveform : np.ndarray
            The waveform data.
        lower_fs_coeff : float, optional
            The lower frequency coefficient, by default 1/250.
        upper_fs_coeff : float, optional
            The upper frequency coefficient, by default 1/5.
        order : int, optional
            The order of the filter, by default 3.
        sos : Any | None, optional
            The second-order sections of the filter, by default None.

        Returns
        -------
        np.ndarray
            The filtered waveform.
        """
        if self.f_s is None:
            raise ValueError("Sampling frequency is unavailable")

        if sos is None:
            sos = butter(order, [self.f_s * lower_fs_coeff, self.f_s * upper_fs_coeff], btype="bandpass", fs=self.f_s, output="sos")

        return sosfiltfilt(sos, waveform)
    
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
        if self.f_s is None:
            raise ValueError("Sampling frequency is unavailable")
        return butter(filter_order, [self.f_s * lower_fs_coeff, self.f_s * upper_fs_coeff], btype="bandpass", fs=self.f_s, output="sos")

    def _apply_preprocessing_pipeline(self, waveform: np.ndarray, gain: float | None, offset: float, sos: Any | None, apply_ungain: bool, apply_filter: bool, lower_fs_coeff: float, upper_fs_coeff: float, filter_order: int) -> np.ndarray:
        """Apply ungain and/or filtering to a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            The waveform data.
        gain : float | None
            The gain value to undo.
        offset : float
            The offset value to undo.
        sos : Any | None
            Precomputed filter coefficients, if any.
        apply_ungain : bool
            Whether to remove gain.
        apply_filter : bool
            Whether to apply a filter.
        lower_fs_coeff : float
            Highpass filter lower limit of C*f_s.
        upper_fs_coeff : float
            Lowpass filter upper limit of C*f_s.
        filter_order : int
            The order of the filter.

        Returns
        -------
        np.ndarray
            The processed waveform.
        """
        if apply_ungain: waveform = self._undo_gain(waveform, gain, offset)
        if apply_filter: waveform = self._butterworth_filter(waveform, lower_fs_coeff, upper_fs_coeff, filter_order, sos)
        return waveform

    def _fetch_preprocess_rows(self, waveform: str, gain_column: str | None, offset_column: str | None) -> Iterator[tuple[int, np.ndarray, float | None, float]]:
        """Fetch rows needed for preprocessing a waveform.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.
        gain_column : str | None
            The name of the gain column, if any.
        offset_column : str | None
            The name of the offset column, if any.

        Yields
        ------
        tuple[int, np.ndarray, float | None, float]
            collection_index, waveform_data, gain, offset for each row.
        """
        columns = [self._quote(waveform)]
        if gain_column: columns.append(self._quote(gain_column))
        if offset_column: columns.append(self._quote(offset_column))

        query = f"SELECT collection_index,{','.join(columns)} FROM {self._quote(self.acoustics_table)} ORDER BY collection_index"

        for row in self.connection.execute(query):
            collection_index, blob = row[:2]
            waveform_data = self.deserialize_array(blob).astype(np.float32, copy=False)
            i = 2
            gain = float(row[i]) if gain_column else None
            if gain_column:
                i += 1
            offset = float(row[i]) if offset_column else 0.0
            yield collection_index, waveform_data, gain, offset
  
    def preprocess_waveform(self, waveform: str, apply_ungain: bool = False, apply_filter: bool = False, gain_column: str | None = None, offset_column: str | None = None, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, filter_order: int = 3) -> int:
        if not self.has_waveform(waveform): raise ValueError(f"Unknown waveform: {waveform}")
        if apply_ungain and gain_column is None: raise ValueError("gain_column is required when apply_ungain=True")
        if not apply_ungain: gain_column = None
        if apply_filter and self.f_s is None: raise ValueError("Sampling frequency is unavailable")
        if not apply_filter: lower_fs_coeff = upper_fs_coeff = filter_order = None

        sos = self._build_filter_sos(lower_fs_coeff, upper_fs_coeff, filter_order) if apply_filter else None

        metadata = {
            "apply_ungain": apply_ungain,
            "apply_filter": apply_filter,
            "lower_fs_coeff": lower_fs_coeff,
            "upper_fs_coeff": upper_fs_coeff,
            "filter_order": filter_order,
        }

        reference_id = self.create_reference("preprocessed", "preprocessed", metadata=metadata)

        rows = []
        for collection_index, waveform_data, gain, offset in tqdm(self._fetch_preprocess_rows(waveform, gain_column, offset_column), total=len(self), desc=f"Preprocessing {waveform}"):
            if self._analysis_exists(collection_index, waveform, "preprocessed", "waveform", reference_id):
                continue

            waveform_data = self._apply_preprocessing_pipeline(waveform_data, gain, offset, sos, apply_ungain, apply_filter, lower_fs_coeff, upper_fs_coeff, filter_order)
            rows.append((collection_index, waveform, "preprocessed", "waveform", waveform_data, "time", "ns", "voltage", "mV", reference_id))

        self.store_analysis_results_batch(rows)
        self.calculate_hilbert_analysis(source_analysis_name="preprocessed", source_result_name="waveform", reference_id=reference_id)
        self.connection.commit()
        
        self._compute_analysis_maxs(lazy=True)
        return reference_id
    
    def fetch_preprocessed_waveform(self, waveform: str, row: int, reference_id: int | None = None) -> np.ndarray:
        """Fetch the preprocessed waveform for a specific acquisition.

        Parameters
        ----------
        waveform : str
            The name of the waveform.
        row : int
            The row index of the acquisition.
        reference_id : int | None, optional
            The reference ID, by default None.

        Returns
        -------
        np.ndarray
            The preprocessed waveform.
        """
        collection_index = self.get_acquisition_index(row)
        query = f"SELECT value FROM {self._quote(self.analysis_table)} WHERE collection_index=? AND waveform=? AND analysis_name='preprocessed' AND result_name='waveform'"
        params = [collection_index, waveform]

        if reference_id is not None:
            query += " AND reference_id=?"
            params.append(reference_id)

        query += " ORDER BY rowid DESC LIMIT 1"
        result = self.connection.execute(query, params).fetchone()

        if result is None:
            raise ValueError(f"No preprocessed waveform for {waveform}, row={row}")

        return self.deserialize_array(result[0])

    def calculate_hilbert_raw(self) -> None:
        """Compute the Hilbert window for a waveform's raw data.

        Parameters
        ----------
        waveform : str
            The waveform name.
        """
        for waveform in self.waveform_columns:
            reference_id = self.create_reference("hilbert_window", None, metadata={})

            results = []
            for row in tqdm(range(len(self)), desc=f"Hilbert transform for {waveform}"):
                collection_index = self.get_acquisition_index(row)
                if self._analysis_exists(collection_index, waveform, "hilbert_window", "raw", reference_id): continue
                waveform_ = self.fetch_waveform(waveform, row)
                results.append((collection_index, waveform, 
                                "hilbert_window", "raw",
                                self._hilbert_window(waveform_),
                                "time", "ns", "voltage", "mV", 
                                reference_id))
            self.store_analysis_results_batch(results)
        self.connection.commit()
        
    def calculate_hilbert_analysis(self, source_analysis_name: str = "preprocessed", source_result_name: str = "waveform", reference_id: int | None = 1) -> None:
        """Compute the Hilbert window for a waveform's stored analysis results.

        Parameters
        ----------
        waveform : str
            The waveform name.
        source_analysis_name : str, optional
            The analysis_name to read input data from, by default "preprocessed".
        source_result_name : str, optional
            The result_name to read input data from, by default "waveform".
        reference_id : int | None, optional
            The reference_id to read input data from, by default None.

        Returns
        -------
        int
            The reference ID of the hilbert_window analysis.
        """
        for waveform in self.waveform_columns:
            hilbert_reference_id = self.create_reference("hilbert_window", None, metadata={})
            combined_result_name = f"{source_analysis_name}_{source_result_name}"
            results = []
            analysis_batch = self.fetch_analysis_values_batch(waveform, source_analysis_name, source_result_name, reference_id)
            for collection_index, waveform_ in tqdm(analysis_batch, total=len(self), desc=f"Hilbert transform for {waveform}" ):
                if self._analysis_exists(collection_index, waveform, "hilbert_window", combined_result_name, hilbert_reference_id): continue
                results.append((collection_index, waveform, 
                                "hilbert_window", combined_result_name,
                                self._hilbert_window(waveform_), 
                                "time", "ns", "voltage", "mV", hilbert_reference_id))

            self.store_analysis_results_batch(results)
        self.connection.commit()
            
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    
    def _analysis_exists(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, reference_id: int | None) -> bool:
        """Check if an analysis result already exists.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        waveform : str
            The waveform name.
        analysis_name : str
            The name of the analysis.
        result_name : str
            The name of the result.
        reference_id : int | None
            The reference ID.

        Returns
        -------
        bool
            True if a matching row already exists.
        """
        query = f"""
            SELECT 1 FROM {self._quote(self.analysis_table)}
            WHERE collection_index=? AND waveform=? AND analysis_name=? AND result_name=? AND reference_id IS ?
            LIMIT 1
        """
        return self.connection.execute(query, (collection_index, waveform, analysis_name, result_name, reference_id)).fetchone() is not None

    def create_analysis_table(self) -> None:
        """Create the analysis table in the SQLite database if it does not exist.
        """
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._quote(self.analysis_table)} (
                collection_index INTEGER NOT NULL,
                waveform TEXT NOT NULL,
                analysis_name TEXT NOT NULL,
                result_name TEXT NOT NULL,
                value BLOB,
                x_axis TEXT,
                x_unit TEXT,
                y_axis TEXT,
                y_unit TEXT,
                max_value REAL,
                reference_id INTEGER,
                PRIMARY KEY (collection_index,waveform,analysis_name,result_name,reference_id)
            )
        """)
        self.connection.commit()
    
    def store_analysis_results_batch(self, rows: list[tuple]) -> None:
        """Store multiple analysis results in the analysis table.

        Parameters
        ----------
        rows : list[tuple]
            List of tuples: (collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id).
        """
        params = []
        for collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id in rows:
            if isinstance(value, np.ndarray):
                value = self.serialize_array(value)
            params.append((collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id))

        query = f"""
            INSERT OR REPLACE INTO {self._quote(self.analysis_table)}
            (collection_index,waveform,analysis_name,result_name,value,x_axis,x_unit,y_axis,y_unit,reference_id)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        self.connection.executemany(query, params)

    def store_analysis_result(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, value: Any, x_axis: str | None = None, x_unit: str | None = None, y_axis: str | None = None, y_unit: str | None = None, reference_id: int | None = None) -> None:
        """Store a single analysis result in the analysis table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        waveform : str
            The waveform data.
        analysis_name : str
            The name of the analysis.
        result_name : str
            The name of the result.
        value : Any
            The value of the result.
        x_axis : str | None, optional
            The x-axis label, by default None.
        x_unit : str | None, optional
            The x-axis unit, by default None.
        y_axis : str | None, optional
            The y-axis label, by default None.
        y_unit : str | None, optional
            The y-axis unit, by default None.
        reference_id : int | None, optional
            The reference ID, by default None.
        """
        self.store_analysis_results_batch([(collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id)])
    
    def fetch_analysis_value(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, reference_id: int | None = None) -> Any:
        """Fetch analysis data result from the analysis table.

        Parameters
        ----------
        collection_index : int
            The index of the collection.
        waveform : str
            The waveform data.
        analysis_name : str
            The name of the analysis.
        result_name : str
            The name of the result.
        reference_id : int | None, optional
            The reference ID, by default None.

        Returns
        -------
        Any
            The value of the analysis result.
        """
        query = f"""
            SELECT value FROM {self._quote(self.analysis_table)}
            WHERE collection_index=? AND waveform=? AND analysis_name=? AND result_name=?
        """

        params = [collection_index, waveform, analysis_name, result_name]

        if reference_id is not None:
            query += " AND reference_id=?"
            params.append(reference_id)

        query += " ORDER BY rowid DESC LIMIT 1"
        result = self.connection.execute(query, params).fetchone()

        if result is None:
            raise ValueError("Analysis result not found")

        value = result[0]
        return self.deserialize_array(value) if isinstance(value, (bytes, bytearray, memoryview)) else value

    def fetch_analysis_values_batch(self, waveform: str, analysis_name: str, result_name: str, reference_id: int | None = None) -> Iterator[tuple[int, Any]]:
        """Fetch all analysis values for a waveform/analysis/result across all collection indices.

        Parameters
        ----------
        waveform : str
            The waveform name.
        analysis_name : str
            The name of the analysis.
        result_name : str
            The name of the result.
        reference_id : int | None, optional
            The reference ID, by default None.

        Yields
        ------
        tuple[int, Any]
            collection_index, value for each matching row.
        """
        
        query = f"""
            SELECT collection_index,value FROM {self._quote(self.analysis_table)}
            WHERE waveform=? AND analysis_name=? AND result_name=? AND reference_id IS ?
            ORDER BY collection_index
        """
        for collection_index, value in self.connection.execute(query, (waveform, analysis_name, result_name, reference_id)):
            value = self.deserialize_array(value) if isinstance(value, (bytes, bytearray, memoryview)) else value
            yield collection_index, value

    def list_analysis_results(self, collection_index: int | None = None, waveform: str | None = None, analysis_name: str | None = None, result_name: str | None = None) -> list[dict[str, Any]]:
        """List analysis results from the analysis table with optional filters.

        Parameters
        ----------
        collection_index : int | None, optional
            The index of the collection to filter by, by default None.
        waveform : str | None, optional
            The waveform to filter by, by default None.
        analysis_name : str | None, optional
            The analysis name to filter by, by default None.
        result_name : str | None, optional
            The result name to filter by, by default None.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries containing the analysis results.
        """
        conditions, params = [], []

        if collection_index is not None:
            conditions.append("collection_index=?")
            params.append(collection_index)
        if waveform is not None:
            conditions.append("waveform=?")
            params.append(waveform)
        if analysis_name is not None:
            conditions.append("analysis_name=?")
            params.append(analysis_name)
        if result_name is not None:
            conditions.append("result_name=?")
            params.append(result_name)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT collection_index,waveform,analysis_name,result_name,x_axis,x_unit,y_axis,y_unit,reference_id
            FROM {self._quote(self.analysis_table)} {where}
            ORDER BY collection_index
        """

        columns = ["collection_index", "waveform", "analysis_name", "result_name", "x_axis", "x_unit", "y_axis", "y_unit", "reference_id"]
        return [dict(zip(columns, row)) for row in self.connection.execute(query, params)]

    def unique_tuples(self, column_names: list[str] | tuple[str, ...], table: str | None = None) -> set[tuple[Any, ...]]:
        """Return the unique tuples for the specified columns.

        Parameters
        ----------
        column_names : list[str] | tuple[str, ...]
            Names of the columns to include in each tuple.
        table : str | None, optional
            Table to query. Defaults to the analysis table.

        Returns
        -------
        set[tuple[Any, ...]]
            The distinct combinations of the requested column values.
        """
        if not column_names:
            raise ValueError("column_names must contain at least one column")

        table_name = table or self.analysis_table
        selected_columns = ",".join(self._quote(column) for column in column_names)
        query = f"SELECT DISTINCT {selected_columns} FROM {self._quote(table_name)}"
        return {tuple(row) for row in self.connection.execute(query)}
    
    def _compute_analysis_maxs(self, lazy: bool = True) -> None: # TODO: fix this
        print("Fetching pending...")
        if lazy: # write non lazer version
            pending = self.connection.execute(f"""
                SELECT rowid,waveform,analysis_name,result_name,reference_id,value
                FROM {self._quote(self.analysis_table)}
                WHERE max_value IS NULL
            """).fetchall()       
        else:
            pending = self.connection.execute(f"""
                SELECT rowid,waveform,analysis_name,result_name,reference_id,value
                FROM {self._quote(self.analysis_table)}
            """).fetchall()
        if pending:
            touched_keys = set()
            update_params = []
            for rowid, waveform, analysis_name, result_name, reference_id, value in tqdm(pending, total=len(pending), desc="Computing analysis max values"):
                data = self.deserialize_array(value) if isinstance(value, (bytes, bytearray, memoryview)) else value
                row_max = self._absolute_max(data) if isinstance(data, np.ndarray) else abs(float(data))
                update_params.append((row_max, rowid))
                touched_keys.add((analysis_name, result_name, reference_id))

            self.connection.executemany(f"""
                UPDATE {self._quote(self.analysis_table)} SET max_value=? WHERE rowid=?
            """, update_params)
            self.connection.commit()
        self.set_global_analysis_maxs()

    @profile
    def set_global_analysis_maxs(self) -> None:
        """Update the maximum values for all analysis results in the database.
        """
        print("Updating global analysis max values...")
        touched_keys = self.unique_tuples(["analysis_name", "result_name", "reference_id"], self.analysis_table)

        for key in touched_keys:
            # analysis_name, result_name, reference_id = key
            rows = self.connection.execute(f"""
                SELECT waveform,MAX(max_value) FROM {self._quote(self.analysis_table)}
                WHERE analysis_name=? AND result_name=? AND reference_id IS ?
                GROUP BY waveform
            """, key).fetchall()

            self.analysis_maxs_.setdefault(key, {}).update(dict(rows))
            
    # -------------------------------------------------------------------------
    # References
    # -------------------------------------------------------------------------

    def create_reference_table(self) -> None:
        """Create the analysis reference table in the SQLite database if it does not exist.
        """
        
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._quote(self.reference_table)} (
                reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_name TEXT NOT NULL,
                reference_name TEXT,
                value BLOB,
                x_axis BLOB,
                x_unit TEXT,
                metadata TEXT
            )
        """)
        self.connection.commit()
    
    def _find_reference(self, analysis_name: str, reference_name: str | None, value: Any, x_axis: Any, x_unit: str | None, metadata_json: str | None) -> int | None:
        """Check for an existing reference matching all fields exactly.

        Returns
        -------
        int | None
            The existing reference_id, or None if no match exists.
        """
        row = self.connection.execute(f"""
            SELECT reference_id FROM {self._quote(self.reference_table)}
            WHERE analysis_name=? AND reference_name IS ? AND value IS ?
            AND x_axis IS ? AND x_unit IS ? AND metadata IS ?
            LIMIT 1
        """, (analysis_name, reference_name, value, x_axis, x_unit, metadata_json)).fetchone()
        return int(row[0]) if row else None

    def create_reference(self, analysis_name: str, reference_name: str, value: Any = None, x_axis: Any | None = None, x_unit: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        """Create a new reference in the SQLite database, or return the existing matching one.
        """
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)
        if isinstance(x_axis, np.ndarray):
            x_axis = self.serialize_array(x_axis)
        metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None

        existing_id = self._find_reference(analysis_name, reference_name, value, x_axis, x_unit, metadata_json)
        if existing_id is not None:
            return existing_id

        query = f"""
            INSERT INTO {self._quote(self.reference_table)}
            (analysis_name,reference_name,value,x_axis,x_unit,metadata)
            VALUES (?,?,?,?,?,?)
        """
        cursor = self.connection.execute(query, (analysis_name, reference_name, value, x_axis, x_unit, metadata_json))
        self.connection.commit()
        return int(cursor.lastrowid)

    def fetch_reference(self, reference_id: int) -> dict[str, Any]:
        """Fetch a reference from the SQLite database by its ID.

        Parameters
        ----------
        reference_id : int
            The ID of the reference to fetch.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the reference data.
        """
        query = f"""
            SELECT reference_id,analysis_name,reference_name,value,x_axis,x_unit,metadata
            FROM {self._quote(self.reference_table)}
            WHERE reference_id=?
        """

        row = self.connection.execute(query, (reference_id,)).fetchone()

        if row is None:
            raise ValueError(f"Reference {reference_id} does not exist")

        value = self.deserialize_array(row[3]) if isinstance(row[3], (bytes, bytearray, memoryview)) else row[3]
        x_axis = self.deserialize_array(row[4]) if isinstance(row[4], (bytes, bytearray, memoryview)) else row[4]

        return {
            "reference_id": row[0],
            "analysis_name": row[1],
            "reference_name": row[2],
            "value": value,
            "x_axis": x_axis,
            "x_unit": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    def list_references(self, analysis_name: str | None = None) -> list[dict[str, Any]]:
        """List all references in the SQLite database for a type of analysis method (fft, cwt, etc).

        Parameters
        ----------
        analysis_name : str | None, optional
            The name of the analysis to filter by, or None to list all references, by default None.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, each containing a reference's data.
        """
        query = f"SELECT reference_id,analysis_name,reference_name FROM {self._quote(self.reference_table)}"
        params = ()

        if analysis_name is not None:
            query += " WHERE analysis_name=?"
            params = (analysis_name,)

        query += " ORDER BY reference_id"

        columns = ["reference_id", "analysis_name", "reference_name"]
        return [dict(zip(columns, row)) for row in self.connection.execute(query, params)]

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


class AcousticsScanDatabase(AcousticsDatabase):
    """Acoustics database subclass for 2D scan experiments (collection_index <-> X,Z coordinates).
    """
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.X_, self.Z_ = self.get_grid_shape()

    def clear_grid_cache(self) -> None:
        """Clear cached grid arrays for waveform and analysis results."""
        self._fetch_waveform_grid_cached.cache_clear()
        self._fetch_analysis_grid_cached.cache_clear()
        self._fetch_hilbert_grid_cached.cache_clear()

    # -------------------------------------------------------------------------
    # collection indexing
    # -------------------------------------------------------------------------

    def get_axis_names(self) -> tuple[str, str]:
        """Get the primary and secondary axis column names.

        Returnsk
        -------
        tuple[str, str]
            The primary and secondary axis names.
        """
        return self.parameters["primaryAxis"], self.parameters["secondaryAxis"]

    def get_grid_shape(self) -> tuple[int, int]:
        """Get the scan grid shape.

        Returns
        -------
        tuple[int, int]
            (n_secondary, n_primary) step counts.
        """
        n_primary = math.ceil(self.parameters["primaryAxisRange"] / abs(self.parameters["primaryAxisStep"])) + 1
        n_secondary = math.ceil(self.parameters["secondaryAxisRange"] / abs(self.parameters["secondaryAxisStep"])) + 1
        return  n_primary, n_secondary

    def fetch_collection_index(self, x: float, z: float, ) -> int:
        return x + -z*self.X_ 

    def fetch_time_index(self, t: float) -> tuple[float, int]:
        """Fetch the time index corresponding to a given time value.

        Parameters
        ----------
        t : float
            The time value.

        Returns
        -------
        tuple[float, int]
            The nearest stored time value and its index.
        """
        time_zero = self.parameters["measureDelay"] * 1e3
        time_step = self.parameters["dt"] * 1e9
        idx = int(round((t - time_zero) / time_step))
        return time_zero + idx * time_step, idx
    
    # -------------------------------------------------------------------------
    # acoustics
    # -------------------------------------------------------------------------
    def fetch_raw_image(self, waveform: str) -> np.ndarray:
        """Fetch a column's values reshaped into the scan's 2D grid.

        Parameters
        ----------
        waveform : str
            The name of the waveform.

        Returns
        -------
        np.ndarray
            Array of shape (n_secondary, n_primary).
        """
        values = self.fetch_column(waveform, self.acoustics_table)
        return np.asarray(values).reshape(self.Z_, self.X_)

    @lru_cache(maxsize=32)
    def _fetch_waveform_grid_cached(self, waveform: str) -> np.ndarray:
        """Cached waveform grid result."""
        if not self.has_waveform(waveform):
            raise ValueError(f"Unknown waveform: {waveform}")

        blobs = self.fetch_column(waveform, self.acoustics_table)
        waveforms = np.stack([self.deserialize_array(b) for b in blobs])
        return waveforms.reshape(self.Z_,self.X_,-1)

    def fetch_waveform_grid(self, waveform: str) -> np.ndarray:
        """Fetch a waveform column reshaped into the scan's 2D grid.

        Parameters
        ----------
        waveform : str
            The name of the waveform column.

        Returns
        -------
        np.ndarray
            Array of shape (n_secondary, n_primary, n_samples).
        """
        return self._fetch_waveform_grid_cached(waveform)

    def fetch_raw_image(self, waveform, t_idx):
        return self.fetch_waveform_grid(waveform)[:, :, t_idx]

    def fetch_raw_waveform(self, waveform, x_idx, z_idx):
        return self.fetch_waveform_grid(waveform)[z_idx, x_idx, :]
    
    # -------------------------------------------------------------------------
    # analysis
    # -------------------------------------------------------------------------

    @lru_cache(maxsize=32)
    def _fetch_analysis_grid_cached(self, waveform: str, analysis_name: str, result_name: str, reference_id: int | None) -> np.ndarray:
        """Cached analysis grid result.
        """
        query = f"""
            SELECT collection_index, value FROM {self._quote(self.analysis_table)}
            WHERE waveform=? AND analysis_name=? AND result_name=?
        """
        params = [waveform, analysis_name, result_name]

        if reference_id is not None:
            query += " AND reference_id=?"
            params.append(reference_id)

        query += " ORDER BY collection_index"
        rows = self.connection.execute(query, params).fetchall()

        expected = self.Z_ * self.X_
        if len(rows) != expected: raise ValueError(f"Expected {expected} results for grid, found {len(rows)}")

        values = [
            self.deserialize_array(v) if isinstance(v, (bytes, bytearray, memoryview)) else v
            for _, v in rows
        ]

        if isinstance(values[0], np.ndarray):
            return np.stack(values).reshape(self.Z_, self.X_, -1)
        return np.asarray(values).reshape(self.Z_, self.X_)

    def fetch_analysis_grid(self, waveform: str, analysis_name: str, result_name: str, reference_id: int | None = None) -> np.ndarray:
        """Fetch an analysis result reshaped into the scan's 2D grid.

        Parameters
        ----------
        waveform : str
            The waveform name.
        analysis_name : str
            The name of the analysis.
        result_name : str
            The name of the result.
        reference_id : int | None, optional
            The reference ID, by default None.

        Returns
        -------
        np.ndarray
            Array of shape (n_secondary, n_primary) for scalar results, or
            (n_secondary, n_primary, n_samples) for array results.
        """
        return self._fetch_analysis_grid_cached(waveform, analysis_name, result_name, reference_id)

    @lru_cache(maxsize=32)
    def _fetch_hilbert_grid_cached(self, waveform: str, reference_id: int) -> np.ndarray:
        """Return the cached magnitude Hilbert transform of a processed grid."""
        processed_grid = self.fetch_analysis_grid(
            waveform,
            "preprocessed",
            "waveform",
            reference_id,
        )
        return np.abs(hilbert(processed_grid, axis=-1))

    def fetch_hilbert_grid(self, waveform: str, reference_id: int) -> np.ndarray:
        """Return a cached Hilbert-envelope grid for processed waveform data."""
        return self._fetch_hilbert_grid_cached(waveform, reference_id)

    def fetch_latest_analysis_reference(
        self,
        waveform: str,
        analysis_name: str,
        result_name: str,
    ) -> int:
        """Return the newest analysis reference for a waveform and result."""
        row = self.connection.execute(
            f"""
            SELECT reference_id
            FROM {self._quote(self.analysis_table)}
            WHERE waveform=? AND analysis_name=? AND result_name=?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (waveform, analysis_name, result_name),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"No {analysis_name}/{result_name} data is available for {waveform}"
            )
        return int(row[0])
  
      
    
# needs testing
class AcousticsSweepDatabase(AcousticsDatabase):
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
class AcousticsDatabaseFrequencyDomain(AcousticsDatabase):
    """A class for handling frequency domain data in an acoustics database.

    This class extends the AcousticsDatabase class to provide additional functionality
    for working with frequency domain data, such as FFT results.
    """

    def __init__(self, database_path: str):
        """Initialize the AcousticsDatabaseFrequencyDomain instance.

        Parameters
        ----------
        database_path : str
            The path to the SQLite database file.
        """
        super().__init__(database_path)
        
        
        
        
class AcousticsDatabaseCWT(AcousticsDatabaseFrequencyDomain):
    """A class for handling continuous wavelet transform data in an acoustics database.

    This class extends the AcousticsDatabaseFrequencyDomain class to provide additional functionality
    for working with continuous wavelet transform data.
    """

    def __init__(self, database_path: str):
        """Initialize the AcousticsDatabaseCWT instance.

        Parameters
        ----------
        database_path : str
            The path to the SQLite database file.
        """
        super().__init__(database_path)
        
        

class AcousticsDatabaseTMM(AcousticsDatabaseFrequencyDomain):
    """A class for handling time-frequency map data in an acoustics database.

    This class extends the AcousticsDatabase class to provide additional functionality
    for working with time-frequency map data.
    """

    def __init__(self, database_path: str):
        """Initialize the AcousticsDatabaseTMM instance.

        Parameters
        ----------
        database_path : str
            The path to the SQLite database file.
        """
        super().__init__(database_path)
        