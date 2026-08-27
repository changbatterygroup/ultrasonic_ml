from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


class AcousticsDatabase:
    """Base class for SQLite acoustics data and analysis."""

    def __init__(self, database_path: str | Path) -> None:
        '''Initialize the AcousticsDatabase.
        Args:
            database_path (str | Path): The path to the SQLite database file.
        '''
        
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

        self.connect()
        self.waveform_columns = self.get_waveform_columns()
        self.parameters = self.get_parameters()
        self.initialize_frequency_parameters()
        self.create_analysis_table()
        self.create_reference_table()

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        '''Connect to the SQLite database.
        '''
        self.connection = sqlite3.connect(self.database_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def close(self) -> None:
        '''Close the SQLite database connection.
        '''
        if self.connection:
            self.connection.close()
        self.connection = self.cursor = None

    def __enter__(self) -> AcousticsDatabase:
        '''Enter the context manager.
        Returns:
            The AcousticsDatabase instance.
        '''
        
        return self

    def __exit__(self, *args: Any) -> None:
        '''Exit the context manager and close the database connection.
        Args:
            *args (Any): The exception type, value, and traceback.
        '''
        self.close()

    def get_tables(self) -> list[str]:
        '''Get the list of tables in the SQLite database.
        Returns:
            A list of table names.
        '''
        rows = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r[0] for r in rows]

    def get_columns(self, table: str) -> list[str]:
        '''Get the list of columns in a table.
        Args:
            table (str): The name of the table.
        Returns:
            A list of column names.'''
        rows = self.connection.execute(f"PRAGMA table_info({self._quote(table)})").fetchall()
        return [r[1] for r in rows]

    def fetch_column(self, column: str, table: str = "acoustics") -> list[Any]:
        '''Fetch all values from a specific column in a table.
        Args:
            column (str): The name of the column.
            table (str): The name of the table. Defaults to "acoustics".
        Returns:
            A list of values from the specified column.'''
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)}"
        return [r[0] for r in self.connection.execute(query)]

    def fetch_value(self, column: str, row: int, table: str = "acoustics") -> Any:
        '''Fetch a single value from a specific column and row in a table.
        Args:
            column (str): The name of the column.
            row (int): The row index.
            table (str): The name of the table. Defaults to "acoustics".
        Returns:
            The value from the specified column and row.'''
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
        '''Get the parameters from the parameters table.
        Returns:
            A dictionary of parameters.
        '''
        columns = self.get_columns(self.parameters_table)
        if not columns:
            return {}
        query = f"SELECT {','.join(map(self._quote, columns))} FROM {self._quote(self.parameters_table)} LIMIT 1"
        row = self.connection.execute(query).fetchone()
        return dict(zip(columns, row)) if row else {}

    def initialize_frequency_parameters(self) -> None:
        '''Initialize frequency-related parameters from the database.
        '''
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
                self._add_parameter_column(key)
                self._write_parameter(key, value)

        self.parameters.update({k: v for k, v in values.items() if v is not None})

    def get_datetime(self, row: int = 0) -> datetime:
        '''Get the datetime of a specific acquisition.
        Args:
            row (int): The row index of the acquisition.
        Returns:
            The datetime of the acquisition.'''
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

    def get_waveform_columns(self) -> list[str]:
        '''Get the list of waveform columns in the acoustics table.
        Returns:
            A list of waveform column names.'''
        excluded = {"collection_index", "X", "Z", "time", "time_collected", "frequency"}
        return [c for c in self.get_columns(self.acoustics_table) if c not in excluded]

    def has_waveform(self, waveform: str) -> bool:
        '''Check if a waveform column exists in the acoustics table.
        Args:
            waveform (str): The name of the waveform column.
        Returns:
            True if the waveform column exists, False otherwise.'''
        return waveform in self.waveform_columns

    def fetch_waveform(self, waveform: str, row: int) -> np.ndarray:
        '''Fetch a waveform from the acoustics table.
        Args:
            waveform (str): The name of the waveform column.
            row (int): The row index of the acquisition.
        Returns:
            The waveform as a NumPy array.'''
        
        if not self.has_waveform(waveform):
            raise ValueError(f"Unknown waveform: {waveform}")
        return self.deserialize_array(self.fetch_value(waveform, row))

    def fetch_waveform_batch(self, waveform: str, rows: Any) -> list[np.ndarray]:
        '''Fetch a batch of waveforms from the acoustics table.
        Args:
            waveform (str): The name of the waveform column.
            rows (Any): An iterable of row indices.
        Returns:
            A list of waveforms as NumPy arrays.'''
        return [self.fetch_waveform(waveform, int(row)) for row in rows]

    def fetch_time(self, row: int = 0) -> np.ndarray:
        '''Fetch the time array from the acoustics table.
        Args:
            row (int): The row index of the acquisition.
        Returns:
            The time array as a NumPy array.'''
        return self.deserialize_array(self.fetch_value("time", row))

    def fetch_waveform_value(self, waveform: str, index: int, row: int) -> float:
        '''Fetch a specific value from a waveform at a given index and row.
        Args:
            waveform (str): The name of the waveform column.
            index (int): The index of the value in the waveform array.
            row (int): The row index of the acquisition.
        Returns:
            The value from the waveform at the specified index and row.'''
        return float(self.fetch_waveform(waveform, row)[index])

    def fetch_index_across_acquisitions(self, waveform: str, index: int) -> np.ndarray:
        '''Fetch a specific index value from a waveform across all acquisitions.
        Args:
            waveform (str): The name of the waveform column.
            index (int): The index of the value in the waveform array.
        Returns:
            A NumPy array of values from the waveform at the specified index across all acquisitions.'''
        query = f"SELECT {self._quote(waveform)} FROM {self._quote(self.acoustics_table)}"
        return np.asarray([self.deserialize_array(r[0])[index] for r in self.connection.execute(query)])

    def get_acquisition_count(self) -> int:
        '''Get the total number of acquisitions in the acoustics table.
        Returns:
            The total number of acquisitions.'''
        
        query = f"SELECT COUNT(*) FROM {self._quote(self.acoustics_table)}"
        return int(self.connection.execute(query).fetchone()[0])

    def get_acquisition_index(self, row: int) -> int:
        '''Get the collection index of a specific acquisition.
        Args:
            row (int): The row index of the acquisition.
        Returns:
            The collection index of the acquisition.'''
        return int(self.fetch_value(self.index_column, row))

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def preprocess(self, waveform: str, apply_ungain: bool = False, apply_filter: bool = False, gain_column: str | None = None, offset_column: str | None = None, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, filter_order: int = 3, calculate_absolute_max: bool = True) -> int:
        '''Preprocess a waveform.
        Args:
            waveform (str): The name of the waveform column.
            apply_ungain (bool): Whether to remove gain.
            apply_filter (bool): Whether to apply a filter.
            gain_column (str | None): The name of the gain column.
            offset_column (str | None): The name of the offset column.
            lower_fs_coeff (float): The lower sampling frequency coefficient. Highpass filter lower limit of C*f_s 
            upper_fs_coeff (float): The upper sampling frequency coefficient. Lowpass filter upper limit of C*f_s
            filter_order (int): The order of the filter.
            calculate_absolute_max (bool): Whether to calculate the absolute maximum.
        Returns:
            The reference ID of the preprocessed data.'''
        if not self.has_waveform(waveform):
            raise ValueError(f"Unknown waveform: {waveform}")
        if apply_ungain and gain_column is None:
            raise ValueError("gain_column is required when apply_ungain=True")
        if apply_filter and self.f_s is None:
            raise ValueError("Sampling frequency is unavailable")

        sos = None
        if apply_filter:
            sos = butter(filter_order, [self.f_s * lower_fs_coeff, self.f_s * upper_fs_coeff], btype="bandpass", fs=self.f_s, output="sos")

        metadata = {
            "waveform": waveform,
            "apply_ungain": apply_ungain,
            "apply_filter": apply_filter,
            "gain_column": gain_column,
            "offset_column": offset_column,
            "lower_fs_coeff": lower_fs_coeff,
            "upper_fs_coeff": upper_fs_coeff,
            "filter_order": filter_order,
            "calculate_absolute_max": calculate_absolute_max,
        }

        reference_id = self.create_reference("preprocessed", waveform, metadata=metadata)

        columns = [self._quote(waveform)]
        if gain_column:
            columns.append(self._quote(gain_column))
        if offset_column:
            columns.append(self._quote(offset_column))

        query = f"SELECT collection_index,{','.join(columns)} FROM {self._quote(self.acoustics_table)} ORDER BY collection_index"

        overall_max = 0.0

        for row in self.connection.execute(query):
            collection_index, blob = row[:2]
            waveform_data = self.deserialize_array(blob).astype(np.float32, copy=False)
            i = 2

            if apply_ungain:
                gain = float(row[i])
                i += 1
                offset = float(row[i]) if offset_column else 0.0
                waveform_data = self._undo_gain(waveform_data, gain, offset)

            if apply_filter:
                waveform_data = self._butterworth_filter(waveform_data, lower_fs_coeff, upper_fs_coeff, filter_order, sos)

            maximum = self._absolute_max(waveform_data)
            overall_max = max(overall_max, maximum)

            self.store_analysis_result(collection_index, waveform, "preprocessed", "waveform", waveform_data, "time", "ns", "voltage", "mV", reference_id)

            if calculate_absolute_max:
                self.store_analysis_result(collection_index, waveform, "preprocessed", "absolute_max", maximum, None, None, None, "mV", reference_id)

        self.connection.commit()
        self.absolute_max[waveform] = overall_max
        self.absolute_max_overall = max(self.absolute_max.values())
        return reference_id

    def fetch_preprocessed_waveform(self, waveform: str, row: int, reference_id: int | None = None) -> np.ndarray:
        '''Fetch the preprocessed waveform for a specific acquisition.
        Args:
            waveform (str): The name of the waveform.
            row (int): The row index of the acquisition.
            reference_id (int | None): The reference ID, if applicable.
        Returns:
            np.ndarray: The preprocessed waveform.
        '''
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

    def _undo_gain(self, waveform: np.ndarray, gain: float, offset: float = 0.0) -> np.ndarray:
        '''Undo the gain and offset applied to a waveform.
        Args:
            waveform (np.ndarray): The waveform data.
            gain (float): The gain value to undo.
            offset (float): The offset value to undo. Defaults to 0.0.
        Returns:
            np.ndarray: The waveform with the gain and offset undone.
        '''
        return (waveform - offset) / gain

    def _butterworth_filter(self, waveform: np.ndarray, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, order: int = 3, sos: Any | None = None) -> np.ndarray:
        '''Apply a Butterworth bandpass filter to the waveform.
        Args:
            waveform (np.ndarray): The waveform data.
            lower_fs_coeff (float): The lower frequency coefficient. Defaults to 1/250.
            upper_fs_coeff (float): The upper frequency coefficient. Defaults to 1/5.
            order (int): The order of the filter. Defaults to 3.
            sos (Any | None): The second-order sections of the filter. Defaults to None.
        Returns:
            np.ndarray: The filtered waveform.
        '''
        if self.f_s is None:
            raise ValueError("Sampling frequency is unavailable")

        if sos is None:
            sos = butter(order, [self.f_s * lower_fs_coeff, self.f_s * upper_fs_coeff], btype="bandpass", fs=self.f_s, output="sos")

        return sosfiltfilt(sos, waveform)

    @staticmethod
    def _absolute_max(waveform: np.ndarray) -> float:
        '''Calculate the absolute maximum of a waveform.
        Args:
            waveform (np.ndarray): The waveform data.
        Returns:
            float: The absolute maximum of the waveform.
        '''
        return float(np.max(np.abs(waveform)))

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def create_analysis_table(self) -> None:
        '''Create the analysis table in the SQLite database if it does not exist.'''
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
                reference_id INTEGER,
                PRIMARY KEY (collection_index,waveform,analysis_name,result_name,reference_id)
            )
        """)
        self.connection.commit()

    def store_analysis_result(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, value: Any, x_axis: str | None = None, x_unit: str | None = None, y_axis: str | None = None, y_unit: str | None = None, reference_id: int | None = None) -> None:
        '''Store an analysis result in the analysis table.
        Args:
            collection_index (int): The index of the collection.
            waveform (str): The waveform data.
            analysis_name (str): The name of the analysis.
            result_name (str): The name of the result.
            value (Any): The value of the result.
            x_axis (str | None): The x-axis label. Defaults to None.
            x_unit (str | None): The x-axis unit. Defaults to None.
            y_axis (str | None): The y-axis label. Defaults to None.
            y_unit (str | None): The y-axis unit. Defaults to None.
            reference_id (int | None): The reference ID. Defaults to None.
        '''
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)

        query = f"""
            INSERT OR REPLACE INTO {self._quote(self.analysis_table)}
            (collection_index,waveform,analysis_name,result_name,value,x_axis,x_unit,y_axis,y_unit,reference_id)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """

        self.connection.execute(query, (collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id))

    def fetch_analysis_result(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, reference_id: int | None = None) -> Any:
        '''Fetch an analysis result from the analysis table.
        Args:
            collection_index (int): The index of the collection.
            waveform (str): The waveform data.
            analysis_name (str): The name of the analysis.
            result_name (str): The name of the result.
            reference_id (int | None): The reference ID. Defaults to None.
        Returns:
            Any: The value of the analysis result.
        '''
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

        query = f"""
            SELECT x_axis, x_unit, y_axis, y_unit FROM {self._quote(self.analysis_table)}
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

        x_axis, x_unit, y_unit = result[0], result[1], result[2]
        return {
            "x_axis": self.deserialize_array(x_axis) if isinstance(x_axis, (bytes, bytearray, memoryview)) else x_axis,
            "x_unit": x_unit,
            "y_unit": y_unit
        }

    def list_analysis_results(self, collection_index: int | None = None, waveform: str | None = None, analysis_name: str | None = None, result_name: str | None = None) -> list[dict[str, Any]]:
        '''List analysis results from the analysis table with optional filters.
        Args:
            collection_index (int | None): The index of the collection to filter by. Defaults to None.
            waveform (str | None): The waveform to filter by. Defaults to None.
            analysis_name (str | None): The analysis name to filter by. Defaults to None.
            result_name (str | None): The result name to filter by. Defaults to None.
        Returns:
            A list of dictionaries containing the analysis results.'''
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

    # -------------------------------------------------------------------------
    # References
    # -------------------------------------------------------------------------

    def create_reference_table(self) -> None:
        '''Create the analysis reference table in the SQLite database if it does not exist.'''
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._quote(self.reference_table)} (
                reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_name TEXT NOT NULL,
                reference_name TEXT NOT NULL,
                value BLOB,
                x_axis BLOB,
                x_unit TEXT,
                metadata TEXT
            )
        """)
        self.connection.commit()

    def create_reference(self, analysis_name: str, reference_name: str, value: Any = None, x_axis: Any | None = None, x_unit: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        '''Create a new reference in the SQLite database.'''
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)
        if isinstance(x_axis, np.ndarray):
            x_axis = self.serialize_array(x_axis)

        query = f"""
            INSERT INTO {self._quote(self.reference_table)}
            (analysis_name,reference_name,value,x_axis,x_unit,metadata)
            VALUES (?,?,?,?,?,?)
        """

        cursor = self.connection.execute(query, (analysis_name, reference_name, value, x_axis, x_unit, json.dumps(metadata) if metadata else None))
        self.connection.commit()
        return int(cursor.lastrowid)

    def fetch_reference(self, reference_id: int) -> dict[str, Any]:
        '''Fetch a reference from the SQLite database by its ID.
        Args:
            reference_id (int): The ID of the reference to fetch.
        Returns:
            A dictionary containing the reference data.'''
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
        '''List all references in the SQLite database for a type of analysis method (fft, cwt, etc)
        Args:
            analysis_name (str | None): The name of the analysis to filter by, or None to list all references.
        Returns:
            A list of dictionaries, each containing a reference's data.'''
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
        '''Serialize a NumPy array to bytes for storage in the SQLite database.
        Args:
            array (np.ndarray): The NumPy array to serialize.
        Returns:
            bytes: The serialized byte representation of the array.'''
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def deserialize_array(blob: Any) -> np.ndarray:
        '''Deserialize bytes from the SQLite database back into a NumPy array.
        Args:
            blob (Any): The byte representation of the array.
        Returns:
            np.ndarray: The deserialized NumPy array.'''
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

    def _add_parameter_column(self, column: str) -> None:
        '''Add a new column to the parameters table if it does not already exist.
        Args:
            column (str): The name of the column to add.'''
        if column not in self.get_columns(self.parameters_table):
            self.connection.execute(f"ALTER TABLE {self._quote(self.parameters_table)} ADD COLUMN {self._quote(column)} REAL")

    def _write_parameter(self, column: str, value: Any) -> None:
        '''Write a parameter value to the parameters table.
        Args:
            column (str): The name of the parameter column.
            value (Any): The value to write to the parameter column.'''
        self.connection.execute(f"UPDATE {self._quote(self.parameters_table)} SET {self._quote(column)}=?", (value,))
        self.connection.commit()

    def _check_column(self, table: str, column: str) -> None:
        '''Check if a column exists in a table.
        Args:
            table (str): The name of the table.
            column (str): The name of the column.
        Raises:
            ValueError: If the column does not exist in the table.'''
        if column not in self.get_columns(table):
            raise ValueError(f"Column '{column}' does not exist in '{table}'")

    @staticmethod
    def _quote(identifier: str) -> str:
        '''Replace quotes with double quotesin an identifier (table or column name) for use in SQL queries.
        Args:
            identifier (str): The identifier to quote.
        Returns:
            str: The quoted identifier.'''
        return '"' + identifier.replace('"', '""') + '"'