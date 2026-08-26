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
        self.connection = sqlite3.connect(self.database_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
        self.connection = self.cursor = None

    def __enter__(self) -> AcousticsDatabase:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_tables(self) -> list[str]:
        rows = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r[0] for r in rows]

    def get_columns(self, table: str) -> list[str]:
        rows = self.connection.execute(f"PRAGMA table_info({self._quote(table)})").fetchall()
        return [r[1] for r in rows]

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        columns = self.get_columns(self.parameters_table)
        if not columns:
            return {}
        query = f"SELECT {','.join(map(self._quote, columns))} FROM {self._quote(self.parameters_table)} LIMIT 1"
        row = self.connection.execute(query).fetchone()
        return dict(zip(columns, row)) if row else {}

    def initialize_frequency_parameters(self) -> None:
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
        value = self.fetch_value("time_collected", row)
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if isinstance(value, (int, float, np.number)):
            return datetime.fromtimestamp(float(value))
        raise TypeError(f"Unsupported time_collected type: {type(value)}")

    # -------------------------------------------------------------------------
    # Waveforms
    # -------------------------------------------------------------------------

    def get_waveform_columns(self) -> list[str]:
        excluded = {"collection_index", "X", "Z", "time", "time_collected", "frequency"}
        return [c for c in self.get_columns(self.acoustics_table) if c not in excluded]

    def has_waveform(self, waveform: str) -> bool:
        return waveform in self.waveform_columns

    def fetch_column(self, column: str, table: str = "acoustics") -> list[Any]:
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)}"
        return [r[0] for r in self.connection.execute(query)]

    def fetch_value(self, column: str, row: int, table: str = "acoustics") -> Any:
        self._check_column(table, column)
        query = f"SELECT {self._quote(column)} FROM {self._quote(table)} LIMIT 1 OFFSET ?"
        result = self.connection.execute(query, (row,)).fetchone()
        if result is None:
            raise IndexError(f"Row {row} does not exist.")
        return result[0]

    def fetch_waveform(self, waveform: str, row: int) -> np.ndarray:
        if not self.has_waveform(waveform):
            raise ValueError(f"Unknown waveform: {waveform}")
        return self.deserialize_array(self.fetch_value(waveform, row))

    def fetch_waveform_batch(self, waveform: str, rows: Any) -> list[np.ndarray]:
        return [self.fetch_waveform(waveform, int(row)) for row in rows]

    def fetch_time(self, row: int = 0) -> np.ndarray:
        return self.deserialize_array(self.fetch_value("time", row))

    def fetch_waveform_value(self, waveform: str, index: int, row: int) -> float:
        return float(self.fetch_waveform(waveform, row)[index])

    def fetch_index_across_acquisitions(self, waveform: str, index: int) -> np.ndarray:
        query = f"SELECT {self._quote(waveform)} FROM {self._quote(self.acoustics_table)}"
        return np.asarray([self.deserialize_array(r[0])[index] for r in self.connection.execute(query)])

    def get_acquisition_count(self) -> int:
        query = f"SELECT COUNT(*) FROM {self._quote(self.acoustics_table)}"
        return int(self.connection.execute(query).fetchone()[0])

    def get_acquisition_index(self, row: int) -> int:
        return int(self.fetch_value(self.index_column, row))

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def preprocess(self, waveform: str, apply_ungain: bool = False, apply_filter: bool = False, gain_column: str | None = None, offset_column: str | None = None, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, filter_order: int = 3, calculate_absolute_max: bool = True) -> int:
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
        return (waveform - offset) / gain

    def _butterworth_filter(self, waveform: np.ndarray, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, order: int = 3, sos: Any | None = None) -> np.ndarray:
        if self.f_s is None:
            raise ValueError("Sampling frequency is unavailable")

        if sos is None:
            sos = butter(order, [self.f_s * lower_fs_coeff, self.f_s * upper_fs_coeff], btype="bandpass", fs=self.f_s, output="sos")

        return sosfiltfilt(sos, waveform)

    @staticmethod
    def _absolute_max(waveform: np.ndarray) -> float:
        return float(np.max(np.abs(waveform)))

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def create_analysis_table(self) -> None:
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
        if isinstance(value, np.ndarray):
            value = self.serialize_array(value)

        query = f"""
            INSERT OR REPLACE INTO {self._quote(self.analysis_table)}
            (collection_index,waveform,analysis_name,result_name,value,x_axis,x_unit,y_axis,y_unit,reference_id)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """

        self.connection.execute(query, (collection_index, waveform, analysis_name, result_name, value, x_axis, x_unit, y_axis, y_unit, reference_id))

    def fetch_analysis_result(self, collection_index: int, waveform: str, analysis_name: str, result_name: str, reference_id: int | None = None) -> Any:
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
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def deserialize_array(blob: Any) -> np.ndarray:
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
        if column not in self.get_columns(self.parameters_table):
            self.connection.execute(f"ALTER TABLE {self._quote(self.parameters_table)} ADD COLUMN {self._quote(column)} REAL")

    def _write_parameter(self, column: str, value: Any) -> None:
        self.connection.execute(f"UPDATE {self._quote(self.parameters_table)} SET {self._quote(column)}=?", (value,))
        self.connection.commit()

    def _check_column(self, table: str, column: str) -> None:
        if column not in self.get_columns(table):
            raise ValueError(f"Column '{column}' does not exist in '{table}'")

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'