import os
import json
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple
from .errors import SchemaError, CorruptionError

HEADER_FORMAT = ">IIB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SUPPORTED_TYPES = {
    "INT": int,
    "FLOAT": (int, float),
    "STR": str,
    "BOOL": bool,
}


def safe_replace(src: str, dst: str) -> None:
    """Windows/OneDrive-safe atomic file replacement with brief retry on file lock."""
    for attempt in range(15):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(0.005)


class Column:
    def __init__(
        self,
        name: str,
        data_type: str,
        not_null: bool = False,
        unique: bool = False,
        foreign_key: Optional[Tuple[str, str, str]] = None,
    ):
        data_type = data_type.upper()
        if data_type not in SUPPORTED_TYPES:
            raise SchemaError(f"Unsupported column type '{data_type}'. Allowed: {list(SUPPORTED_TYPES.keys())}")
        self.name = name
        self.data_type = data_type
        self.not_null = not_null
        self.unique = unique
        # (ref_table, ref_column, on_delete_action: "RESTRICT" | "CASCADE")
        self.foreign_key = tuple(foreign_key) if foreign_key else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "not_null": self.not_null,
            "unique": self.unique,
            "foreign_key": list(self.foreign_key) if self.foreign_key else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Column":
        fk_raw = data.get("foreign_key")
        fk_tuple = (fk_raw[0], fk_raw[1], fk_raw[2]) if fk_raw else None
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            not_null=data.get("not_null", False),
            unique=data.get("unique", False),
            foreign_key=fk_tuple,
        )


class TableSchema:
    def __init__(self, table_name: str, columns: List[Column]):
        if not columns:
            raise SchemaError("Table must define at least one column.")
        seen = set()
        for col in columns:
            if col.name in seen:
                raise SchemaError(f"Duplicate column name '{col.name}' in table '{table_name}'.")
            seen.add(col.name)
        self.table_name = table_name
        self.columns = columns
        self.col_map = {c.name: c for c in columns}

    def validate_and_coerce(self, record: Dict[str, Any], existing_records: List[Dict[str, Any]], ignore_row_id: Optional[int] = None) -> Dict[str, Any]:
        for key in record:
            if key not in self.col_map and key != "_rid":
                raise SchemaError(f"Unknown column '{key}' for table '{self.table_name}'.")

        validated: Dict[str, Any] = {}
        for col in self.columns:
            val = record.get(col.name, None)

            if val is None:
                if col.not_null or col.unique:
                    raise SchemaError(f"Column '{col.name}' cannot be NULL.")
                validated[col.name] = None
                continue

            if col.data_type == "INT" and isinstance(val, bool):
                raise SchemaError(f"Column '{col.name}' expects INT, got BOOL.")
            if col.data_type == "FLOAT" and isinstance(val, bool):
                raise SchemaError(f"Column '{col.name}' expects FLOAT, got BOOL.")

            expected_py_type = SUPPORTED_TYPES[col.data_type]
            if not isinstance(val, expected_py_type):
                raise SchemaError(
                    f"Column '{col.name}' expects {col.data_type}, got {type(val).__name__} ({val!r})."
                )

            if col.data_type == "FLOAT":
                val = float(val)

            if col.unique:
                for row in existing_records:
                    if ignore_row_id is not None and row.get("_rid") == ignore_row_id:
                        continue
                    if row.get(col.name) == val:
                        raise SchemaError(f"UNIQUE constraint failed on '{self.table_name}.{col.name}' for value {val!r}.")

            validated[col.name] = val

        return validated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSchema":
        return cls(
            table_name=data["table_name"],
            columns=[Column.from_dict(c) for c in data["columns"]],
        )


class BinaryStorageEngine:
    """
    Custom binary file storage using length-prefixed frames with CRC32 integrity checks
    and tombstone markers for O(1) in-place deletes.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta_path = os.path.join(self.data_dir, "catalog.meta.json")

    def _table_file(self, table_name: str) -> str:
        return os.path.join(self.data_dir, f"{table_name}.mdb")

    def save_catalog(self, schemas: Dict[str, TableSchema], next_rids: Dict[str, int]) -> None:
        payload = {
            "schemas": {name: s.to_dict() for name, s in schemas.items()},
            "next_rids": next_rids,
        }
        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        safe_replace(tmp_path, self.meta_path)

    def load_catalog(self) -> Tuple[Dict[str, TableSchema], Dict[str, int]]:
        if not os.path.exists(self.meta_path):
            return {}, {}
        with open(self.meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        schemas = {name: TableSchema.from_dict(s) for name, s in payload.get("schemas", {}).items()}
        next_rids = payload.get("next_rids", {})
        return schemas, next_rids

    def append_record(self, table_name: str, record: Dict[str, Any]) -> int:
        file_path = self._table_file(table_name)
        raw_bytes = json.dumps(record, separators=(",", ":")).encode("utf-8")
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        header = struct.pack(HEADER_FORMAT, crc, len(raw_bytes), 0)

        with open(file_path, "ab") as f:
            offset = f.tell()
            f.write(header + raw_bytes)
            f.flush()
            os.fsync(f.fileno())
        return offset

    def mark_deleted(self, table_name: str, offset: int) -> None:
        file_path = self._table_file(table_name)
        if not os.path.exists(file_path):
            return
        with open(file_path, "r+b") as f:
            f.seek(offset + 8)
            f.write(struct.pack(">B", 1))
            f.flush()
            os.fsync(f.fileno())

    def load_table_records(self, table_name: str) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        file_path = self._table_file(table_name)
        if not os.path.exists(file_path):
            return [], {}

        records: List[Dict[str, Any]] = []
        offsets: Dict[int, int] = {}

        with open(file_path, "rb") as f:
            while True:
                offset = f.tell()
                header_bytes = f.read(HEADER_SIZE)
                if not header_bytes:
                    break
                if len(header_bytes) < HEADER_SIZE:
                    break

                expected_crc, length, tombstone = struct.unpack(HEADER_FORMAT, header_bytes)
                payload_bytes = f.read(length)
                if len(payload_bytes) < length:
                    break

                actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    raise CorruptionError(
                        f"Data corruption detected in '{table_name}.mdb' at offset {offset}."
                    )

                if tombstone == 0:
                    rec = json.loads(payload_bytes.decode("utf-8"))
                    records.append(rec)
                    offsets[rec["_rid"]] = offset

        return records, offsets

    def rewrite_table(self, table_name: str, active_records: List[Dict[str, Any]]) -> Dict[int, int]:
        file_path = self._table_file(table_name)
        tmp_path = file_path + ".tmp"
        new_offsets: Dict[int, int] = {}

        with open(tmp_path, "wb") as f:
            for rec in active_records:
                offset = f.tell()
                raw_bytes = json.dumps(rec, separators=(",", ":")).encode("utf-8")
                crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
                header = struct.pack(HEADER_FORMAT, crc, len(raw_bytes), 0)
                f.write(header + raw_bytes)
                new_offsets[rec["_rid"]] = offset
            f.flush()
            os.fsync(f.fileno())

        safe_replace(tmp_path, file_path)
        return new_offsets