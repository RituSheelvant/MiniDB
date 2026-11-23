import copy
import fnmatch
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .errors import ExecutionError, SchemaError, TransactionError
from .index import BaseIndex, BTreeRangeIndex, HashIndex, QueryPlanner
from .parser import (
    AlterTableStmt,
    Condition,
    CreateIndexStmt,
    CreateTableStmt,
    DeleteStmt,
    DescribeTableStmt,
    DropIndexStmt,
    DropTableStmt,
    ExplainStmt,
    HavingClause,
    HavingCondition,
    InsertStmt,
    SelectItem,
    SelectStmt,
    ShowTablesStmt,
    UpdateStmt,
    VacuumStmt,
    WhereClause,
    parse_query,
)
from .storage import BinaryStorageEngine, Column, TableSchema, safe_replace
from .wal import WriteAheadLog


class ReadWriteLock:
    """
    Allows multiple concurrent readers while guaranteeing exclusive access for writers.
    Prevents reader/writer race conditions across threads.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer = False

    @contextmanager
    def read_lock(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self):
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


class MiniDB:
    """
    Embeddable custom database engine supporting custom query parsing,
    parameterized queries, schema evolution (ALTER TABLE), foreign keys
    (REFERENCES / ON DELETE CASCADE & RESTRICT), DISTINCT, HAVING, column aliases (AS),
    multi-field & composite indexing, OR index unions, INNER/LEFT multi-table JOINs,
    WAL crash recovery, thread-safe concurrency, and Pythonic helper APIs.
    """

    def __init__(self, db_dir: str = "./minidb_data"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.storage = BinaryStorageEngine(self.db_dir)
        self.wal = WriteAheadLog(os.path.join(self.db_dir, "minidb.wal"))
        self.rwlock = ReadWriteLock()

        self.schemas: Dict[str, TableSchema] = {}
        self.next_rids: Dict[str, int] = {}
        self.tables: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.offsets: Dict[str, Dict[int, int]] = {}
        self.indexes: Dict[str, Dict[Tuple[str, ...], BaseIndex]] = {}
        self.index_meta: List[Dict[str, Any]] = []

        self._tx_local = threading.local()
        self._closed = False
        self.last_recovery_stats: Dict[str, Any] = {}

        self._startup_and_recover()

    def __enter__(self) -> "MiniDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._get_tx_state() is not None:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        self.close()

    def close(self) -> None:
        """Flushes catalog metadata and marks the database handle closed."""
        if self._closed:
            return
        with self.rwlock.write_lock():
            if self._get_tx_state() is not None:
                self.rollback()
            self.storage.save_catalog(self.schemas, self.next_rids)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ExecutionError("Cannot operate on a closed MiniDB instance.")

    def _get_tx_state(self) -> Optional[Dict[str, Any]]:
        return getattr(self._tx_local, "tx", None)

    def _set_tx_state(self, state: Optional[Dict[str, Any]]) -> None:
        self._tx_local.tx = state

    def _startup_and_recover(self) -> None:
        self.schemas, self.next_rids = self.storage.load_catalog()

        for table_name in self.schemas:
            records, offsets = self.storage.load_table_records(table_name)
            self.tables[table_name] = {r["_rid"]: r for r in records}
            self.offsets[table_name] = offsets
            self.indexes[table_name] = {}

        idx_meta_path = os.path.join(self.db_dir, "indexes.meta.json")
        if os.path.exists(idx_meta_path):
            with open(idx_meta_path, "r", encoding="utf-8") as f:
                self.index_meta = json.load(f)

        committed_ops, discarded_uncommitted = self.wal.read_committed_operations()

        for entry in committed_ops:
            op = entry["op"]
            tbl = entry["table"]
            if op == "CREATE_TABLE":
                if tbl not in self.schemas and entry.get("schema"):
                    schema = TableSchema.from_dict(entry["schema"])
                    self.schemas[tbl] = schema
                    self.next_rids[tbl] = 1
                    self.tables[tbl] = {}
                    self.offsets[tbl] = {}
                    self.indexes[tbl] = {}
            elif op == "ALTER_TABLE":
                if tbl in self.schemas and entry.get("schema"):
                    new_schema = TableSchema.from_dict(entry["schema"])
                    self.schemas[tbl] = new_schema
                    after_rows = entry.get("after")
                    if isinstance(after_rows, dict):
                        self.tables[tbl] = {int(k): v for k, v in after_rows.items()}
            elif op == "DROP_TABLE":
                self.schemas.pop(tbl, None)
                self.next_rids.pop(tbl, None)
                self.tables.pop(tbl, None)
                self.offsets.pop(tbl, None)
                self.indexes.pop(tbl, None)
                mdb_path = self.storage._table_file(tbl)
                if os.path.exists(mdb_path):
                    os.remove(mdb_path)
            elif op == "INSERT" and tbl in self.tables:
                rec = entry["after"]
                rid = rec["_rid"]
                self.tables[tbl][rid] = rec
                if rid >= self.next_rids.get(tbl, 1):
                    self.next_rids[tbl] = rid + 1
            elif op == "UPDATE" and tbl in self.tables:
                rec = entry["after"]
                self.tables[tbl][rec["_rid"]] = rec
            elif op == "DELETE" and tbl in self.tables:
                rec = entry["before"]
                self.tables[tbl].pop(rec["_rid"], None)

        for tbl, rec_map in self.tables.items():
            ordered_recs = [rec_map[k] for k in sorted(rec_map.keys())]
            self.offsets[tbl] = self.storage.rewrite_table(tbl, ordered_recs)

        self.storage.save_catalog(self.schemas, self.next_rids)
        self.wal.truncate()

        valid_meta = []
        for meta in self.index_meta:
            tbl = meta["table_name"]
            cols = meta.get("columns") or [meta["column_name"]]
            if tbl in self.schemas and all(c in self.schemas[tbl].col_map for c in cols):
                self._build_index_in_memory(meta["index_name"], tbl, cols, meta["index_type"])
                valid_meta.append(meta)
        if len(valid_meta) != len(self.index_meta):
            self.index_meta = valid_meta
            self._save_index_meta()

        self.last_recovery_stats = {
            "replayed_committed_ops": len(committed_ops),
            "discarded_uncommitted_txns": discarded_uncommitted,
        }

    def _save_index_meta(self) -> None:
        idx_meta_path = os.path.join(self.db_dir, "indexes.meta.json")
        tmp_path = idx_meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.index_meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        safe_replace(tmp_path, idx_meta_path)

    def _build_index_in_memory(self, index_name: str, table_name: str, columns: List[str], index_type: str) -> BaseIndex:
        idx: BaseIndex
        if index_type == "HASH":
            idx = HashIndex(index_name, table_name, columns)
        else:
            idx = BTreeRangeIndex(index_name, table_name, columns)

        for rid, rec in self.tables[table_name].items():
            idx.insert(idx.extract_key(rec), rid)

        self.indexes.setdefault(table_name, {})[tuple(columns)] = idx
        return idx

    # --- Foreign Key Helpers ---

    def _validate_foreign_key_definitions(self, table_name: str, cols: List[Column]) -> None:
        for col in cols:
            if not col.foreign_key:
                continue
            ref_tbl, ref_col, _ = col.foreign_key
            if ref_tbl == table_name:
                ref_col_obj = next((c for c in cols if c.name == ref_col), None)
            else:
                if ref_tbl not in self.schemas:
                    raise SchemaError(
                        f"Foreign key on '{table_name}.{col.name}' references non-existent table '{ref_tbl}'."
                    )
                ref_col_obj = self.schemas[ref_tbl].col_map.get(ref_col)

            if ref_col_obj is None:
                raise SchemaError(
                    f"Foreign key on '{table_name}.{col.name}' references non-existent column '{ref_tbl}.{ref_col}'."
                )
            if not ref_col_obj.unique:
                raise SchemaError(
                    f"Foreign key target '{ref_tbl}.{ref_col}' must have a UNIQUE constraint."
                )
            if ref_col_obj.data_type != col.data_type:
                raise SchemaError(
                    f"Foreign key type mismatch: '{table_name}.{col.name}' ({col.data_type}) vs '{ref_tbl}.{ref_col}' ({ref_col_obj.data_type})."
                )

    def _check_foreign_key_values(self, schema: TableSchema, record: Dict[str, Any]) -> None:
        for col in schema.columns:
            if not col.foreign_key:
                continue
            val = record.get(col.name)
            if val is None:
                continue
            ref_tbl, ref_col, _ = col.foreign_key
            if ref_tbl not in self.tables:
                raise SchemaError(f"Referenced parent table '{ref_tbl}' does not exist.")

            ref_idx = self.indexes.get(ref_tbl, {}).get((ref_col,))
            if ref_idx is not None:
                matches = ref_idx.lookup("=", val) or set()
                exists = any(rid in self.tables[ref_tbl] for rid in matches)
            else:
                exists = any(r.get(ref_col) == val for r in self.tables[ref_tbl].values())

            if not exists:
                raise SchemaError(
                    f"Foreign key constraint failed on '{schema.table_name}.{col.name}': value {val!r} not found in '{ref_tbl}.{ref_col}'."
                )

    def _get_referencing_columns(self, parent_table: str) -> List[Tuple[str, Column]]:
        refs: List[Tuple[str, Column]] = []
        for child_tbl, child_schema in self.schemas.items():
            for col in child_schema.columns:
                if col.foreign_key and col.foreign_key[0] == parent_table:
                    refs.append((child_tbl, col))
        return refs

    # --- Transaction Management (NO-STEAL / FORCE WAL Protocol) ---

    def begin(self) -> str:
        self._ensure_open()
        if self._get_tx_state() is not None:
            raise TransactionError("Nested transactions are not supported.")
        tx_id = str(uuid.uuid4())
        self.wal.append(tx_id=tx_id, op="BEGIN")
        self._set_tx_state({
            "tx_id": tx_id,
            "undo_log": [],
            "disk_writes": [],
            "catalog_changed": False,
        })
        return tx_id

    def commit(self) -> None:
        tx = self._get_tx_state()
        if tx is None:
            raise TransactionError("No active transaction to commit.")

        self.wal.append(tx_id=tx["tx_id"], op="COMMIT")

        for disk_op, tbl, rid, rec in tx["disk_writes"]:
            if disk_op == "APPEND":
                offset = self.storage.append_record(tbl, rec)
                self.offsets[tbl][rid] = offset
            elif disk_op == "DELETE":
                old_offset = self.offsets[tbl].pop(rid, None)
                if old_offset is not None:
                    self.storage.mark_deleted(tbl, old_offset)
            elif disk_op == "REWRITE_TABLE":
                rec_map = self.tables[tbl]
                ordered_recs = [rec_map[k] for k in sorted(rec_map.keys())]
                self.offsets[tbl] = self.storage.rewrite_table(tbl, ordered_recs)
            elif disk_op == "REMOVE_TABLE_FILE":
                mdb_path = self.storage._table_file(tbl)
                if os.path.exists(mdb_path):
                    os.remove(mdb_path)

        if tx["catalog_changed"]:
            self.storage.save_catalog(self.schemas, self.next_rids)

        self._set_tx_state(None)

    def rollback(self) -> None:
        tx = self._get_tx_state()
        if tx is None:
            raise TransactionError("No active transaction to rollback.")

        for action, tbl, rid, old_rec in reversed(tx["undo_log"]):
            if action == "UNDO_INSERT":
                rec = self.tables[tbl].pop(rid, None)
                if rec:
                    for idx in self.indexes.get(tbl, {}).values():
                        idx.remove(idx.extract_key(rec), rid)

            elif action == "UNDO_UPDATE":
                curr_rec = self.tables[tbl].get(rid)
                if curr_rec:
                    for idx in self.indexes.get(tbl, {}).values():
                        idx.remove(idx.extract_key(curr_rec), rid)
                self.tables[tbl][rid] = old_rec
                for idx in self.indexes.get(tbl, {}).values():
                    idx.insert(idx.extract_key(old_rec), rid)

            elif action == "UNDO_DELETE":
                self.tables[tbl][rid] = old_rec
                for idx in self.indexes.get(tbl, {}).values():
                    idx.insert(idx.extract_key(old_rec), rid)

            elif action == "UNDO_CREATE_TABLE":
                self.schemas.pop(tbl, None)
                self.tables.pop(tbl, None)
                self.offsets.pop(tbl, None)
                self.indexes.pop(tbl, None)
                self.next_rids.pop(tbl, None)

        self.wal.append(tx_id=tx["tx_id"], op="ROLLBACK")
        self._set_tx_state(None)

    @contextmanager
    def transaction(self):
        self.begin()
        try:
            yield self
            self.commit()
        except Exception:
            if self._get_tx_state() is not None:
                self.rollback()
            raise

    # --- Developer Ergonomics & Convenience APIs ---

    def query(self, sql: str, params: Any = None) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns the list of row dicts directly."""
        res = self.execute(sql, params=params)
        if "rows" not in res:
            raise ExecutionError("query() can only be used with SELECT statements.")
        return res["rows"]

    def fetchall(self, sql: str, params: Any = None) -> List[Dict[str, Any]]:
        """Alias for query(): returns all matching rows as a list of dicts."""
        return self.query(sql, params=params)

    def fetchone(self, sql: str, params: Any = None) -> Optional[Dict[str, Any]]:
        """Executes a SELECT query and returns the first matching row dict, or None."""
        rows = self.query(sql, params=params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Any = None) -> Any:
        """Executes a SELECT query and returns the first column of the first row, or None."""
        row = self.fetchone(sql, params=params)
        if not row:
            return None
        return next(iter(row.values()))

    def executemany(self, sql: str, param_seq: Iterable[Any]) -> Dict[str, Any]:
        """Executes a parameterized DML query across many parameter sets in one atomic transaction."""
        self._ensure_open()
        start_time = time.perf_counter()
        total_affected = 0
        batch_count = 0

        with self.rwlock.write_lock():
            auto_tx = self._get_tx_state() is None
            if auto_tx:
                self.begin()
            try:
                for p in param_seq:
                    res = self.execute(sql, params=p)
                    total_affected += res.get("rows_affected", 0)
                    batch_count += 1
                if auto_tx:
                    self.commit()
            except Exception:
                if auto_tx and self._get_tx_state() is not None:
                    self.rollback()
                raise

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 4)
        return {
            "status": "OK",
            "batches": batch_count,
            "rows_affected": total_affected,
            "stats": {"execution_time_ms": elapsed_ms},
        }

    def backup(self, dest_dir: str, overwrite: bool = False) -> Dict[str, Any]:
        """Creates a consistent point-in-time snapshot of all catalog, index, and .mdb table files."""
        self._ensure_open()
        if self._get_tx_state() is not None:
            raise TransactionError("Cannot create a backup while a transaction is active.")

        if os.path.exists(dest_dir):
            if not overwrite:
                raise ExecutionError(f"Backup destination '{dest_dir}' already exists (pass overwrite=True to replace).")
            shutil.rmtree(dest_dir)

        with self.rwlock.read_lock():
            os.makedirs(dest_dir, exist_ok=True)
            copied_files: List[str] = []
            for fname in os.listdir(self.db_dir):
                if fname.endswith(".mdb") or fname in ("catalog.meta.json", "indexes.meta.json"):
                    src = os.path.join(self.db_dir, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(dest_dir, fname))
                        copied_files.append(fname)

        return {
            "status": "OK",
            "backup_dir": dest_dir,
            "files_copied": sorted(copied_files),
        }

    # --- Query Execution Entry Point ---

    def execute(self, query: str, params: Any = None) -> Dict[str, Any]:
        self._ensure_open()
        start_time = time.perf_counter()
        stmt = parse_query(query, params=params)

        if isinstance(stmt, SelectStmt):
            with self.rwlock.read_lock():
                result = self._exec_select(stmt)
        elif isinstance(stmt, ExplainStmt):
            with self.rwlock.read_lock():
                result = self._exec_explain(stmt)
        elif isinstance(stmt, ShowTablesStmt):
            with self.rwlock.read_lock():
                result = self._exec_show_tables()
        elif isinstance(stmt, DescribeTableStmt):
            with self.rwlock.read_lock():
                result = self._exec_describe_table(stmt)
        elif isinstance(stmt, VacuumStmt):
            with self.rwlock.write_lock():
                result = self._exec_vacuum(stmt)
        else:
            with self.rwlock.write_lock():
                auto_tx = self._get_tx_state() is None
                if auto_tx:
                    self.begin()
                try:
                    if isinstance(stmt, CreateTableStmt):
                        result = self._exec_create_table(stmt)
                    elif isinstance(stmt, AlterTableStmt):
                        result = self._exec_alter_table(stmt)
                    elif isinstance(stmt, DropTableStmt):
                        result = self._exec_drop_table(stmt)
                    elif isinstance(stmt, CreateIndexStmt):
                        result = self._exec_create_index(stmt)
                    elif isinstance(stmt, DropIndexStmt):
                        result = self._exec_drop_index(stmt)
                    elif isinstance(stmt, InsertStmt):
                        result = self._exec_insert(stmt)
                    elif isinstance(stmt, UpdateStmt):
                        result = self._exec_update(stmt)
                    elif isinstance(stmt, DeleteStmt):
                        result = self._exec_delete(stmt)
                    else:
                        raise ExecutionError(f"Unsupported AST node: {type(stmt).__name__}")

                    if auto_tx:
                        self.commit()
                except Exception:
                    if auto_tx and self._get_tx_state() is not None:
                        self.rollback()
                    raise

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 4)
        result.setdefault("stats", {})["execution_time_ms"] = elapsed_ms
        return result

    # --- Introspection, Utility & Maintenance Executors ---

    def _exec_show_tables(self) -> Dict[str, Any]:
        tables_info = [
            {
                "table": tbl,
                "rows": len(self.tables[tbl]),
                "columns": len(self.schemas[tbl].columns),
                "indexes": len(self.indexes.get(tbl, {})),
            }
            for tbl in sorted(self.schemas.keys())
        ]
        return {"status": "OK", "tables": tables_info}

    def _exec_describe_table(self, stmt: DescribeTableStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        cols_info = []
        for c in schema.columns:
            col_entry: Dict[str, Any] = {
                "name": c.name,
                "type": getattr(c, "data_type", getattr(c, "type", "STR")),
                "not_null": c.not_null,
                "unique": c.unique,
            }
            if c.foreign_key:
                col_entry["references"] = f"{c.foreign_key[0]}({c.foreign_key[1]}) ON DELETE {c.foreign_key[2]}"
            cols_info.append(col_entry)

        idx_info = [
            {
                "index_name": idx.index_name,
                "columns": idx.columns,
                "type": idx.index_type,
            }
            for idx in self.indexes.get(stmt.table_name, {}).values()
        ]
        return {
            "status": "OK",
            "table": stmt.table_name,
            "row_count": len(self.tables[stmt.table_name]),
            "columns": cols_info,
            "indexes": idx_info,
        }

    def _exec_explain(self, stmt: ExplainStmt) -> Dict[str, Any]:
        sel = stmt.select_stmt
        if sel.table_name not in self.schemas:
            raise ExecutionError(f"Table '{sel.table_name}' does not exist.")
        if sel.joins:
            join_steps = []
            for j in sel.joins:
                if j.right_table not in self.schemas:
                    raise ExecutionError(f"Joined table '{j.right_table}' does not exist.")
                r_col = j.right_col.split(".")[-1]
                has_right_idx = (r_col,) in self.indexes.get(j.right_table, {})
                algo = "INDEX_NESTED_LOOP_JOIN" if has_right_idx else "HASH_JOIN"
                join_steps.append({
                    "join_type": j.join_type,
                    "right_table": j.right_table,
                    "condition": f"{j.left_col} = {j.right_col}",
                    "algorithm": algo,
                })
            return {
                "status": "OK",
                "plan": {
                    "strategy": "MULTI_STAGE_JOIN",
                    "base_table": sel.table_name,
                    "join_pipeline": join_steps,
                },
            }

        table_map = self.tables[sel.table_name]
        table_indexes = self.indexes.get(sel.table_name, {})
        plan = QueryPlanner.choose_plan(len(table_map), sel.where, table_indexes)
        est_scan = len(plan["candidate_rids"]) if plan["candidate_rids"] is not None else len(table_map)
        return {
            "status": "OK",
            "plan": {
                "strategy": plan["strategy"],
                "index_used": plan["index_used"],
                "estimated_records_to_scan": est_scan,
                "total_table_records": len(table_map),
                "reason": plan["reason"],
            },
        }

    def _exec_vacuum(self, stmt: VacuumStmt) -> Dict[str, Any]:
        if self._get_tx_state() is not None:
            raise TransactionError("VACUUM cannot run inside an active transaction.")
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")

        mdb_path = self.storage._table_file(stmt.table_name)
        bytes_before = os.path.getsize(mdb_path) if os.path.exists(mdb_path) else 0

        rec_map = self.tables[stmt.table_name]
        ordered_recs = [rec_map[k] for k in sorted(rec_map.keys())]
        self.offsets[stmt.table_name] = self.storage.rewrite_table(stmt.table_name, ordered_recs)

        bytes_after = os.path.getsize(mdb_path) if os.path.exists(mdb_path) else 0
        return {
            "status": "OK",
            "message": f"Compacted table '{stmt.table_name}'.",
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "bytes_reclaimed": max(0, bytes_before - bytes_after),
        }

    # --- DDL & DML Executors ---

    def _exec_create_table(self, stmt: CreateTableStmt) -> Dict[str, Any]:
        if stmt.table_name in self.schemas:
            raise SchemaError(f"Table '{stmt.table_name}' already exists.")

        cols = [Column(name, dtype, nn, uq, fk) for name, dtype, nn, uq, fk in stmt.columns]
        self._validate_foreign_key_definitions(stmt.table_name, cols)
        schema = TableSchema(stmt.table_name, cols)

        tx = self._get_tx_state()
        assert tx is not None
        self.wal.append(tx_id=tx["tx_id"], op="CREATE_TABLE", table=stmt.table_name, schema=schema.to_dict())

        self.schemas[stmt.table_name] = schema
        self.next_rids[stmt.table_name] = 1
        self.tables[stmt.table_name] = {}
        self.offsets[stmt.table_name] = {}
        self.indexes[stmt.table_name] = {}

        tx["undo_log"].append(("UNDO_CREATE_TABLE", stmt.table_name, -1, None))
        tx["catalog_changed"] = True

        for col in cols:
            if col.unique:
                idx_name = f"uq_{stmt.table_name}_{col.name}"
                self._build_index_in_memory(idx_name, stmt.table_name, [col.name], "HASH")
                self.index_meta.append({
                    "index_name": idx_name,
                    "table_name": stmt.table_name,
                    "columns": [col.name],
                    "index_type": "HASH",
                })
                self._save_index_meta()

        return {"status": "OK", "message": f"Table '{stmt.table_name}' created."}

    def _exec_alter_table(self, stmt: AlterTableStmt) -> Dict[str, Any]:
        if self._get_tx_state() is not None and len(self._get_tx_state()["undo_log"]) > 0:
            raise TransactionError("ALTER TABLE cannot be combined with other operations inside an active transaction.")
        tbl = stmt.table_name
        if tbl not in self.schemas:
            raise ExecutionError(f"Table '{tbl}' does not exist.")

        old_schema = self.schemas[tbl]
        tx = self._get_tx_state()
        assert tx is not None

        if stmt.action == "ADD_COLUMN":
            assert stmt.column_def is not None
            col_name, dtype, not_null, unique, foreign_key = stmt.column_def
            if col_name in old_schema.col_map:
                raise SchemaError(f"Column '{col_name}' already exists in table '{tbl}'.")

            new_col = Column(col_name, dtype, not_null, unique, foreign_key)
            all_cols = list(old_schema.columns) + [new_col]
            self._validate_foreign_key_definitions(tbl, all_cols)
            new_schema = TableSchema(tbl, all_cols)

            fill_val = stmt.default_value if stmt.has_default else None
            if fill_val is None and (not_null or unique) and len(self.tables[tbl]) > 0:
                raise SchemaError(
                    f"Cannot add NOT NULL or UNIQUE column '{col_name}' to non-empty table '{tbl}' without DEFAULT."
                )

            updated_rows: Dict[int, Dict[str, Any]] = {}
            validated_list: List[Dict[str, Any]] = []
            for rid in sorted(self.tables[tbl].keys()):
                row_copy = copy.deepcopy(self.tables[tbl][rid])
                row_copy[col_name] = fill_val
                clean = new_schema.validate_and_coerce(row_copy, validated_list)
                self._check_foreign_key_values(new_schema, clean)
                clean["_rid"] = rid
                validated_list.append(clean)
                updated_rows[rid] = clean

            self.wal.append(
                tx_id=tx["tx_id"],
                op="ALTER_TABLE",
                table=tbl,
                schema=new_schema.to_dict(),
                after={str(k): v for k, v in updated_rows.items()},
            )

            self.schemas[tbl] = new_schema
            self.tables[tbl] = updated_rows

            if unique:
                idx_name = f"uq_{tbl}_{col_name}"
                self._build_index_in_memory(idx_name, tbl, [col_name], "HASH")
                self.index_meta.append({
                    "index_name": idx_name,
                    "table_name": tbl,
                    "columns": [col_name],
                    "index_type": "HASH",
                })
                self._save_index_meta()

            tx["disk_writes"].append(("REWRITE_TABLE", tbl, -1, None))
            tx["catalog_changed"] = True
            return {"status": "OK", "message": f"Added column '{col_name}' to '{tbl}'."}

        elif stmt.action == "DROP_COLUMN":
            col_name = stmt.column_name
            assert col_name is not None
            if col_name not in old_schema.col_map:
                raise ExecutionError(f"Column '{col_name}' does not exist in table '{tbl}'.")
            if len(old_schema.columns) <= 1:
                raise SchemaError(f"Cannot drop the only remaining column '{col_name}' from table '{tbl}'.")

            for child_tbl, child_col in self._get_referencing_columns(tbl):
                if child_col.foreign_key and child_col.foreign_key[1] == col_name:
                    raise SchemaError(
                        f"Cannot drop column '{tbl}.{col_name}': referenced by foreign key '{child_tbl}.{child_col.name}'."
                    )

            remaining_cols = [c for c in old_schema.columns if c.name != col_name]
            new_schema = TableSchema(tbl, remaining_cols)

            updated_rows = {}
            for rid, row in self.tables[tbl].items():
                row_copy = copy.deepcopy(row)
                row_copy.pop(col_name, None)
                updated_rows[rid] = row_copy

            self.wal.append(
                tx_id=tx["tx_id"],
                op="ALTER_TABLE",
                table=tbl,
                schema=new_schema.to_dict(),
                after={str(k): v for k, v in updated_rows.items()},
            )

            self.schemas[tbl] = new_schema
            self.tables[tbl] = updated_rows

            for cols_key in list(self.indexes.get(tbl, {}).keys()):
                if col_name in cols_key:
                    del self.indexes[tbl][cols_key]
            new_meta = [
                m for m in self.index_meta
                if not (m["table_name"] == tbl and col_name in (m.get("columns") or [m["column_name"]]))
            ]
            if len(new_meta) != len(self.index_meta):
                self.index_meta = new_meta
                self._save_index_meta()

            tx["disk_writes"].append(("REWRITE_TABLE", tbl, -1, None))
            tx["catalog_changed"] = True
            return {"status": "OK", "message": f"Dropped column '{col_name}' from '{tbl}'."}

        elif stmt.action == "RENAME_COLUMN":
            old_col = stmt.column_name
            new_col = stmt.new_column_name
            assert old_col is not None and new_col is not None
            if old_col not in old_schema.col_map:
                raise ExecutionError(f"Column '{old_col}' does not exist in table '{tbl}'.")
            if new_col in old_schema.col_map:
                raise SchemaError(f"Column '{new_col}' already exists in table '{tbl}'.")

            for child_tbl, child_col in self._get_referencing_columns(tbl):
                if child_col.foreign_key and child_col.foreign_key[1] == old_col:
                    raise SchemaError(
                        f"Cannot rename column '{tbl}.{old_col}': referenced by foreign key '{child_tbl}.{child_col.name}'."
                    )

            new_cols = []
            for c in old_schema.columns:
                if c.name == old_col:
                    new_cols.append(Column(new_col, c.data_type, c.not_null, c.unique, c.foreign_key))
                else:
                    new_cols.append(c)
            new_schema = TableSchema(tbl, new_cols)

            updated_rows = {}
            for rid, row in self.tables[tbl].items():
                row_copy = copy.deepcopy(row)
                row_copy[new_col] = row_copy.pop(old_col, None)
                updated_rows[rid] = row_copy

            self.wal.append(
                tx_id=tx["tx_id"],
                op="ALTER_TABLE",
                table=tbl,
                schema=new_schema.to_dict(),
                after={str(k): v for k, v in updated_rows.items()},
            )

            self.schemas[tbl] = new_schema
            self.tables[tbl] = updated_rows

            meta_changed = False
            for m in self.index_meta:
                if m["table_name"] == tbl:
                    cols_list = m.get("columns") or [m["column_name"]]
                    if old_col in cols_list:
                        old_key = tuple(cols_list)
                        self.indexes.get(tbl, {}).pop(old_key, None)
                        renamed_cols = [new_col if c == old_col else c for c in cols_list]
                        m["columns"] = renamed_cols
                        m.pop("column_name", None)
                        self._build_index_in_memory(m["index_name"], tbl, renamed_cols, m["index_type"])
                        meta_changed = True
            if meta_changed:
                self._save_index_meta()

            tx["disk_writes"].append(("REWRITE_TABLE", tbl, -1, None))
            tx["catalog_changed"] = True
            return {"status": "OK", "message": f"Renamed column '{old_col}' to '{new_col}' in '{tbl}'."}

        raise ExecutionError(f"Unsupported ALTER TABLE action '{stmt.action}'.")

    def _exec_drop_table(self, stmt: DropTableStmt) -> Dict[str, Any]:
        if self._get_tx_state() is not None and len(self._get_tx_state()["undo_log"]) > 0:
            raise TransactionError("DROP TABLE cannot be combined with other operations inside an active transaction.")
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")

        refs = [
            (child_tbl, col.name)
            for child_tbl, col in self._get_referencing_columns(stmt.table_name)
            if child_tbl != stmt.table_name
        ]
        if refs:
            child_tbl, child_col = refs[0]
            raise SchemaError(
                f"Cannot drop table '{stmt.table_name}': referenced by foreign key '{child_tbl}.{child_col}'."
            )

        tx = self._get_tx_state()
        assert tx is not None
        self.wal.append(tx_id=tx["tx_id"], op="DROP_TABLE", table=stmt.table_name)

        self.schemas.pop(stmt.table_name, None)
        self.next_rids.pop(stmt.table_name, None)
        self.tables.pop(stmt.table_name, None)
        self.offsets.pop(stmt.table_name, None)
        self.indexes.pop(stmt.table_name, None)

        self.index_meta = [m for m in self.index_meta if m["table_name"] != stmt.table_name]
        self._save_index_meta()

        tx["disk_writes"].append(("REMOVE_TABLE_FILE", stmt.table_name, -1, None))
        tx["catalog_changed"] = True
        return {"status": "OK", "message": f"Table '{stmt.table_name}' dropped."}

    def _exec_create_index(self, stmt: CreateIndexStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        for col in stmt.columns:
            if col not in schema.col_map:
                raise ExecutionError(f"Column '{col}' does not exist in table '{stmt.table_name}'.")

        col_tuple = tuple(stmt.columns)
        self._build_index_in_memory(stmt.index_name, stmt.table_name, stmt.columns, stmt.index_type)
        self.index_meta = [
            m for m in self.index_meta
            if not (
                m["table_name"] == stmt.table_name
                and tuple(m.get("columns") or [m["column_name"]]) == col_tuple
            )
        ]
        self.index_meta.append({
            "index_name": stmt.index_name,
            "table_name": stmt.table_name,
            "columns": list(stmt.columns),
            "index_type": stmt.index_type,
        })
        self._save_index_meta()

        return {
            "status": "OK",
            "message": f"Index '{stmt.index_name}' ({stmt.index_type}) created on {stmt.table_name}({', '.join(stmt.columns)}).",
        }

    def _exec_drop_index(self, stmt: DropIndexStmt) -> Dict[str, Any]:
        found = False
        for tbl, idx_map in list(self.indexes.items()):
            for cols_key, idx_obj in list(idx_map.items()):
                if idx_obj.index_name == stmt.index_name:
                    del idx_map[cols_key]
                    found = True
        if not found:
            raise ExecutionError(f"Index '{stmt.index_name}' does not exist.")

        self.index_meta = [m for m in self.index_meta if m["index_name"] != stmt.index_name]
        self._save_index_meta()
        return {"status": "OK", "message": f"Index '{stmt.index_name}' dropped."}

    def _exec_insert(self, stmt: InsertStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]

        if stmt.columns is not None:
            if len(stmt.columns) != len(stmt.values):
                raise SchemaError("Column count does not match value count in INSERT.")
            raw_rec = dict(zip(stmt.columns, stmt.values))
        else:
            if len(stmt.values) != len(schema.columns):
                raise SchemaError(
                    f"Table '{stmt.table_name}' has {len(schema.columns)} columns, but {len(stmt.values)} values were provided."
                )
            raw_rec = {col.name: val for col, val in zip(schema.columns, stmt.values)}

        existing = list(self.tables[stmt.table_name].values())
        clean_rec = schema.validate_and_coerce(raw_rec, existing)
        self._check_foreign_key_values(schema, clean_rec)

        rid = self.next_rids[stmt.table_name]
        self.next_rids[stmt.table_name] = rid + 1
        clean_rec["_rid"] = rid

        tx = self._get_tx_state()
        assert tx is not None
        self.wal.append(tx_id=tx["tx_id"], op="INSERT", table=stmt.table_name, after=clean_rec)

        self.tables[stmt.table_name][rid] = clean_rec
        for idx in self.indexes.get(stmt.table_name, {}).values():
            idx.insert(idx.extract_key(clean_rec), rid)

        tx["disk_writes"].append(("APPEND", stmt.table_name, rid, clean_rec))
        tx["undo_log"].append(("UNDO_INSERT", stmt.table_name, rid, None))
        tx["catalog_changed"] = True

        return {"status": "OK", "rows_affected": 1, "rid": rid}

    def _compare_values(self, lhs: Any, op: str, rhs: Any) -> bool:
        if lhs is None or rhs is None:
            if op == "=":
                return lhs is None and rhs is None
            if op == "!=":
                return not (lhs is None and rhs is None)
            return False

        if op == "=":
            return lhs == rhs
        elif op == "!=":
            return lhs != rhs
        elif op == "<":
            return lhs < rhs
        elif op == "<=":
            return lhs <= rhs
        elif op == ">":
            return lhs > rhs
        elif op == ">=":
            return lhs >= rhs
        raise ExecutionError(f"Unsupported operator '{op}'.")

    def _evaluate_condition(self, row: Dict[str, Any], cond: Condition, valid_fields: set) -> bool:
        if cond.field not in valid_fields:
            raise ExecutionError(f"Unknown column '{cond.field}' in WHERE clause.")
        lhs = row.get(cond.field)
        rhs = cond.value
        op = cond.op

        if op == "IN":
            return lhs in rhs
        if op == "LIKE":
            if not isinstance(lhs, str):
                return False
            glob_pat = rhs.replace("%", "*").replace("_", "?")
            return fnmatch.fnmatchcase(lhs, glob_pat)

        return self._compare_values(lhs, op, rhs)

    def _matches_where(self, row: Dict[str, Any], where: Optional[WhereClause], valid_fields: set) -> bool:
        if where is None or not where.conditions:
            return True

        result = self._evaluate_condition(row, where.conditions[0], valid_fields)
        for i, conn in enumerate(where.connectors):
            next_val = self._evaluate_condition(row, where.conditions[i + 1], valid_fields)
            if conn == "AND":
                result = result and next_val
            elif conn == "OR":
                result = result or next_val
        return result

    def _select_candidate_rows(self, table_name: str, where: Optional[WhereClause]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        table_map = self.tables[table_name]
        table_indexes = self.indexes.get(table_name, {})
        plan = QueryPlanner.choose_plan(len(table_map), where, table_indexes)

        if plan["strategy"] in ("INDEX_SCAN", "INDEX_UNION_SCAN") and plan["candidate_rids"] is not None:
            candidates = [table_map[rid] for rid in plan["candidate_rids"] if rid in table_map]
        else:
            candidates = list(table_map.values())

        stats = {
            "strategy": plan["strategy"],
            "index_used": plan["index_used"],
            "planner_reason": plan["reason"],
            "records_scanned": len(candidates),
            "total_table_records": len(table_map),
        }
        return candidates, stats

    def _exec_update(self, stmt: UpdateStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        valid_fields = set(schema.col_map.keys())
        candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)

        matched_rows = [r for r in candidates if self._matches_where(r, stmt.where, valid_fields)]
        tx = self._get_tx_state()
        assert tx is not None

        referencing_cols = self._get_referencing_columns(stmt.table_name)
        updated_count = 0

        for old_row in matched_rows:
            rid = old_row["_rid"]
            candidate_copy = copy.deepcopy(old_row)
            for col_name, new_val in stmt.assignments:
                if col_name not in schema.col_map:
                    raise ExecutionError(f"Unknown column '{col_name}' in UPDATE.")
                candidate_copy[col_name] = new_val

            existing_others = list(self.tables[stmt.table_name].values())
            validated = schema.validate_and_coerce(candidate_copy, existing_others, ignore_row_id=rid)
            self._check_foreign_key_values(schema, validated)
            validated["_rid"] = rid

            for child_tbl, child_col in referencing_cols:
                assert child_col.foreign_key is not None
                ref_col = child_col.foreign_key[1]
                if old_row.get(ref_col) != validated.get(ref_col):
                    in_use = any(
                        cr.get(child_col.name) == old_row.get(ref_col)
                        for cr in self.tables.get(child_tbl, {}).values()
                        if not (child_tbl == stmt.table_name and cr.get("_rid") == rid)
                    )
                    if in_use:
                        raise SchemaError(
                            f"Cannot update '{stmt.table_name}.{ref_col}': value {old_row.get(ref_col)!r} is referenced by '{child_tbl}.{child_col.name}'."
                        )

            self.wal.append(tx_id=tx["tx_id"], op="UPDATE", table=stmt.table_name, before=old_row, after=validated)

            self.tables[stmt.table_name][rid] = validated
            for idx in self.indexes.get(stmt.table_name, {}).values():
                idx.remove(idx.extract_key(old_row), rid)
                idx.insert(idx.extract_key(validated), rid)

            tx["disk_writes"].append(("DELETE", stmt.table_name, rid, None))
            tx["disk_writes"].append(("APPEND", stmt.table_name, rid, validated))
            tx["undo_log"].append(("UNDO_UPDATE", stmt.table_name, rid, copy.deepcopy(old_row)))
            updated_count += 1

        return {"status": "OK", "rows_affected": updated_count, "stats": scan_stats}

    def _delete_rows_with_fk(self, table_name: str, rows_to_delete: List[Dict[str, Any]], tx: Dict[str, Any], visited: Optional[set] = None) -> int:
        if visited is None:
            visited = set()

        referencing_cols = self._get_referencing_columns(table_name)
        deleted_total = 0

        for old_row in rows_to_delete:
            rid = old_row["_rid"]
            state_key = (table_name, rid)
            if state_key in visited or rid not in self.tables.get(table_name, {}):
                continue
            visited.add(state_key)

            for child_tbl, child_col in referencing_cols:
                assert child_col.foreign_key is not None
                _, ref_col, on_delete = child_col.foreign_key
                parent_val = old_row.get(ref_col)
                if parent_val is None:
                    continue

                child_matches = [
                    cr for cr in list(self.tables.get(child_tbl, {}).values())
                    if cr.get(child_col.name) == parent_val
                    and not (child_tbl == table_name and cr.get("_rid") == rid)
                ]
                if child_matches:
                    if on_delete == "CASCADE":
                        deleted_total += self._delete_rows_with_fk(child_tbl, child_matches, tx, visited)
                    else:
                        raise SchemaError(
                            f"Cannot delete from '{table_name}': row with {ref_col}={parent_val!r} is referenced by '{child_tbl}.{child_col.name}'."
                        )

            self.wal.append(tx_id=tx["tx_id"], op="DELETE", table=table_name, before=old_row)
            self.tables[table_name].pop(rid, None)
            for idx in self.indexes.get(table_name, {}).values():
                idx.remove(idx.extract_key(old_row), rid)

            tx["disk_writes"].append(("DELETE", table_name, rid, None))
            tx["undo_log"].append(("UNDO_DELETE", table_name, rid, copy.deepcopy(old_row)))
            deleted_total += 1

        return deleted_total

    def _exec_delete(self, stmt: DeleteStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        valid_fields = set(schema.col_map.keys())
        candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)

        matched_rows = [r for r in candidates if self._matches_where(r, stmt.where, valid_fields)]
        tx = self._get_tx_state()
        assert tx is not None

        deleted_count = self._delete_rows_with_fk(stmt.table_name, matched_rows, tx)
        return {"status": "OK", "rows_affected": deleted_count, "stats": scan_stats}

    def _compute_agg_values(self, func: str, field: str, rows: List[Dict[str, Any]], valid_fields: set) -> Any:
        if func == "COUNT":
            if field == "*":
                return len(rows)
            if field not in valid_fields:
                raise ExecutionError(f"Unknown column '{field}' in COUNT().")
            return sum(1 for r in rows if r.get(field) is not None)

        if field not in valid_fields:
            raise ExecutionError(f"Unknown column '{field}' in {func}().")

        vals = [r[field] for r in rows if r.get(field) is not None]
        if not vals:
            return 0 if func == "SUM" else None

        if func == "SUM":
            return sum(vals)
        elif func == "AVG":
            return round(sum(vals) / len(vals), 4)
        elif func == "MIN":
            return min(vals)
        elif func == "MAX":
            return max(vals)
        raise ExecutionError(f"Unsupported aggregation function '{func}'.")

    def _compute_agg(self, item: SelectItem, rows: List[Dict[str, Any]], valid_fields: set) -> Any:
        assert item.func is not None and item.field is not None
        return self._compute_agg_values(item.func, item.field, rows, valid_fields)

    def _evaluate_having_cond(
        self,
        cond: HavingCondition,
        grp_key: Any,
        grp_rows: List[Dict[str, Any]],
        out_row: Dict[str, Any],
        group_by_col: Optional[str],
        valid_fields: set,
    ) -> bool:
        if cond.target_type == "AGG":
            assert cond.func is not None
            lhs = self._compute_agg_values(cond.func, cond.field, grp_rows, valid_fields)
        else:
            if cond.field in out_row:
                lhs = out_row[cond.field]
            elif group_by_col is not None and cond.field == group_by_col:
                lhs = grp_key
            else:
                raise ExecutionError(
                    f"Column '{cond.field}' in HAVING must be in GROUP BY, an alias, or inside an aggregate function."
                )
        return self._compare_values(lhs, cond.op, cond.value)

    def _matches_having(
        self,
        having: Optional[HavingClause],
        grp_key: Any,
        grp_rows: List[Dict[str, Any]],
        out_row: Dict[str, Any],
        group_by_col: Optional[str],
        valid_fields: set,
    ) -> bool:
        if having is None or not having.conditions:
            return True

        result = self._evaluate_having_cond(having.conditions[0], grp_key, grp_rows, out_row, group_by_col, valid_fields)
        for i, conn in enumerate(having.connectors):
            next_val = self._evaluate_having_cond(
                having.conditions[i + 1], grp_key, grp_rows, out_row, group_by_col, valid_fields
            )
            if conn == "AND":
                result = result and next_val
            elif conn == "OR":
                result = result or next_val
        return result

    def _build_joined_rows(self, stmt: SelectStmt) -> Tuple[List[Dict[str, Any]], List[str], set, Dict[str, Any]]:
        base_tbl = stmt.table_name
        if base_tbl not in self.schemas:
            raise ExecutionError(f"Table '{base_tbl}' does not exist.")

        base_schema = self.schemas[base_tbl]
        current_rows: List[Dict[str, Any]] = []
        for r in self.tables[base_tbl].values():
            row_dict = {f"{base_tbl}.{c.name}": r.get(c.name) for c in base_schema.columns}
            current_rows.append(row_dict)

        included_tables = [base_tbl]
        records_scanned = len(current_rows)
        total_records = len(current_rows)
        strategies_used: List[str] = []
        indexes_used: List[str] = []

        for j in stmt.joins:
            right_tbl = j.right_table
            if right_tbl not in self.schemas:
                raise ExecutionError(f"Table '{right_tbl}' does not exist.")
            right_schema = self.schemas[right_tbl]
            r_col = j.right_col.split(".")[-1]
            if r_col not in right_schema.col_map:
                raise ExecutionError(f"Unknown join column '{j.right_col}'.")

            right_map = self.tables[right_tbl]
            total_records += len(right_map)
            right_idx = self.indexes.get(right_tbl, {}).get((r_col,))
            null_right_template = {f"{right_tbl}.{c.name}": None for c in right_schema.columns}

            next_rows: List[Dict[str, Any]] = []

            if right_idx is not None:
                strategies_used.append(("LEFT_" if j.join_type == "LEFT" else "") + "INDEX_NESTED_LOOP_JOIN")
                indexes_used.append(f"{right_idx.index_name}({right_tbl}.{r_col})")
                for lrow in current_rows:
                    l_val = lrow.get(j.left_col) if "." in j.left_col else self._resolve_unqualified(lrow, j.left_col, included_tables)
                    matched_rids = (right_idx.lookup("=", l_val) or set()) if l_val is not None else set()
                    records_scanned += len(matched_rids)
                    if matched_rids:
                        for rrid in matched_rids:
                            rrow = right_map.get(rrid)
                            if rrow is not None:
                                merged = dict(lrow)
                                for c in right_schema.columns:
                                    merged[f"{right_tbl}.{c.name}"] = rrow.get(c.name)
                                next_rows.append(merged)
                    elif j.join_type == "LEFT":
                        merged = dict(lrow)
                        merged.update(null_right_template)
                        next_rows.append(merged)
            else:
                strategies_used.append(("LEFT_" if j.join_type == "LEFT" else "") + "HASH_JOIN")
                hash_buckets: Dict[Any, List[Dict[str, Any]]] = {}
                for rrow in right_map.values():
                    records_scanned += 1
                    kv = rrow.get(r_col)
                    if kv is not None:
                        hash_buckets.setdefault(kv, []).append(rrow)

                for lrow in current_rows:
                    l_val = lrow.get(j.left_col) if "." in j.left_col else self._resolve_unqualified(lrow, j.left_col, included_tables)
                    matches = hash_buckets.get(l_val, []) if l_val is not None else []
                    if matches:
                        for rrow in matches:
                            merged = dict(lrow)
                            for c in right_schema.columns:
                                merged[f"{right_tbl}.{c.name}"] = rrow.get(c.name)
                            next_rows.append(merged)
                    elif j.join_type == "LEFT":
                        merged = dict(lrow)
                        merged.update(null_right_template)
                        next_rows.append(merged)

            current_rows = next_rows
            included_tables.append(right_tbl)

        col_counts: Dict[str, int] = {}
        for tbl in included_tables:
            for c in self.schemas[tbl].columns:
                col_counts[c.name] = col_counts.get(c.name, 0) + 1

        all_star_cols: List[str] = []
        valid_fields: set = set()
        for tbl in included_tables:
            for c in self.schemas[tbl].columns:
                q = f"{tbl}.{c.name}"
                all_star_cols.append(q)
                valid_fields.add(q)
                if col_counts[c.name] == 1:
                    valid_fields.add(c.name)

        for row in current_rows:
            for tbl in included_tables:
                for c in self.schemas[tbl].columns:
                    if col_counts[c.name] == 1:
                        row[c.name] = row[f"{tbl}.{c.name}"]

        strategy_label = strategies_used[0] if len(strategies_used) == 1 else " -> ".join(strategies_used)
        stats = {
            "strategy": strategy_label,
            "index_used": ", ".join(indexes_used) if indexes_used else None,
            "planner_reason": f"Joined {', '.join(included_tables)} using {strategy_label}.",
            "records_scanned": records_scanned,
            "total_table_records": total_records,
        }
        return current_rows, all_star_cols, valid_fields, stats

    def _resolve_unqualified(self, row: Dict[str, Any], col: str, tables: List[str]) -> Any:
        matches = [row[f"{t}.{col}"] for t in tables if f"{t}.{col}" in row]
        if not matches:
            raise ExecutionError(f"Unknown join column '{col}'.")
        if len(matches) > 1:
            raise ExecutionError(f"Ambiguous join column '{col}'; qualify as table.{col}.")
        return matches[0]

    def _exec_select(self, stmt: SelectStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")

        if stmt.joins:
            candidates, star_cols, valid_fields, scan_stats = self._build_joined_rows(stmt)
        else:
            schema = self.schemas[stmt.table_name]
            star_cols = [c.name for c in schema.columns]
            valid_fields = set(star_cols)
            candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)

        filtered = [r for r in candidates if self._matches_where(r, stmt.where, valid_fields)]

        has_agg = any(item.expr_type == "AGG" for item in stmt.items)
        output_rows: List[Dict[str, Any]] = []

        if stmt.group_by or has_agg:
            if stmt.group_by:
                if stmt.group_by not in valid_fields:
                    raise ExecutionError(f"Unknown GROUP BY column '{stmt.group_by}'.")
                groups: Dict[Any, List[Dict[str, Any]]] = {}
                for r in filtered:
                    groups.setdefault(r.get(stmt.group_by), []).append(r)
            else:
                groups = {"__all__": filtered}

            for grp_key, grp_rows in groups.items():
                out_row: Dict[str, Any] = {}
                for item in stmt.items:
                    if item.expr_type == "COLUMN":
                        if stmt.group_by and item.field != stmt.group_by:
                            raise ExecutionError(
                                f"Column '{item.field}' must appear in GROUP BY or be inside an aggregate function."
                            )
                        out_row[item.alias or item.field] = grp_key
                    elif item.expr_type == "AGG":
                        assert item.alias is not None
                        out_row[item.alias] = self._compute_agg(item, grp_rows, valid_fields)
                    else:
                        raise ExecutionError("SELECT * cannot be combined with GROUP BY or aggregates.")

                if self._matches_having(stmt.having, grp_key, grp_rows, out_row, stmt.group_by, valid_fields):
                    output_rows.append(out_row)
        else:
            for r in filtered:
                out_row = {}
                for item in stmt.items:
                    if item.expr_type == "STAR":
                        for col_name in star_cols:
                            out_row[col_name] = r.get(col_name)
                    elif item.expr_type == "COLUMN":
                        if item.field not in valid_fields:
                            raise ExecutionError(f"Unknown column '{item.field}' in SELECT list.")
                        out_row[item.alias or item.field] = r.get(item.field)
                output_rows.append(out_row)

        if stmt.distinct:
            seen_rows = set()
            deduped: List[Dict[str, Any]] = []
            for row in output_rows:
                sig = tuple((k, type(v).__name__, v) for k, v in row.items())
                if sig not in seen_rows:
                    seen_rows.add(sig)
                    deduped.append(row)
            output_rows = deduped

        if stmt.order_by:
            sort_field, is_desc = stmt.order_by
            if output_rows and sort_field not in output_rows[0]:
                if not (stmt.group_by or has_agg) and sort_field in valid_fields:
                    paired = sorted(
                        zip(filtered, output_rows),
                        key=lambda pair: (pair[0].get(sort_field) is None, pair[0].get(sort_field)),
                        reverse=is_desc,
                    )
                    output_rows = [p[1] for p in paired]
                else:
                    raise ExecutionError(f"Cannot ORDER BY unknown field '{sort_field}'.")
            else:
                output_rows.sort(
                    key=lambda row: (row.get(sort_field) is None, row.get(sort_field)),
                    reverse=is_desc,
                )

        if stmt.limit is not None:
            output_rows = output_rows[: stmt.limit]

        scan_stats["rows_returned"] = len(output_rows)
        return {
            "status": "OK",
            "rows": output_rows,
            "stats": scan_stats,
        }