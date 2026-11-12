import copy
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from .errors import ExecutionError, SchemaError, TransactionError
from .index import BaseIndex, BTreeRangeIndex, HashIndex, QueryPlanner
from .parser import (
    Condition,
    CreateIndexStmt,
    CreateTableStmt,
    DeleteStmt,
    ExplainStmt,
    InsertStmt,
    SelectItem,
    SelectStmt,
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
    multi-field indexing, cost-based planning, WAL crash recovery, and concurrency.
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
        self.indexes: Dict[str, Dict[str, BaseIndex]] = {}
        self.index_meta: List[Dict[str, str]] = []

        self._tx_local = threading.local()
        self.last_recovery_stats: Dict[str, Any] = {}

        self._startup_and_recover()

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

        for meta in self.index_meta:
            tbl = meta["table_name"]
            col = meta["column_name"]
            if tbl in self.schemas and col in self.schemas[tbl].col_map:
                self._build_index_in_memory(meta["index_name"], tbl, col, meta["index_type"])

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

    def _build_index_in_memory(self, index_name: str, table_name: str, column_name: str, index_type: str) -> BaseIndex:
        idx: BaseIndex
        if index_type == "HASH":
            idx = HashIndex(index_name, table_name, column_name)
        else:
            idx = BTreeRangeIndex(index_name, table_name, column_name)

        for rid, rec in self.tables[table_name].items():
            idx.insert(rec.get(column_name), rid)

        self.indexes.setdefault(table_name, {})[column_name] = idx
        return idx

    # --- Transaction Management (NO-STEAL / FORCE WAL Protocol) ---

    def begin(self) -> str:
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
                    for col, idx in self.indexes.get(tbl, {}).items():
                        idx.remove(rec.get(col), rid)

            elif action == "UNDO_UPDATE":
                curr_rec = self.tables[tbl].get(rid)
                if curr_rec:
                    for col, idx in self.indexes.get(tbl, {}).items():
                        idx.remove(curr_rec.get(col), rid)
                self.tables[tbl][rid] = old_rec
                for col, idx in self.indexes.get(tbl, {}).items():
                    idx.insert(old_rec.get(col), rid)

            elif action == "UNDO_DELETE":
                self.tables[tbl][rid] = old_rec
                for col, idx in self.indexes.get(tbl, {}).items():
                    idx.insert(old_rec.get(col), rid)

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

    # --- Query Execution Entry Point ---

    def execute(self, query: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        stmt = parse_query(query)

        if isinstance(stmt, SelectStmt):
            with self.rwlock.read_lock():
                result = self._exec_select(stmt)
        elif isinstance(stmt, ExplainStmt):
            with self.rwlock.read_lock():
                result = self._exec_explain(stmt)
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
                    elif isinstance(stmt, CreateIndexStmt):
                        result = self._exec_create_index(stmt)
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

    # --- Utility & Maintenance Executors ---

    def _exec_explain(self, stmt: ExplainStmt) -> Dict[str, Any]:
        sel = stmt.select_stmt
        if sel.table_name not in self.schemas:
            raise ExecutionError(f"Table '{sel.table_name}' does not exist.")
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

        cols = [Column(name, dtype, nn, uq) for name, dtype, nn, uq in stmt.columns]
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
                self._build_index_in_memory(idx_name, stmt.table_name, col.name, "HASH")
                self.index_meta.append({
                    "index_name": idx_name,
                    "table_name": stmt.table_name,
                    "column_name": col.name,
                    "index_type": "HASH",
                })
                self._save_index_meta()

        return {"status": "OK", "message": f"Table '{stmt.table_name}' created."}

    def _exec_create_index(self, stmt: CreateIndexStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        if stmt.column_name not in schema.col_map:
            raise ExecutionError(f"Column '{stmt.column_name}' does not exist in table '{stmt.table_name}'.")

        self._build_index_in_memory(stmt.index_name, stmt.table_name, stmt.column_name, stmt.index_type)
        self.index_meta = [
            m for m in self.index_meta
            if not (m["table_name"] == stmt.table_name and m["column_name"] == stmt.column_name)
        ]
        self.index_meta.append({
            "index_name": stmt.index_name,
            "table_name": stmt.table_name,
            "column_name": stmt.column_name,
            "index_type": stmt.index_type,
        })
        self._save_index_meta()

        return {
            "status": "OK",
            "message": f"Index '{stmt.index_name}' ({stmt.index_type}) created on {stmt.table_name}({stmt.column_name}).",
        }

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

        rid = self.next_rids[stmt.table_name]
        self.next_rids[stmt.table_name] = rid + 1
        clean_rec["_rid"] = rid

        tx = self._get_tx_state()
        assert tx is not None
        self.wal.append(tx_id=tx["tx_id"], op="INSERT", table=stmt.table_name, after=clean_rec)

        self.tables[stmt.table_name][rid] = clean_rec
        for col_name, idx in self.indexes.get(stmt.table_name, {}).items():
            idx.insert(clean_rec.get(col_name), rid)

        tx["disk_writes"].append(("APPEND", stmt.table_name, rid, clean_rec))
        tx["undo_log"].append(("UNDO_INSERT", stmt.table_name, rid, None))
        tx["catalog_changed"] = True

        return {"status": "OK", "rows_affected": 1, "rid": rid}

    def _evaluate_condition(self, row: Dict[str, Any], cond: Condition, schema: TableSchema) -> bool:
        if cond.field not in schema.col_map:
            raise ExecutionError(f"Unknown column '{cond.field}' in WHERE clause.")
        lhs = row.get(cond.field)
        rhs = cond.value
        op = cond.op

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

    def _matches_where(self, row: Dict[str, Any], where: Optional[WhereClause], schema: TableSchema) -> bool:
        if where is None or not where.conditions:
            return True

        result = self._evaluate_condition(row, where.conditions[0], schema)
        for i, conn in enumerate(where.connectors):
            next_val = self._evaluate_condition(row, where.conditions[i + 1], schema)
            if conn == "AND":
                result = result and next_val
            elif conn == "OR":
                result = result or next_val
        return result

    def _select_candidate_rows(self, table_name: str, where: Optional[WhereClause]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        table_map = self.tables[table_name]
        table_indexes = self.indexes.get(table_name, {})
        plan = QueryPlanner.choose_plan(len(table_map), where, table_indexes)

        if plan["strategy"] == "INDEX_SCAN" and plan["candidate_rids"] is not None:
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
        candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)

        matched_rows = [r for r in candidates if self._matches_where(r, stmt.where, schema)]
        tx = self._get_tx_state()
        assert tx is not None

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
            validated["_rid"] = rid

            self.wal.append(tx_id=tx["tx_id"], op="UPDATE", table=stmt.table_name, before=old_row, after=validated)

            self.tables[stmt.table_name][rid] = validated
            for col_name, idx in self.indexes.get(stmt.table_name, {}).items():
                idx.remove(old_row.get(col_name), rid)
                idx.insert(validated.get(col_name), rid)

            tx["disk_writes"].append(("DELETE", stmt.table_name, rid, None))
            tx["disk_writes"].append(("APPEND", stmt.table_name, rid, validated))
            tx["undo_log"].append(("UNDO_UPDATE", stmt.table_name, rid, copy.deepcopy(old_row)))
            updated_count += 1

        return {"status": "OK", "rows_affected": updated_count, "stats": scan_stats}

    def _exec_delete(self, stmt: DeleteStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]
        candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)

        matched_rows = [r for r in candidates if self._matches_where(r, stmt.where, schema)]
        tx = self._get_tx_state()
        assert tx is not None

        deleted_count = 0
        for old_row in matched_rows:
            rid = old_row["_rid"]
            self.wal.append(tx_id=tx["tx_id"], op="DELETE", table=stmt.table_name, before=old_row)

            self.tables[stmt.table_name].pop(rid, None)
            for col_name, idx in self.indexes.get(stmt.table_name, {}).items():
                idx.remove(old_row.get(col_name), rid)

            tx["disk_writes"].append(("DELETE", stmt.table_name, rid, None))
            tx["undo_log"].append(("UNDO_DELETE", stmt.table_name, rid, copy.deepcopy(old_row)))
            deleted_count += 1

        return {"status": "OK", "rows_affected": deleted_count, "stats": scan_stats}

    def _compute_agg(self, item: SelectItem, rows: List[Dict[str, Any]], schema: TableSchema) -> Any:
        func = item.func
        field = item.field

        if func == "COUNT":
            if field == "*":
                return len(rows)
            if field not in schema.col_map:
                raise ExecutionError(f"Unknown column '{field}' in COUNT().")
            return sum(1 for r in rows if r.get(field) is not None)

        if field not in schema.col_map:
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

    def _exec_select(self, stmt: SelectStmt) -> Dict[str, Any]:
        if stmt.table_name not in self.schemas:
            raise ExecutionError(f"Table '{stmt.table_name}' does not exist.")
        schema = self.schemas[stmt.table_name]

        candidates, scan_stats = self._select_candidate_rows(stmt.table_name, stmt.where)
        filtered = [r for r in candidates if self._matches_where(r, stmt.where, schema)]

        has_agg = any(item.expr_type == "AGG" for item in stmt.items)
        output_rows: List[Dict[str, Any]] = []

        if stmt.group_by or has_agg:
            if stmt.group_by:
                if stmt.group_by not in schema.col_map:
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
                        out_row[item.alias] = self._compute_agg(item, grp_rows, schema)
                    else:
                        raise ExecutionError("SELECT * cannot be combined with GROUP BY or aggregates.")
                output_rows.append(out_row)
        else:
            for r in filtered:
                out_row = {}
                for item in stmt.items:
                    if item.expr_type == "STAR":
                        for col in schema.columns:
                            out_row[col.name] = r.get(col.name)
                    elif item.expr_type == "COLUMN":
                        if item.field not in schema.col_map:
                            raise ExecutionError(f"Unknown column '{item.field}' in SELECT list.")
                        out_row[item.field] = r.get(item.field)
                output_rows.append(out_row)

        if stmt.order_by:
            sort_field, is_desc = stmt.order_by
            if output_rows and sort_field not in output_rows[0]:
                if not (stmt.group_by or has_agg) and sort_field in schema.col_map:
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