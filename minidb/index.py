import bisect
from typing import Any, Dict, List, Optional, Set, Tuple
from .parser import WhereClause


class BaseIndex:
    def __init__(self, index_name: str, table_name: str, column_name: str, index_type: str):
        self.index_name = index_name
        self.table_name = table_name
        self.column_name = column_name
        self.index_type = index_type

    def insert(self, key: Any, rid: int) -> None:
        raise NotImplementedError

    def remove(self, key: Any, rid: int) -> None:
        raise NotImplementedError

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class HashIndex(BaseIndex):
    """O(1) hash-bucket index optimized for exact equality (=) lookups."""

    def __init__(self, index_name: str, table_name: str, column_name: str):
        super().__init__(index_name, table_name, column_name, "HASH")
        self.buckets: Dict[Any, Set[int]] = {}

    def insert(self, key: Any, rid: int) -> None:
        if key is None:
            return
        self.buckets.setdefault(key, set()).add(rid)

    def remove(self, key: Any, rid: int) -> None:
        if key in self.buckets:
            self.buckets[key].discard(rid)
            if not self.buckets[key]:
                del self.buckets[key]

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        if op == "=" and value is not None:
            return set(self.buckets.get(value, set()))
        return None  # Hash index cannot answer range queries (<, >, !=)

    def clear(self) -> None:
        self.buckets.clear()


class BTreeRangeIndex(BaseIndex):
    """
    Sorted key-RID search structure supporting O(log N) binary-search range scans
    (=, <, <=, >, >=) as well as point queries.
    """

    def __init__(self, index_name: str, table_name: str, column_name: str):
        super().__init__(index_name, table_name, column_name, "BTREE")
        self.entries: List[Tuple[Any, int]] = []  # Maintained in sorted order by (key, rid)

    def insert(self, key: Any, rid: int) -> None:
        if key is None:
            return
        bisect.insort(self.entries, (key, rid))

    def remove(self, key: Any, rid: int) -> None:
        if key is None:
            return
        idx = bisect.bisect_left(self.entries, (key, rid))
        if idx < len(self.entries) and self.entries[idx] == (key, rid):
            self.entries.pop(idx)

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        if value is None or op == "!=":
            return None

        if not self.entries:
            return set()

        # Extract keys lazily via bisect using sentinel tuples
        # Since rid is always >= 1, (value, -1) precedes all entries with key == value
        # and (value, float('inf')) follows all entries with key == value.
        if op == "=":
            left = bisect.bisect_left(self.entries, (value, -1))
            right = bisect.bisect_right(self.entries, (value, float("inf")))
            return {rid for _, rid in self.entries[left:right]}

        elif op == "<":
            right = bisect.bisect_left(self.entries, (value, -1))
            return {rid for _, rid in self.entries[:right]}

        elif op == "<=":
            right = bisect.bisect_right(self.entries, (value, float("inf")))
            return {rid for _, rid in self.entries[:right]}

        elif op == ">":
            left = bisect.bisect_right(self.entries, (value, float("inf")))
            return {rid for _, rid in self.entries[left:]}

        elif op == ">=":
            left = bisect.bisect_left(self.entries, (value, -1))
            return {rid for _, rid in self.entries[left:]}

        return None

    def clear(self) -> None:
        self.entries.clear()


class QueryPlanner:
    """
    Cost-based planner that chooses between an IndexScan and FullTableScan.
    For very small tables (< MIN_ROWS_FOR_INDEX), sequential memory scans have lower
    overhead than index traversal. For larger datasets, matching indexes cut scanned
    records from O(N) down to O(log N + K).
    """

    MIN_ROWS_FOR_INDEX = 25

    @classmethod
    def choose_plan(
        cls,
        table_size: int,
        where: Optional[WhereClause],
        table_indexes: Dict[str, BaseIndex],  # column_name -> BaseIndex
    ) -> Dict[str, Any]:
        if not where or not where.conditions or not table_indexes:
            return {
                "strategy": "FULL_TABLE_SCAN",
                "index_used": None,
                "candidate_rids": None,
                "reason": "No applicable WHERE filter or index available.",
            }

        # If all connectors are AND, we can intersect candidate RIDs from any indexed predicate
        all_and = all(c == "AND" for c in where.connectors)
        if not all_and and len(where.conditions) > 1:
            return {
                "strategy": "FULL_TABLE_SCAN",
                "index_used": None,
                "candidate_rids": None,
                "reason": "OR predicates require full scan unless all branches are indexed.",
            }

        if table_size < cls.MIN_ROWS_FOR_INDEX:
            return {
                "strategy": "FULL_TABLE_SCAN",
                "index_used": None,
                "candidate_rids": None,
                "reason": f"Table size ({table_size}) below index threshold ({cls.MIN_ROWS_FOR_INDEX}); sequential scan is cheaper.",
            }

        matched_indexes: List[str] = []
        candidate_rids: Optional[Set[int]] = None

        for cond in where.conditions:
            idx = table_indexes.get(cond.field)
            if idx is not None:
                rids = idx.lookup(cond.op, cond.value)
                if rids is not None:
                    matched_indexes.append(f"{idx.index_name}({idx.index_type} on {cond.field})")
                    if candidate_rids is None:
                        candidate_rids = rids
                    else:
                        candidate_rids &= rids

        if matched_indexes and candidate_rids is not None:
            return {
                "strategy": "INDEX_SCAN",
                "index_used": ", ".join(matched_indexes),
                "candidate_rids": candidate_rids,
                "reason": f"Reduced scan space from {table_size} rows to {len(candidate_rids)} candidate RIDs.",
            }

        return {
            "strategy": "FULL_TABLE_SCAN",
            "index_used": None,
            "candidate_rids": None,
            "reason": "Predicate operator unsupported by available index.",
        }