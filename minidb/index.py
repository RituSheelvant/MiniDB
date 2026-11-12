import bisect
from typing import Any, Dict, List, Optional, Set, Tuple
from .parser import Condition, WhereClause


class BaseIndex:
    def __init__(self, index_name: str, table_name: str, columns: List[str], index_type: str):
        self.index_name = index_name
        self.table_name = table_name
        self.columns = list(columns)
        self.column_name = self.columns[0]  # Primary/leading column convenience attribute
        self.index_type = index_type

    def extract_key(self, record: Dict[str, Any]) -> Optional[Any]:
        if len(self.columns) == 1:
            return record.get(self.columns[0])
        vals = tuple(record.get(c) for c in self.columns)
        if any(v is None for v in vals):
            return None
        return vals

    def insert(self, key: Any, row_id: int) -> None:
        raise NotImplementedError

    def remove(self, key: Any, row_id: int) -> None:
        raise NotImplementedError

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        raise NotImplementedError


class HashIndex(BaseIndex):
    """O(1) hash bucket index supporting single-column and composite equality lookups."""

    def __init__(self, index_name: str, table_name: str, columns: List[str]):
        super().__init__(index_name, table_name, columns, "HASH")
        self._buckets: Dict[Any, Set[int]] = {}

    def insert(self, key: Any, row_id: int) -> None:
        if key is None:
            return
        self._buckets.setdefault(key, set()).add(row_id)

    def remove(self, key: Any, row_id: int) -> None:
        if key is None or key not in self._buckets:
            return
        self._buckets[key].discard(row_id)
        if not self._buckets[key]:
            del self._buckets[key]

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        if op == "=" and value is not None:
            return set(self._buckets.get(value, set()))
        return None


class BTreeRangeIndex(BaseIndex):
    """
    Sorted B-Tree directory using binary search (bisect) for O(log N) range queries
    (<, <=, >, >=, =) on single or composite multi-column keys.
    """

    def __init__(self, index_name: str, table_name: str, columns: List[str]):
        super().__init__(index_name, table_name, columns, "BTREE")
        self._keys: List[Any] = []
        self._rids: List[int] = []

    def insert(self, key: Any, row_id: int) -> None:
        if key is None:
            return
        idx = bisect.bisect_left(self._keys, key)
        self._keys.insert(idx, key)
        self._rids.insert(idx, row_id)

    def remove(self, key: Any, row_id: int) -> None:
        if key is None:
            return
        left = bisect.bisect_left(self._keys, key)
        right = bisect.bisect_right(self._keys, key)
        for i in range(left, right):
            if self._rids[i] == row_id:
                self._keys.pop(i)
                self._rids.pop(i)
                return

    def lookup(self, op: str, value: Any) -> Optional[Set[int]]:
        if value is None:
            return None

        if op == "=":
            left = bisect.bisect_left(self._keys, value)
            right = bisect.bisect_right(self._keys, value)
            return set(self._rids[left:right])
        elif op == "<":
            idx = bisect.bisect_left(self._keys, value)
            return set(self._rids[:idx])
        elif op == "<=":
            idx = bisect.bisect_right(self._keys, value)
            return set(self._rids[:idx])
        elif op == ">":
            idx = bisect.bisect_right(self._keys, value)
            return set(self._rids[idx:])
        elif op == ">=":
            idx = bisect.bisect_left(self._keys, value)
            return set(self._rids[idx:])
        return None

    def lookup_prefix_range(self, prefix_vals: Tuple[Any, ...], range_op: str, range_val: Any) -> Set[int]:
        """
        Accelerates composite BTree queries where leading columns use '='
        and the next column uses a range operator (e.g., dept = 'Eng' AND salary > 100000).
        """
        if any(v is None for v in prefix_vals) or range_val is None:
            return set()
        # Locate the slice matching prefix_vals via leading-element comparison
        matched: Set[int] = set()
        plen = len(prefix_vals)
        for k, rid in zip(self._keys, self._rids):
            if not isinstance(k, tuple) or len(k) <= plen:
                continue
            if k[:plen] == prefix_vals:
                target = k[plen]
                if (
                    (range_op == "=" and target == range_val)
                    or (range_op == "<" and target < range_val)
                    or (range_op == "<=" and target <= range_val)
                    or (range_op == ">" and target > range_val)
                    or (range_op == ">=" and target >= range_val)
                ):
                    matched.add(rid)
            elif k[:plen] > prefix_vals:
                break
        return matched


class QueryPlanner:
    """
    Cost-based query planner supporting single-column indexes, composite multi-column indexes,
    AND index intersections, and OR index unions.
    """

    INDEX_THRESHOLD = 25

    @classmethod
    def _split_or_branches(cls, where: WhereClause) -> List[List[Condition]]:
        branches: List[List[Condition]] = [[]]
        for i, cond in enumerate(where.conditions):
            branches[-1].append(cond)
            if i < len(where.connectors) and where.connectors[i] == "OR":
                branches.append([])
        return branches

    @classmethod
    def _plan_and_branch(
        cls,
        conditions: List[Condition],
        indexes: Dict[Tuple[str, ...], BaseIndex],
    ) -> Tuple[Optional[Set[int]], List[str]]:
        """Resolves a single AND-conjunction branch using composite and single-column indexes."""
        cond_by_col: Dict[str, List[Condition]] = {}
        for c in conditions:
            col_name = c.field.split(".")[-1]
            cond_by_col.setdefault(col_name, []).append(c)

        candidate_rids: Optional[Set[int]] = None
        used_indexes: List[str] = []

        # 1. Try composite multi-column indexes first (highest selectivity)
        composite_indexes = sorted(
            [idx for cols, idx in indexes.items() if len(cols) > 1],
            key=lambda i: len(i.columns),
            reverse=True,
        )
        for idx in composite_indexes:
            # Exact equality across all columns in the composite index
            eq_vals: List[Any] = []
            all_eq = True
            for col in idx.columns:
                eq_conds = [c for c in cond_by_col.get(col, []) if c.op == "=" and c.value is not None]
                if eq_conds:
                    eq_vals.append(eq_conds[0].value)
                else:
                    all_eq = False
                    break

            if all_eq:
                rids = idx.lookup("=", tuple(eq_vals))
                if rids is not None:
                    candidate_rids = rids if candidate_rids is None else (candidate_rids & rids)
                    used_indexes.append(f"{idx.index_name}({idx.index_type} on {','.join(idx.columns)})")
                    continue

            # Prefix equality + trailing range on composite BTree index
            if isinstance(idx, BTreeRangeIndex) and len(idx.columns) >= 2:
                prefix_vals: List[Any] = []
                for col in idx.columns[:-1]:
                    eq_conds = [c for c in cond_by_col.get(col, []) if c.op == "=" and c.value is not None]
                    if eq_conds:
                        prefix_vals.append(eq_conds[0].value)
                    else:
                        break
                if len(prefix_vals) == len(idx.columns) - 1:
                    last_col = idx.columns[-1]
                    range_conds = [
                        c for c in cond_by_col.get(last_col, [])
                        if c.op in ("<", "<=", ">", ">=") and c.value is not None
                    ]
                    if range_conds:
                        rc = range_conds[0]
                        rids = idx.lookup_prefix_range(tuple(prefix_vals), rc.op, rc.value)
                        candidate_rids = rids if candidate_rids is None else (candidate_rids & rids)
                        used_indexes.append(f"{idx.index_name}({idx.index_type} on {','.join(idx.columns)})")

        # 2. Intersect any matching single-column indexes
        for cond in conditions:
            col_name = cond.field.split(".")[-1]
            single_idx = indexes.get((col_name,))
            if single_idx is not None:
                rids = single_idx.lookup(cond.op, cond.value)
                if rids is not None:
                    candidate_rids = rids if candidate_rids is None else (candidate_rids & rids)
                    label = f"{single_idx.index_name}({single_idx.index_type} on {col_name})"
                    if label not in used_indexes:
                        used_indexes.append(label)

        return candidate_rids, used_indexes

    @classmethod
    def choose_plan(
        cls,
        table_size: int,
        where: Optional[WhereClause],
        indexes: Dict[Tuple[str, ...], BaseIndex],
    ) -> Dict[str, Any]:
        if where is None or not where.conditions or not indexes:
            return {
                "strategy": "FULL_TABLE_SCAN",
                "index_used": None,
                "candidate_rids": None,
                "reason": "No applicable WHERE filter or index available.",
            }

        if table_size < cls.INDEX_THRESHOLD:
            return {
                "strategy": "FULL_TABLE_SCAN",
                "index_used": None,
                "candidate_rids": None,
                "reason": f"Table size ({table_size}) below index threshold ({cls.INDEX_THRESHOLD}); sequential scan is cheaper.",
            }

        or_branches = cls._split_or_branches(where)
        union_rids: Set[int] = set()
        all_used_indexes: List[str] = []

        for branch in or_branches:
            branch_rids, branch_indexes = cls._plan_and_branch(branch, indexes)
            if branch_rids is None:
                # If any OR branch lacks an index, we must fall back to a full scan for correctness
                reason = (
                    "At least one OR branch lacks an index; full scan required."
                    if len(or_branches) > 1
                    else "Predicate operator unsupported by available index."
                )
                return {
                    "strategy": "FULL_TABLE_SCAN",
                    "index_used": None,
                    "candidate_rids": None,
                    "reason": reason,
                }
            union_rids |= branch_rids
            for label in branch_indexes:
                if label not in all_used_indexes:
                    all_used_indexes.append(label)

        strategy = "INDEX_UNION_SCAN" if len(or_branches) > 1 else "INDEX_SCAN"
        return {
            "strategy": strategy,
            "index_used": ", ".join(all_used_indexes),
            "candidate_rids": union_rids,
            "reason": f"Reduced scan space from {table_size} rows to {len(union_rids)} candidate RIDs via {strategy}.",
        }