# MiniDB — Custom Embeddable Database & Query Execution Engine

MiniDB is a zero-dependency, embeddable Python database library featuring a custom binary storage format, hand-written lexer and recursive-descent query parser, single-column and composite multi-column Hash/B-Tree indexes, a cost-based query planner (`EXPLAIN`), Hash & Index Nested-Loop `JOIN`s, Write-Ahead Logging (WAL) crash recovery, online `VACUUM` compaction, and thread-safe concurrency.

---

## Architecture Overview

1. **Storage Layer (`minidb/storage.py`)**: Stores table records in binary `.mdb` files using length-prefixed frames (`[CRC32 (4B) | Length (4B) | Tombstone (1B) | Payload]`) with `os.fsync()` durability, $O(1)$ tombstone deletes, online `VACUUM` compaction, and strict schema enforcement (`INT`, `FLOAT`, `STR`, `BOOL`, `NOT NULL`, `UNIQUE`).
2. **Write-Ahead Log & Recovery (`minidb/wal.py`, `minidb/engine.py`)**: Implements a NO-STEAL / FORCE transaction protocol. All mutations are logged with CRC32 checksums before committing. Upon restart, committed transactions are replayed and incomplete/uncommitted transactions from hard crashes (`SIGKILL`) are discarded.
3. **Custom Query Language & AST (`minidb/parser.py`)**: Hand-written regex tokenizer and recursive-descent parser (no external SQL parsing libraries). Supports `CREATE TABLE`, `CREATE INDEX` (single & composite), `INSERT`, `SELECT` (`INNER JOIN`, `WHERE` with `AND`/`OR`, `GROUP BY`, `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `ORDER BY`, `LIMIT`), `UPDATE`, `DELETE`, `EXPLAIN`, and `VACUUM`.
4. **Indexing & Cost-Based Query Planner (`minidb/index.py`)**: Implements `HashIndex` ($O(1)$ equality) and `BTreeRangeIndex` ($O(\log N)$ binary-search range scans) on single or composite multi-column keys. `QueryPlanner` supports `INDEX_SCAN`, `INDEX_UNION_SCAN` (for `OR` predicates), `HASH_JOIN`, and `INDEX_NESTED_LOOP_JOIN`, returning execution telemetry (`execution_time_ms`, `records_scanned`, `index_used`).
5. **Concurrency (`minidb/engine.py`)**: Uses a custom `ReadWriteLock` permitting parallel read queries while serializing write transactions.

---

## Quick Usage Example

```python
from minidb import MiniDB

db = MiniDB("./my_database")
db.execute("CREATE TABLE users (id INT UNIQUE, name STR NOT NULL, dept STR, salary FLOAT)")
db.execute("CREATE INDEX idx_dept_sal ON users (dept, salary) USING BTREE")

with db.transaction():
    db.execute("INSERT INTO users VALUES (1, 'Alice', 'Eng', 120000.0)")
    db.execute("INSERT INTO users VALUES (2, 'Bob', 'Eng', 135000.0)")

result = db.execute("SELECT name, salary FROM users WHERE dept = 'Eng' AND salary > 125000.0")
print(result["rows"])   # [{'name': 'Bob', 'salary': 135000.0}]
print(result["stats"])  # {'strategy': 'INDEX_SCAN', 'index_used': 'idx_dept_sal(BTREE on dept,salary)', ...}