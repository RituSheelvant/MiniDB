import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from minidb import CorruptionError, MiniDB, QuerySyntaxError, SchemaError


class TestMiniDB(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="minidb_test_")
        self.db = MiniDB(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_crud_and_schema_constraints(self):
        self.db.execute("CREATE TABLE users (id INT UNIQUE, name STR NOT NULL, score FLOAT, active BOOL)")
        self.db.execute("INSERT INTO users VALUES (1, 'Alice', 95.5, TRUE)")
        self.db.execute("INSERT INTO users VALUES (2, 'Bob', 82.0, FALSE)")

        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES (1, 'DupID', 70.0, TRUE)")
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES (3, NULL, 70.0, TRUE)")
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES ('bad_int', 'Eve', 70.0, TRUE)")

        res = self.db.execute("SELECT name, score FROM users WHERE active = TRUE")
        self.assertEqual(res["rows"], [{"name": "Alice", "score": 95.5}])

        self.db.execute("UPDATE users SET score = 99.0 WHERE id = 2")
        res2 = self.db.execute("SELECT score FROM users WHERE id = 2")
        self.assertEqual(res2["rows"][0]["score"], 99.0)

        self.db.execute("DELETE FROM users WHERE id = 1")
        res3 = self.db.execute("SELECT * FROM users")
        self.assertEqual(len(res3["rows"]), 1)

    def test_filtering_grouping_aggregation_sorting(self):
        self.db.execute("CREATE TABLE sales (id INT UNIQUE, dept STR, amount FLOAT)")
        rows = [
            (1, "Eng", 100.0),
            (2, "Eng", 200.0),
            (3, "Sales", 150.0),
            (4, "Sales", 50.0),
            (5, "HR", 80.0),
        ]
        for r in rows:
            self.db.execute(f"INSERT INTO sales VALUES ({r[0]}, '{r[1]}', {r[2]})")

        res = self.db.execute(
            "SELECT dept, COUNT(id), SUM(amount), AVG(amount) FROM sales "
            "WHERE amount >= 50.0 GROUP BY dept ORDER BY SUM(amount) DESC"
        )
        self.assertEqual(len(res["rows"]), 3)
        self.assertEqual(res["rows"][0]["dept"], "Eng")
        self.assertEqual(res["rows"][0]["SUM(amount)"], 300.0)
        self.assertEqual(res["rows"][0]["AVG(amount)"], 150.0)

    def test_indexing_explain_and_query_planner(self):
        self.db.execute("CREATE TABLE items (id INT UNIQUE, category STR, price INT)")
        for i in range(1, 101):
            cat = "Books" if i % 2 == 0 else "Tech"
            self.db.execute(f"INSERT INTO items VALUES ({i}, '{cat}', {i * 10})")

        plan_before = self.db.execute("EXPLAIN SELECT id FROM items WHERE price > 900")
        self.assertEqual(plan_before["plan"]["strategy"], "FULL_TABLE_SCAN")

        self.db.execute("CREATE INDEX idx_price ON items (price) USING BTREE")
        plan_after = self.db.execute("EXPLAIN SELECT id FROM items WHERE price > 900")
        self.assertEqual(plan_after["plan"]["strategy"], "INDEX_SCAN")
        self.assertEqual(plan_after["plan"]["estimated_records_to_scan"], 10)

        with_idx = self.db.execute("SELECT id FROM items WHERE price > 900")
        self.assertEqual(with_idx["stats"]["strategy"], "INDEX_SCAN")
        self.assertEqual(with_idx["stats"]["records_scanned"], 10)
        self.assertEqual(len(with_idx["rows"]), 10)

    def test_composite_indexes_and_or_index_unions(self):
        self.db.execute("CREATE TABLE events (id INT UNIQUE, service STR, severity INT)")
        for i in range(1, 61):
            svc = ["auth", "billing", "search"][i % 3]
            sev = (i % 5) + 1
            self.db.execute(f"INSERT INTO events VALUES ({i}, '{svc}', {sev})")

        # 1. Composite multi-column index (service, severity)
        self.db.execute("CREATE INDEX idx_svc_sev ON events (service, severity) USING BTREE")
        comp_res = self.db.execute("SELECT id FROM events WHERE service = 'auth' AND severity = 5")
        self.assertEqual(comp_res["stats"]["strategy"], "INDEX_SCAN")
        self.assertIn("idx_svc_sev", comp_res["stats"]["index_used"])
        self.assertEqual(len(comp_res["rows"]), 4)

        # 2. OR Index Union across single-column index
        self.db.execute("CREATE INDEX idx_sev ON events (severity) USING HASH")
        or_res = self.db.execute("SELECT id FROM events WHERE severity = 1 OR severity = 5")
        self.assertEqual(or_res["stats"]["strategy"], "INDEX_UNION_SCAN")
        self.assertEqual(or_res["stats"]["records_scanned"], 24)
        self.assertEqual(len(or_res["rows"]), 24)

    def test_inner_joins_index_and_hash(self):
        self.db.execute("CREATE TABLE customers (id INT UNIQUE, name STR, tier STR)")
        self.db.execute("CREATE TABLE orders (oid INT UNIQUE, cust_id INT, total FLOAT)")

        self.db.execute("INSERT INTO customers VALUES (1, 'Aarav', 'Gold')")
        self.db.execute("INSERT INTO customers VALUES (2, 'Diya', 'Silver')")
        self.db.execute("INSERT INTO orders VALUES (101, 1, 250.0)")
        self.db.execute("INSERT INTO orders VALUES (102, 1, 150.0)")
        self.db.execute("INSERT INTO orders VALUES (103, 2, 500.0)")

        # Hash Join (cust_id has no index yet)
        hj = self.db.execute(
            "SELECT customers.name, SUM(orders.total) FROM customers "
            "JOIN orders ON customers.id = orders.cust_id "
            "GROUP BY customers.name ORDER BY SUM(orders.total) DESC"
        )
        self.assertEqual(hj["stats"]["strategy"], "HASH_JOIN")
        self.assertEqual(hj["rows"][0], {"customers.name": "Diya", "SUM(orders.total)": 500.0})
        self.assertEqual(hj["rows"][1], {"customers.name": "Aarav", "SUM(orders.total)": 400.0})

        # Index Nested-Loop Join (after indexing orders.cust_id)
        self.db.execute("CREATE INDEX idx_cust_id ON orders (cust_id) USING HASH")
        inlj = self.db.execute(
            "SELECT customers.name, orders.total FROM customers "
            "INNER JOIN orders ON customers.id = orders.cust_id "
            "WHERE customers.tier = 'Gold' ORDER BY orders.total ASC"
        )
        self.assertEqual(inlj["stats"]["strategy"], "INDEX_NESTED_LOOP_JOIN")
        self.assertEqual(len(inlj["rows"]), 2)
        self.assertEqual(inlj["rows"][0]["orders.total"], 150.0)

    def test_vacuum_compaction_and_crc_corruption_detection(self):
        self.db.execute("CREATE TABLE logs (id INT UNIQUE, msg STR)")
        for i in range(1, 11):
            self.db.execute(f"INSERT INTO logs VALUES ({i}, 'log_entry_{i}')")
        self.db.execute("DELETE FROM logs WHERE id <= 8")

        vac = self.db.execute("VACUUM logs")
        self.assertGreater(vac["bytes_reclaimed"], 0)
        self.assertEqual(len(self.db.execute("SELECT * FROM logs")["rows"]), 2)

        mdb_file = os.path.join(self.test_dir, "logs.mdb")
        with open(mdb_file, "r+b") as f:
            data = bytearray(f.read())
            data[-1] ^= 0xFF
            f.seek(0)
            f.write(data)

        with self.assertRaises(CorruptionError):
            MiniDB(self.test_dir)

    def test_adversarial_edge_cases(self):
        self.db.execute("CREATE TABLE edge (id INT UNIQUE, val FLOAT, tag STR);")
        self.db.execute("INSERT INTO edge VALUES (1, -42.5, 'neg');")
        self.db.execute("INSERT INTO edge VALUES (2, NULL, 'empty');")
        self.db.execute("INSERT INTO edge VALUES (3, 100.0, 'pos');")

        res = self.db.execute("SELECT id FROM edge WHERE val < 0 OR val > 50 ORDER BY val ASC;")
        self.assertEqual(res["rows"], [{"id": 1}, {"id": 3}])

        with self.assertRaises(QuerySyntaxError):
            self.db.execute("INVALID COMMAND FROM edge")

    def test_transactions_and_rollback(self):
        self.db.execute("CREATE TABLE accounts (id INT UNIQUE, balance INT)")
        self.db.execute("INSERT INTO accounts VALUES (1, 500)")

        try:
            with self.db.transaction():
                self.db.execute("UPDATE accounts SET balance = 200 WHERE id = 1")
                raise RuntimeError("Simulated business logic failure!")
        except RuntimeError:
            pass

        res = self.db.execute("SELECT balance FROM accounts WHERE id = 1")
        self.assertEqual(res["rows"][0]["balance"], 500)

    def test_crash_recovery_sigkill(self):
        self.db.execute("CREATE TABLE ledger (id INT UNIQUE, note STR)")
        self.db.execute("INSERT INTO ledger VALUES (1, 'committed_row')")

        crash_script = (
            "import os, signal; "
            "from minidb import MiniDB; "
            f"db = MiniDB({self.test_dir!r}); "
            "db.begin(); "
            "db.execute(\"INSERT INTO ledger VALUES (2, 'uncommitted_crash_row')\"); "
            "os.kill(os.getpid(), signal.SIGKILL) if hasattr(signal, 'SIGKILL') else os._exit(1)"
        )
        proc = subprocess.run([sys.executable, "-c", crash_script])
        self.assertNotEqual(proc.returncode, 0)

        recovered_db = MiniDB(self.test_dir)
        self.assertEqual(recovered_db.last_recovery_stats["discarded_uncommitted_txns"], 1)
        res = recovered_db.execute("SELECT * FROM ledger")
        self.assertEqual(res["rows"], [{"id": 1, "note": "committed_row"}])

    def test_concurrency_threads(self):
        self.db.execute("CREATE TABLE counter (id INT UNIQUE, val INT)")
        for i in range(1, 21):
            self.db.execute(f"INSERT INTO counter VALUES ({i}, {i})")

        errors = []

        def reader():
            try:
                for _ in range(25):
                    res = self.db.execute("SELECT COUNT(id) FROM counter WHERE val > 5")
                    self.assertEqual(res["rows"][0]["COUNT(id)"], 15)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(21, 36):
                    self.db.execute(f"INSERT INTO counter VALUES ({i}, 0)")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()