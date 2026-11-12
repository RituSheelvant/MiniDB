import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from minidb import MiniDB, QuerySyntaxError, SchemaError, TransactionError


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

        # Constraint checks
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES (1, 'DupID', 70.0, TRUE)")  # UNIQUE violation
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES (3, NULL, 70.0, TRUE)")  # NOT NULL violation
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO users VALUES ('bad_int', 'Eve', 70.0, TRUE)")  # Type mismatch

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

    def test_indexing_and_query_planner(self):
        self.db.execute("CREATE TABLE items (id INT UNIQUE, category STR, price INT)")
        for i in range(1, 101):
            cat = "Books" if i % 2 == 0 else "Tech"
            self.db.execute(f"INSERT INTO items VALUES ({i}, '{cat}', {i * 10})")

        # Before index on price -> FULL_TABLE_SCAN
        no_idx = self.db.execute("SELECT id FROM items WHERE price > 900")
        self.assertEqual(no_idx["stats"]["strategy"], "FULL_TABLE_SCAN")
        self.assertEqual(no_idx["stats"]["records_scanned"], 100)

        # Create B-Tree range index on price -> INDEX_SCAN
        self.db.execute("CREATE INDEX idx_price ON items (price) USING BTREE")
        with_idx = self.db.execute("SELECT id FROM items WHERE price > 900")
        self.assertEqual(with_idx["stats"]["strategy"], "INDEX_SCAN")
        self.assertEqual(with_idx["stats"]["records_scanned"], 10)
        self.assertEqual(len(with_idx["rows"]), 10)

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
        # 1. Commit baseline data
        self.db.execute("CREATE TABLE ledger (id INT UNIQUE, note STR)")
        self.db.execute("INSERT INTO ledger VALUES (1, 'committed_row')")

        # 2. Spawn child process that starts a txn, inserts uncommitted rows, and hard-kills itself
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

        # 3. Reopen database -> uncommitted row must be discarded, committed row preserved
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