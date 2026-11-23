import os
import shutil
import tempfile
import unittest

from minidb.engine import MiniDB
from minidb.errors import ExecutionError, QuerySyntaxError, SchemaError


class TestMiniDBExtensions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="minidb_ext_test_")
        self.db = MiniDB(self.test_dir)

    def tearDown(self):
        if not self.db._closed:
            self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Parameterized Queries (? and :named)
    def test_parameterized_queries_and_injection_safety(self):
        self.db.execute("CREATE TABLE users (id INT UNIQUE, username STR NOT NULL, active BOOL)")
        self.db.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            (1, "alice' OR '1'='1", True),
        )
        self.db.execute(
            "INSERT INTO users VALUES (:id, :user, :active)",
            {"id": 2, "user": "bob", "active": False},
        )

        # Injection string is treated strictly as literal data, not SQL syntax
        rows = self.db.query("SELECT * FROM users WHERE username = ?", ("alice' OR '1'='1",))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 1)

        with self.assertRaises(QuerySyntaxError):
            self.db.execute("SELECT * FROM users WHERE id = ?", (1, 2))  # Extra param

        with self.assertRaises(QuerySyntaxError):
            self.db.execute("SELECT * FROM users WHERE id = :missing", {"id": 1})

    # 2. Schema Evolution (ALTER TABLE ADD / DROP / RENAME COLUMN)
    def test_alter_table_lifecycle_and_persistence(self):
        self.db.execute("CREATE TABLE products (id INT UNIQUE, name STR NOT NULL)")
        self.db.execute("INSERT INTO products VALUES (1, 'Laptop')")
        self.db.execute("INSERT INTO products VALUES (2, 'Mouse')")

        # ADD COLUMN with DEFAULT
        self.db.execute("ALTER TABLE products ADD COLUMN stock INT NOT NULL DEFAULT 10")
        rows = self.db.query("SELECT * FROM products ORDER BY id ASC")
        self.assertEqual(rows[0]["stock"], 10)
        self.assertEqual(rows[1]["stock"], 10)

        # ADD NOT NULL without DEFAULT on non-empty table must fail
        with self.assertRaises(SchemaError):
            self.db.execute("ALTER TABLE products ADD COLUMN sku STR NOT NULL")

        # RENAME COLUMN
        self.db.execute("ALTER TABLE products RENAME COLUMN name TO title")
        row = self.db.fetchone("SELECT title FROM products WHERE id = 1")
        self.assertEqual(row, {"title": "Laptop"})

        # DROP COLUMN
        self.db.execute("ALTER TABLE products DROP COLUMN stock")
        desc = self.db.execute("DESCRIBE products")
        col_names = [c["name"] for c in desc["columns"]]
        self.assertEqual(col_names, ["id", "title"])

        # Reopen DB to verify catalog & binary records persisted cleanly
        self.db.close()
        reopened = MiniDB(self.test_dir)
        persisted = reopened.query("SELECT * FROM products ORDER BY id ASC")
        self.assertEqual(persisted, [{"id": 1, "title": "Laptop"}, {"id": 2, "title": "Mouse"}])
        reopened.close()

    # 3. Foreign Keys (REFERENCES, ON DELETE RESTRICT & ON DELETE CASCADE)
    def test_foreign_keys_restrict_and_cascade(self):
        self.db.execute("CREATE TABLE departments (dept_id INT UNIQUE, dept_name STR NOT NULL)")
        self.db.execute(
            "CREATE TABLE employees ("
            "emp_id INT UNIQUE, "
            "name STR NOT NULL, "
            "dept_id INT NOT NULL REFERENCES departments(dept_id) ON DELETE RESTRICT)"
        )
        self.db.execute(
            "CREATE TABLE tasks ("
            "task_id INT UNIQUE, "
            "emp_id INT NOT NULL REFERENCES employees(emp_id) ON DELETE CASCADE, "
            "title STR)"
        )

        self.db.execute("INSERT INTO departments VALUES (10, 'AI Engineering')")
        self.db.execute("INSERT INTO employees VALUES (101, 'Ritu', 10)")
        self.db.execute("INSERT INTO tasks VALUES (1001, 101, 'Build MiniDB')")
        self.db.execute("INSERT INTO tasks VALUES (1002, 101, 'Write Tests')")

        # Invalid parent key rejected
        with self.assertRaises(SchemaError):
            self.db.execute("INSERT INTO employees VALUES (102, 'Ghost', 999)")

        # RESTRICT prevents deleting department with active employee
        with self.assertRaises(SchemaError):
            self.db.execute("DELETE FROM departments WHERE dept_id = 10")

        # CASCADE automatically deletes child tasks when employee is deleted
        self.db.execute("DELETE FROM employees WHERE emp_id = 101")
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM tasks"), 0)

        # Now department has no referencing employees and can be deleted
        self.db.execute("DELETE FROM departments WHERE dept_id = 10")
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM departments"), 0)

    # 4. Analytics (SELECT DISTINCT, HAVING, AS Aliases)
    def test_distinct_having_and_aliases(self):
        self.db.execute("CREATE TABLE expenses (id INT UNIQUE, category STR NOT NULL, amount FLOAT)")
        self.db.executemany(
            "INSERT INTO expenses VALUES (?, ?, ?)",
            [
                (1, "Cloud", 1200.0),
                (2, "Cloud", 4500.0),
                (3, "Books", 800.0),
                (4, "Hardware", 6000.0),
            ],
        )

        # SELECT DISTINCT
        cats = self.db.query("SELECT DISTINCT category FROM expenses ORDER BY category ASC")
        self.assertEqual([c["category"] for c in cats], ["Books", "Cloud", "Hardware"])

        # GROUP BY + HAVING + AS aliases
        summary = self.db.query(
            "SELECT category, SUM(amount) AS total_spent, COUNT(*) AS tx_count "
            "FROM expenses "
            "GROUP BY category "
            "HAVING SUM(amount) > 5000 AND COUNT(*) >= 2 "
            "ORDER BY total_spent DESC"
        )
        self.assertEqual(summary, [{"category": "Cloud", "total_spent": 5700.0, "tx_count": 2}])

    # 5. Pythonic Ergonomics (Context Manager, executemany Atomicity, backup)
    def test_ergonomics_executemany_atomicity_and_backup(self):
        backup_dir = tempfile.mkdtemp(prefix="minidb_backup_")
        try:
            with MiniDB(self.test_dir) as db:
                db.execute("CREATE TABLE scores (student_id INT UNIQUE, marks INT)")
                db.executemany(
                    "INSERT INTO scores VALUES (?, ?)",
                    [(1, 95), (2, 88), (3, 91)],
                )
                self.assertEqual(db.scalar("SELECT MAX(marks) FROM scores"), 95)

                # Atomic rollback on batch failure (duplicate student_id=2)
                with self.assertRaises(SchemaError):
                    db.executemany(
                        "INSERT INTO scores VALUES (?, ?)",
                        [(4, 77), (2, 60)],
                    )
                # Row (4, 77) was rolled back atomically
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM scores"), 3)

                # Consistent point-in-time backup
                res = db.backup(backup_dir, overwrite=True)
                self.assertIn("scores.mdb", res["files_copied"])

            # Restore and verify backup database independently
            with MiniDB(backup_dir) as restored_db:
                self.assertEqual(restored_db.scalar("SELECT COUNT(*) FROM scores"), 3)
                self.assertEqual(
                    restored_db.fetchone("SELECT * FROM scores WHERE student_id = ?", (2,)),
                    {"student_id": 2, "marks": 88},
                )
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()