import os
import shutil
import subprocess
import sys
import tempfile
from minidb import MiniDB


def run_demo():
    demo_dir = tempfile.mkdtemp(prefix="minidb_demo_")
    try:
        print(f"=== 1. Initializing MiniDB & Creating Schema ({demo_dir}) ===")
        db = MiniDB(demo_dir)
        db.execute("CREATE TABLE employees (id INT UNIQUE, name STR NOT NULL, dept STR, salary FLOAT)")

        sample_data = [
            (1, "Aarav", "Engineering", 120000.0),
            (2, "Diya", "Engineering", 135000.0),
            (3, "Rohan", "Sales", 85000.0),
            (4, "Meera", "Sales", 92000.0),
            (5, "Kabir", "Product", 110000.0),
        ]
        for emp in sample_data:
            db.execute(f"INSERT INTO employees VALUES ({emp[0]}, '{emp[1]}', '{emp[2]}', {emp[3]})")

        # Add 95 more synthetic rows so the query planner activates index scans
        with db.transaction():
            for i in range(6, 101):
                dept = ["Engineering", "Sales", "Product", "HR"][i % 4]
                sal = 60000.0 + (i * 500)
                db.execute(f"INSERT INTO employees VALUES ({i}, 'Emp_{i}', '{dept}', {sal})")

        print("\n=== 2. Complex Query (Filter + Group By + Aggregations + Order By) ===")
        agg_res = db.execute(
            "SELECT dept, COUNT(id), AVG(salary), MAX(salary) FROM employees "
            "WHERE salary >= 80000.0 GROUP BY dept ORDER BY AVG(salary) DESC"
        )
        for row in agg_res["rows"]:
            print(" ", row)
        print("  Stats:", agg_res["stats"])

        print("\n=== 3. Non-Indexed vs. Indexed Performance Comparison ===")
        q = "SELECT id, name, salary FROM employees WHERE salary > 108000.0"
        res_no_idx = db.execute(q)
        print("  Before Index ->", res_no_idx["stats"])

        db.execute("CREATE INDEX idx_salary ON employees (salary) USING BTREE")
        res_with_idx = db.execute(q)
        print("  After Index  ->", res_with_idx["stats"])

        print("\n=== 4. Simulating Hard Process Crash Mid-Transaction ===")
        crash_code = (
            "import os; "
            "from minidb import MiniDB; "
            f"db = MiniDB({demo_dir!r}); "
            "db.begin(); "
            "db.execute(\"INSERT INTO employees VALUES (999, 'GhostUser', 'Engineering', 999999.0)\"); "
            "print('  [Child Process] Wrote uncommitted record to WAL... crashing NOW!'); "
            "os._exit(1)"
        )
        subprocess.run([sys.executable, "-c", crash_code])

        print("\n=== 5. Restarting MiniDB & Verifying Crash Recovery ===")
        recovered_db = MiniDB(demo_dir)
        print("  Recovery Report:", recovered_db.last_recovery_stats)
        check_ghost = recovered_db.execute("SELECT * FROM employees WHERE id = 999")
        check_valid = recovered_db.execute("SELECT COUNT(id) FROM employees")
        print("  GhostUser rows found (must be 0):", len(check_ghost["rows"]))
        print("  Total consistent rows preserved (must be 100):", check_valid["rows"][0]["COUNT(id)"])
    finally:
        shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    run_demo()