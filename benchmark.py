import os
import shutil
import tempfile
from minidb import MiniDB


def run_benchmarks():
    sizes = [500, 2500, 10000]
    print(f"{'Dataset Size':<14} | {'Scan Type':<18} | {'Records Scanned':<16} | {'Time (ms)':<10}")
    print("-" * 68)

    for n in sizes:
        bench_dir = tempfile.mkdtemp(prefix="minidb_bench_")
        try:
            db = MiniDB(bench_dir)
            db.execute("CREATE TABLE orders (id INT UNIQUE, customer_id INT, amount FLOAT)")

            with db.transaction():
                for i in range(1, n + 1):
                    db.execute(f"INSERT INTO orders VALUES ({i}, {i % 200}, {float(i * 1.5)})")

            query = f"SELECT id, amount FROM orders WHERE amount > {float((n - 20) * 1.5)}"

            # 1. Without Index
            r1 = db.execute(query)
            s1 = r1["stats"]
            print(f"{n:<14} | {'FULL_TABLE_SCAN':<18} | {s1['records_scanned']:<16} | {s1['execution_time_ms']:<10}")

            # 2. With B-Tree Range Index
            db.execute("CREATE INDEX idx_amt ON orders (amount) USING BTREE")
            r2 = db.execute(query)
            s2 = r2["stats"]
            print(f"{n:<14} | {'BTREE_INDEX_SCAN':<18} | {s2['records_scanned']:<16} | {s2['execution_time_ms']:<10}")
            print("-" * 68)
        finally:
            shutil.rmtree(bench_dir, ignore_errors=True)


if __name__ == "__main__":
    run_benchmarks()