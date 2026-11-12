import json
from minidb import MiniDB, MiniDBError

def main():
    db = MiniDB("./manual_demo_db")
    print("MiniDB Interactive Shell Ready! (Data saved in ./manual_demo_db)")
    print("Recovery Stats on Boot:", db.last_recovery_stats)
    print("Commands: Type any query, or BEGIN | COMMIT | ROLLBACK | EXIT\n")

    while True:
        try:
            raw = input("minidb> ").strip()
            if not raw:
                continue
            cmd = raw.upper().rstrip(";")

            if cmd == "EXIT":
                break
            elif cmd == "BEGIN":
                tx_id = db.begin()
                print(f"Transaction started: {tx_id}")
            elif cmd == "COMMIT":
                db.commit()
                print("Transaction committed to disk.")
            elif cmd == "ROLLBACK":
                db.rollback()
                print("Transaction rolled back.")
            else:
                result = db.execute(raw)
                print(json.dumps(result, indent=2))
        except MiniDBError as e:
            print(f"[MiniDB Error] {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[Unexpected Error] {e}")

if __name__ == "__main__":
    main()