import os
import json
import zlib
from typing import Any, Dict, List, Optional, Tuple


class WriteAheadLog:
    """
    Append-only Write-Ahead Log (WAL) ensuring atomicity and crash recovery.
    Each entry is formatted as: <CRC32_HEX>|<JSON_PAYLOAD>\n
    """

    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        os.makedirs(os.path.dirname(os.path.abspath(self.wal_path)), exist_ok=True)
        if not os.path.exists(self.wal_path):
            with open(self.wal_path, "w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())

    def append(
        self,
        tx_id: str,
        op: str,
        table: Optional[str] = None,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "tx_id": tx_id,
            "op": op,
            "table": table,
            "before": before,
            "after": after,
            "schema": schema,
        }
        raw = json.dumps(entry, separators=(",", ":"))
        crc = f"{zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"
        line = f"{crc}|{raw}\n"

        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_committed_operations(self) -> Tuple[List[Dict[str, Any]], int]:
        if not os.path.exists(self.wal_path):
            return [], 0

        entries: List[Dict[str, Any]] = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "|" not in line:
                    continue
                crc_hex, raw = line.split("|", 1)
                expected_crc = f"{zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"
                if crc_hex != expected_crc:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

        committed_tx_ids = set()
        all_begun_tx_ids = set()

        for entry in entries:
            tx_id = entry["tx_id"]
            op = entry["op"]
            if op == "BEGIN":
                all_begun_tx_ids.add(tx_id)
            elif op == "COMMIT":
                committed_tx_ids.add(tx_id)

        uncommitted_count = len(all_begun_tx_ids - committed_tx_ids)

        committed_ops = [
            e for e in entries
            if e["tx_id"] in committed_tx_ids and e["op"] not in ("BEGIN", "COMMIT", "ROLLBACK")
        ]
        return committed_ops, uncommitted_count

    def truncate(self) -> None:
        with open(self.wal_path, "w", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())