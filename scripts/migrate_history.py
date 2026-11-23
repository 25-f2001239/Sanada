"""Idempotent history privacy migration: verifies user_jobs schema and cleans orphaned rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auth


def run_migration() -> dict[str, int]:
    auth.init_db()
    with auth._conn() as con:
        orphans = con.execute(
            "SELECT job_id FROM user_jobs WHERE user_id IS NULL OR user_id NOT IN (SELECT id FROM users)"
        ).fetchall()
        orphan_count = len(orphans)
        if orphan_count:
            con.execute(
                "DELETE FROM user_jobs WHERE user_id IS NULL OR user_id NOT IN (SELECT id FROM users)"
            )
        total = int(con.execute("SELECT COUNT(*) FROM user_jobs").fetchone()[0])
    return {"orphaned_deleted": orphan_count, "remaining_rows": total}


if __name__ == "__main__":
    res = run_migration()
    print(f"History migration complete: orphaned_deleted={res['orphaned_deleted']}, remaining_rows={res['remaining_rows']}")
