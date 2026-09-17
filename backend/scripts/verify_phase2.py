"""Phase 2 acceptance: seed template, clone quickly, check pg_stats vs hints."""

from __future__ import annotations

import sys
import time

from optimizer.config import load_settings
from optimizer.contracts import ColumnHint
from optimizer.ingest import parse_ddl, parse_query
from optimizer.sandbox import clone_from_template, connect, drop_clone
from optimizer.seeder import seed_template

DDL = """
CREATE TABLE orders (
  id integer NOT NULL,
  user_id integer NOT NULL,
  status text,
  amount numeric,
  created_at timestamptz
);
"""

QUERY = """
SELECT id, user_id, status, amount
FROM orders
WHERE user_id = 42 AND status = 'pending' AND created_at >= '2026-01-01'
"""

HINTS = [
    ColumnHint(name="user_id", distinct_ratio=0.02, null_fraction=0.0),
    ColumnHint(
        name="status",
        null_fraction=0.01,
        skew={"completed": 0.90, "pending": 0.08, "cancelled": 0.02},
    ),
    ColumnHint(name="created_at", null_fraction=0.0),
]


def main() -> int:
    settings = load_settings()
    schema = parse_ddl(DDL)
    query = parse_query(QUERY)

    print(f"Seeding {settings.seed_rows} rows into {settings.sandbox.seeded_db}...")
    t0 = time.perf_counter()
    seed_template(schema, hints=HINTS, query=query, settings=settings)
    seed_s = time.perf_counter() - t0
    print(f"Seed complete in {seed_s:.1f}s")

    clone_name = "bench_phase2_clone"
    print(f"Cloning to {clone_name}...")
    t1 = time.perf_counter()
    clone_from_template(settings, clone_name=clone_name)
    clone_s = time.perf_counter() - t1
    print(f"Clone complete in {clone_s:.3f}s")

    try:
        with connect(settings, clone_name) as conn:
            n = conn.execute("SELECT count(*) FROM orders").fetchone()[0]
            print(f"Row count: {n}")

            # user_id distinct ratio ~ 0.02
            nunique = conn.execute("SELECT count(DISTINCT user_id) FROM orders").fetchone()[0]
            ratio = nunique / n
            print(f"user_id distinct_ratio measured={ratio:.4f} (hint=0.02)")

            # status most-common should be completed-dominated
            top = conn.execute(
                """
                SELECT status, count(*)::float / (SELECT count(*) FROM orders) AS frac
                FROM orders
                WHERE status IS NOT NULL
                GROUP BY status
                ORDER BY frac DESC
                LIMIT 1
                """
            ).fetchone()
            print(f"status top value={top[0]!r} frac={top[1]:.3f} (hint completed=0.90)")

            stats = conn.execute(
                """
                SELECT attname, n_distinct,
                       most_common_vals::text,
                       null_frac
                FROM pg_stats
                WHERE tablename = 'orders'
                  AND attname IN ('user_id', 'status')
                ORDER BY attname
                """
            ).fetchall()
            for row in stats:
                print(f"pg_stats {row[0]}: n_distinct={row[1]} null_frac={row[3]} mcv={row[2]}")

        # Gate checks
        ok = True
        if clone_s > 10.0:
            print(f"FAIL: clone took {clone_s:.3f}s (expected a few seconds on tmpfs)")
            ok = False
        else:
            print(f"PASS: clone in {clone_s:.3f}s")

        # distinct_ratio directionally near 0.02 (allow generous band — planner stats are approximate)
        if not (0.005 <= ratio <= 0.05):
            print(f"FAIL: user_id distinct_ratio {ratio:.4f} not near hint 0.02")
            ok = False
        else:
            print("PASS: user_id distinct_ratio matches hint directionally")

        if top[0] != "completed" or top[1] < 0.7:
            print(f"FAIL: status skew not honored ({top})")
            ok = False
        else:
            print("PASS: status skew matches hint")

        return 0 if ok else 1
    finally:
        drop_clone(settings, clone_name=clone_name)


if __name__ == "__main__":
    sys.exit(main())
