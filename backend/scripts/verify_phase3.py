"""
Phase 3 acceptance — rerunnable.

Asserts:
  1. Two runs of the same covering-index candidate land within 15% read median.
  2. Covering index produces Index Only Scan with zero heap fetches.
  3. WAL bytes are measurably higher with the index than without (baseline).

Fixture uses status skew 99% completed / 1% pending (demo scenario).
Baseline is measured first; reported figures are deltas against it.
"""

from __future__ import annotations

import sys

from optimizer.bench import (
    BASELINE_CANDIDATE,
    benchmark_candidate,
    deltas_against_baseline,
    run_benchmark_suite,
)
from optimizer.config import load_settings
from optimizer.contracts import Candidate, ColumnHint
from optimizer.ingest import parse_ddl, parse_query
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

# Cover enough rows for a stable median while remaining an Index Only Scan.
# status = 'pending' is ~1% of the table under the demo skew.
QUERY = """
SELECT user_id, status
FROM orders
WHERE status = 'pending'
"""

HINTS = [
    ColumnHint(name="user_id", distinct_ratio=0.02, null_fraction=0.0),
    ColumnHint(
        name="status",
        null_fraction=0.0,
        skew={"completed": 0.99, "pending": 0.01},
    ),
    ColumnHint(name="created_at", null_fraction=0.0),
]

COVERING_INDEX = Candidate(
    archetype="max_read",
    index_sql=(
        "CREATE INDEX bench_covering_pending "
        "ON orders (status) INCLUDE (user_id) "
        "WHERE status = 'pending'"
    ),
    rewritten_query=None,
    rationale="partial covering index for pending rows (demo skew)",
    proposed_by="phase3_fixture",
)


def _within_pct(a: float, b: float, pct: float = 0.15) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= pct


def main() -> int:
    settings = load_settings()
    schema = parse_ddl(DDL)
    query = parse_query(QUERY)

    print("Seeding fixture (status 99% completed / 1% pending)...")
    seed_template(schema, hints=HINTS, query=query, settings=settings)

    print("Suite run 1: baseline first, then covering index...")
    suite1 = run_benchmark_suite(
        schema, query.reconstructed_sql, [COVERING_INDEX], settings=settings
    )
    d1 = suite1.deltas[0]
    r1 = suite1.results[0]
    b1 = suite1.baseline

    print(
        f"  baseline read_ms={b1.read_ms_median:.3f} wal={b1.wal_bytes} "
        f"plan={b1.plan_node}"
    )
    print(
        f"  candidate read_ms={r1.read_ms_median:.3f} wal={r1.wal_bytes} "
        f"plan={r1.plan_node} heap_fetches={r1.heap_fetches}"
    )
    print(
        f"  deltas: read_ms={d1.read_ms_delta:.3f} "
        f"wal_bytes={d1.wal_bytes_delta} insert_ms={d1.insert_ms_delta:.3f}"
    )

    print("Suite run 2: repeat candidate for stability...")
    # Baseline again so each suite is self-contained; compare candidate read medians.
    suite2 = run_benchmark_suite(
        schema, query.reconstructed_sql, [COVERING_INDEX], settings=settings
    )
    r2 = suite2.results[0]
    print(
        f"  candidate read_ms={r2.read_ms_median:.3f} plan={r2.plan_node} "
        f"heap_fetches={r2.heap_fetches}"
    )

    ok = True

    # Gate 1: read median stability within 15%
    if not _within_pct(r1.read_ms_median, r2.read_ms_median, 0.15):
        print(
            f"FAIL: read medians differ by >15%: "
            f"{r1.read_ms_median:.3f} vs {r2.read_ms_median:.3f}"
        )
        ok = False
    else:
        print("PASS: two runs within 15% read median")

    # Gate 2: Index Only Scan, zero heap fetches
    if r1.plan_node != "Index Only Scan" or r2.plan_node != "Index Only Scan":
        print(
            f"FAIL: expected Index Only Scan, got {r1.plan_node!r} / {r2.plan_node!r}"
        )
        ok = False
    elif r1.heap_fetches != 0 or r2.heap_fetches != 0:
        print(
            f"FAIL: expected zero heap fetches, got "
            f"{r1.heap_fetches} / {r2.heap_fetches}"
        )
        ok = False
    else:
        print("PASS: Index Only Scan with zero heap fetches")

    # Gate 3: WAL higher with index than without (delta > 0)
    if d1.wal_bytes_delta <= 0 or suite2.deltas[0].wal_bytes_delta <= 0:
        print(
            f"FAIL: WAL delta not higher with index "
            f"(run1={d1.wal_bytes_delta}, run2={suite2.deltas[0].wal_bytes_delta})"
        )
        ok = False
    else:
        print(
            f"PASS: WAL delta higher with index "
            f"(+{d1.wal_bytes_delta} / +{suite2.deltas[0].wal_bytes_delta} bytes)"
        )

    # Sanity: deltas helper matches baseline-first contract
    recomputed = deltas_against_baseline(b1, [r1])[0]
    assert recomputed.wal_bytes_delta == d1.wal_bytes_delta
    assert suite1.baseline.candidate.proposed_by == BASELINE_CANDIDATE.proposed_by

    if ok:
        print("Phase 3 acceptance passed")
        return 0
    print("Phase 3 acceptance failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
