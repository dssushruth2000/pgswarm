from __future__ import annotations

import statistics
import time
import uuid
from typing import Any

from psycopg import Connection, sql

from optimizer.config import Settings, load_settings
from optimizer.contracts import (
    BenchDelta,
    BenchResult,
    BenchSuiteResult,
    Candidate,
    ParsedSchema,
)
from optimizer.sandbox import clone_from_template, connect, drop_clone


class BenchError(RuntimeError):
    """Raised when the measurement harness cannot produce a valid result."""


def _median(values: list[float]) -> float:
    if not values:
        raise BenchError("median of empty series")
    return float(statistics.median(values))


def assert_no_hypothetical_indexes(conn: Connection[Any]) -> None:
    """
    Guard: EXPLAIN ANALYZE silently ignores hypopg indexes.
    Refuse ANALYZE entirely while any hypothetical index exists on this connection.
    """
    row = conn.execute(
        """
        SELECT count(*) FROM pg_extension WHERE extname = 'hypopg'
        """
    ).fetchone()
    if not row or row[0] == 0:
        return
    try:
        n = conn.execute("SELECT count(*) FROM hypopg_list_indexes()").fetchone()[0]
    except Exception:
        return
    if n > 0:
        raise BenchError(
            "refusing EXPLAIN ANALYZE while hypothetical indexes are active on this connection"
        )


def _walk_plan(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    for child in node.get("Plans") or []:
        nodes.extend(_walk_plan(child))
    return nodes


def _extract_scan_metrics(plan_json: list[dict[str, Any]]) -> tuple[str, int | None, int]:
    """Return (primary scan node type, heap_fetches, shared_hit_blocks)."""
    root = plan_json[0]["Plan"]
    all_nodes = _walk_plan(root)
    shared_hit = int(root.get("Shared Hit Blocks") or 0)

    scan_priority = (
        "Index Only Scan",
        "Index Scan",
        "Bitmap Index Scan",
        "Bitmap Heap Scan",
        "Seq Scan",
    )
    chosen: dict[str, Any] | None = None
    for want in scan_priority:
        for n in all_nodes:
            if n.get("Node Type") == want:
                chosen = n
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = root

    node_type = str(chosen.get("Node Type") or "Unknown")
    heap_fetches: int | None
    if "Heap Fetches" in chosen:
        heap_fetches = int(chosen["Heap Fetches"])
    elif node_type == "Index Only Scan":
        heap_fetches = 0
    else:
        heap_fetches = None
    return node_type, heap_fetches, shared_hit


def _set_autovacuum(conn: Connection[Any], table: str, enabled: bool) -> None:
    flag = sql.SQL("true") if enabled else sql.SQL("false")
    conn.execute(
        sql.SQL("ALTER TABLE {} SET (autovacuum_enabled = {})").format(
            sql.Identifier(table),
            flag,
        )
    )


def _measure_reads(
    conn: Connection[Any],
    query: str,
    *,
    read_runs: int,
    statement_timeout_ms: int,
) -> tuple[float, str, int | None, int]:
    """
    Read phase only. Must complete before any write measurement on this clone.
    One discarded warmup, then `read_runs` timed EXPLAIN (ANALYZE, BUFFERS).
    """
    assert_no_hypothetical_indexes(conn)
    conn.execute(f"SET statement_timeout = '{statement_timeout_ms}ms'")

    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

    # Discarded warmup — warms buffer cache; timing ignored.
    conn.execute(explain_sql).fetchone()

    times_ms: list[float] = []
    plan_node = "Unknown"
    heap_fetches: int | None = None
    shared_hit = 0

    for _ in range(read_runs):
        assert_no_hypothetical_indexes(conn)
        row = conn.execute(explain_sql).fetchone()
        plan_json = row[0]
        # Planning+execution time from the root wrapper when present
        root_meta = plan_json[0]
        exec_ms = float(root_meta.get("Execution Time") or 0.0)
        times_ms.append(exec_ms)
        plan_node, heap_fetches, shared_hit = _extract_scan_metrics(plan_json)

    return _median(times_ms), plan_node, heap_fetches, shared_hit


def _insert_batch_sql(schema: ParsedSchema, batch_size: int) -> sql.Composed:
    """Synthetic inserts for WAL measurement — schema-driven, not user DDL."""
    cols = schema.columns
    select_parts: list[sql.Composable] = []
    for c in cols:
        t = c.data_type.lower().split("(")[0].strip()
        name = c.name.lower()
        if name == "id" or t in {"serial", "bigserial"}:
            # Offset into a high id range to avoid colliding with seeded PKs if any.
            select_parts.append(
                sql.SQL("(1000000000 + g.i)::int AS {}").format(sql.Identifier(c.name))
            )
        elif t in {"int", "integer", "int2", "int4", "int8", "smallint", "bigint"}:
            select_parts.append(
                sql.SQL("(g.i % 10000)::int AS {}").format(sql.Identifier(c.name))
            )
        elif t in {"numeric", "decimal", "real", "float4", "float8", "double precision"}:
            select_parts.append(
                sql.SQL("(g.i % 100)::numeric AS {}").format(sql.Identifier(c.name))
            )
        elif t in {"bool", "boolean"}:
            select_parts.append(sql.SQL("true AS {}").format(sql.Identifier(c.name)))
        elif "timestamp" in t:
            select_parts.append(
                sql.SQL("now() AS {}").format(sql.Identifier(c.name))
            )
        elif t == "date":
            select_parts.append(
                sql.SQL("CURRENT_DATE AS {}").format(sql.Identifier(c.name))
            )
        else:
            # Prefer pending so write path still touches skewed status indexes if any.
            select_parts.append(
                sql.SQL("'pending'::text AS {}").format(sql.Identifier(c.name))
            )

    return sql.SQL(
        "INSERT INTO {} ({}) SELECT {} FROM generate_series(1, {}) AS g(i)"
    ).format(
        sql.Identifier(schema.table_name),
        sql.SQL(", ").join(sql.Identifier(c.name) for c in cols),
        sql.SQL(", ").join(select_parts),
        sql.Literal(batch_size),
    )


def _measure_writes(
    conn: Connection[Any],
    schema: ParsedSchema,
    *,
    write_runs: int,
    insert_batch_size: int,
) -> tuple[int, float]:
    """
    Write phase only — after reads are finished.
    Autovacuum is disabled for this phase and always re-enabled afterward so the
    next candidate's reads see a normal visibility map.
    """
    table = schema.table_name
    wal_samples: list[int] = []
    insert_ms_samples: list[float] = []
    insert_sql = _insert_batch_sql(schema, insert_batch_size)

    _set_autovacuum(conn, table, False)
    try:
        for _ in range(write_runs):
            conn.execute("CHECKPOINT")
            # insert_lsn, not flush lsn: with synchronous_commit=off,
            # pg_current_wal_lsn() may not advance until a flush, undercounting WAL.
            start_lsn = conn.execute(
                "SELECT pg_current_wal_insert_lsn()"
            ).fetchone()[0]
            t0 = time.perf_counter()
            conn.execute(insert_sql)
            insert_ms_samples.append((time.perf_counter() - t0) * 1000.0)
            end_lsn = conn.execute(
                "SELECT pg_current_wal_insert_lsn()"
            ).fetchone()[0]
            delta = conn.execute(
                "SELECT pg_wal_lsn_diff(%s, %s)", (end_lsn, start_lsn)
            ).fetchone()[0]
            wal_samples.append(int(delta))
    finally:
        # Must re-enable before any subsequent candidate's read phase.
        _set_autovacuum(conn, table, True)

    return int(_median([float(w) for w in wal_samples])), _median(insert_ms_samples)


def _index_bytes(conn: Connection[Any], index_name: str | None) -> int:
    if not index_name:
        return 0
    row = conn.execute(
        "SELECT pg_relation_size(%s::regclass)", (index_name,)
    ).fetchone()
    return int(row[0]) if row else 0


def _created_index_name(conn: Connection[Any], table: str, before: set[str]) -> str | None:
    rows = conn.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE tablename = %s
        """,
        (table,),
    ).fetchall()
    after = {r[0] for r in rows}
    created = after - before
    if not created:
        return None
    # Prefer the newly created one; if multiple, take sorted for stability.
    return sorted(created)[0]


BASELINE_CANDIDATE = Candidate(
    archetype="min_write",
    index_sql=None,
    rewritten_query=None,
    rationale="baseline: no new index",
    proposed_by="system",
)


def benchmark_candidate(
    schema: ParsedSchema,
    query: str,
    candidate: Candidate,
    *,
    settings: Settings | None = None,
    clone_name: str | None = None,
) -> BenchResult:
    """
    Full per-candidate sequence. Order is load-bearing:
    clone → optional CREATE INDEX → VACUUM ANALYZE → reads → writes → sizes → drop.
    """
    s = settings or load_settings()
    name = clone_name or f"bench_{uuid.uuid4().hex[:12]}"
    clone_from_template(s, clone_name=name)

    try:
        with connect(s, name, autocommit=True) as conn:
            conn.execute(f"SET statement_timeout = '{s.statement_timeout_ms}ms'")

            index_name: str | None = None
            before_indexes: set[str] = set()
            if candidate.index_sql:
                rows = conn.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = %s",
                    (schema.table_name,),
                ).fetchall()
                before_indexes = {r[0] for r in rows}
                # Plain CREATE INDEX — reconstructed/agent SQL only, never CONCURRENTLY here.
                conn.execute(candidate.index_sql)
                index_name = _created_index_name(conn, schema.table_name, before_indexes)
                conn.execute(
                    sql.SQL("VACUUM ANALYZE {}").format(
                        sql.Identifier(schema.table_name)
                    )
                )
            else:
                # Baseline still refreshes stats on the clone for a fair read.
                conn.execute(
                    sql.SQL("VACUUM ANALYZE {}").format(
                        sql.Identifier(schema.table_name)
                    )
                )

            # --- Read phase must finish before any writes ---
            read_ms, plan_node, heap_fetches, shared_hit = _measure_reads(
                conn,
                query,
                read_runs=s.read_runs,
                statement_timeout_ms=s.statement_timeout_ms,
            )

            # --- Write phase after reads; autovacuum off only here ---
            wal_bytes, insert_ms = _measure_writes(
                conn,
                schema,
                write_runs=s.write_runs,
                insert_batch_size=s.insert_batch_size,
            )

            idx_bytes = _index_bytes(conn, index_name)

            return BenchResult(
                candidate=candidate,
                read_ms_median=read_ms,
                plan_node=plan_node,
                heap_fetches=heap_fetches,
                shared_hit_blocks=shared_hit,
                wal_bytes=wal_bytes,
                insert_ms_median=insert_ms,
                index_bytes=idx_bytes,
            )
    finally:
        drop_clone(s, clone_name=name)


def deltas_against_baseline(
    baseline: BenchResult, results: list[BenchResult]
) -> list[BenchDelta]:
    """Report every figure as a delta against the no-index baseline."""
    out: list[BenchDelta] = []
    for r in results:
        out.append(
            BenchDelta(
                candidate=r.candidate,
                read_ms_delta=r.read_ms_median - baseline.read_ms_median,
                wal_bytes_delta=r.wal_bytes - baseline.wal_bytes,
                insert_ms_delta=r.insert_ms_median - baseline.insert_ms_median,
                index_bytes=r.index_bytes,
                plan_node=r.plan_node,
                heap_fetches=r.heap_fetches,
                shared_hit_blocks=r.shared_hit_blocks,
                absolute=r,
            )
        )
    return out


def run_benchmark_suite(
    schema: ParsedSchema,
    query: str,
    candidates: list[Candidate],
    *,
    settings: Settings | None = None,
) -> BenchSuiteResult:
    """Baseline first (no new index), then each candidate. Results include deltas."""
    s = settings or load_settings()
    baseline = benchmark_candidate(
        schema, query, BASELINE_CANDIDATE, settings=s
    )
    results: list[BenchResult] = []
    for cand in candidates:
        results.append(benchmark_candidate(schema, query, cand, settings=s))
    return BenchSuiteResult(
        baseline=baseline,
        results=results,
        deltas=deltas_against_baseline(baseline, results),
    )
