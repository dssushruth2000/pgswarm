"""Phase 4 crude viewer: one HTML page, one fetch, unstyled table. No polish."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from optimizer.bench import run_benchmark_suite
from optimizer.config import load_settings
from optimizer.contracts import Candidate, ColumnHint
from optimizer.ingest import parse_ddl, parse_query
from optimizer.sandbox import connect
from optimizer.seeder import seed_template

# Same demo fixture as phase 3 — hardcoded so the viewer needs no input yet.
_DDL = """
CREATE TABLE orders (
  id integer NOT NULL,
  user_id integer NOT NULL,
  status text,
  amount numeric,
  created_at timestamptz
);
"""

_QUERY = """
SELECT user_id, status
FROM orders
WHERE status = 'pending'
"""

_HINTS = [
    ColumnHint(name="user_id", distinct_ratio=0.02, null_fraction=0.0),
    ColumnHint(
        name="status",
        null_fraction=0.0,
        skew={"completed": 0.99, "pending": 0.01},
    ),
    ColumnHint(name="created_at", null_fraction=0.0),
]

_COVERING = Candidate(
    archetype="max_read",
    index_sql=(
        "CREATE INDEX bench_covering_pending "
        "ON orders (status) INCLUDE (user_id) "
        "WHERE status = 'pending'"
    ),
    rewritten_query=None,
    rationale="partial covering index for pending rows (demo skew)",
    proposed_by="phase4_viewer",
)

_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="index-optimizer crude viewer")


def _template_exists(settings) -> bool:
    with connect(settings, settings.sandbox.admin_db, autocommit=True) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.sandbox.seeded_db,),
        ).fetchone()
        return row is not None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/bench")
def bench() -> JSONResponse:
    """Run baseline + one covering-index candidate; return suite JSON with deltas."""
    settings = load_settings()
    schema = parse_ddl(_DDL)
    query = parse_query(_QUERY)
    if not _template_exists(settings):
        seed_template(schema, hints=_HINTS, query=query, settings=settings)
    suite = run_benchmark_suite(
        schema,
        query.reconstructed_sql,
        [_COVERING],
        settings=settings,
    )
    return JSONResponse(suite.model_dump())
