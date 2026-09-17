from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from psycopg import Connection, sql

from optimizer.config import Settings, load_settings
from optimizer.contracts import ColumnDef, ColumnHint, ParsedQuery, ParsedSchema
from optimizer.sandbox import (
    connect,
    create_database,
    mark_as_template,
    terminate_backends,
    unmark_template,
)

UTC = timezone.utc

# Default imbalance for low-cardinality text when no skew hint is given.
_DEFAULT_STATUS_SKEW: dict[str, float] = {
    "completed": 0.90,
    "pending": 0.08,
    "cancelled": 0.02,
}


def _hint_map(hints: Sequence[ColumnHint]) -> dict[str, ColumnHint]:
    return {h.name.lower(): h for h in hints}


def _normalize_type(data_type: str) -> str:
    return data_type.lower().split("(")[0].strip()


def _is_int_type(data_type: str) -> bool:
    t = _normalize_type(data_type)
    return t in {"int", "integer", "int2", "int4", "int8", "smallint", "bigint", "serial", "bigserial"}


def _is_float_type(data_type: str) -> bool:
    t = _normalize_type(data_type)
    return t in {"real", "float4", "float8", "double", "double precision", "numeric", "decimal"}


def _is_bool_type(data_type: str) -> bool:
    return _normalize_type(data_type) in {"bool", "boolean"}


def _is_text_type(data_type: str) -> bool:
    t = _normalize_type(data_type)
    return t in {"text", "varchar", "character", "character varying", "char", "name", "citext"}


def _is_timestamptz_type(data_type: str) -> bool:
    t = _normalize_type(data_type)
    return t in {"timestamptz", "timestamp with time zone"}


def _is_timestamp_type(data_type: str) -> bool:
    t = _normalize_type(data_type)
    return t in {"timestamp", "timestamp without time zone"} or _is_timestamptz_type(data_type)


def _is_date_type(data_type: str) -> bool:
    return _normalize_type(data_type) == "date"


def _skewed_choice(rng: random.Random, skew: Mapping[str, float]) -> str:
    keys = list(skew.keys())
    weights = [skew[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _value_for_column(
    col: ColumnDef,
    *,
    row_i: int,
    n_rows: int,
    hint: ColumnHint | None,
    rng: random.Random,
    now: datetime,
) -> Any:
    null_fraction = hint.null_fraction if hint and hint.null_fraction is not None else 0.0
    if null_fraction > 0 and rng.random() < null_fraction:
        return None

    distinct_ratio = hint.distinct_ratio if hint and hint.distinct_ratio is not None else None

    if _is_int_type(col.data_type):
        if distinct_ratio is not None and distinct_ratio < 1.0:
            domain = max(1, int(n_rows * distinct_ratio))
            return rng.randint(1, domain)
        # sequential primary-key style by default
        return row_i

    if _is_float_type(col.data_type):
        return rng.random() * 1000.0

    if _is_bool_type(col.data_type):
        # Realistic imbalance, not 50/50
        return rng.random() < 0.85

    if _is_timestamp_type(col.data_type) or _is_date_type(col.data_type):
        # Spread over ~2 years with a denser recent tail
        if rng.random() < 0.35:
            age_days = rng.expovariate(1 / 14.0)  # denser near now
            age_days = min(age_days, 60.0)
        else:
            age_days = rng.uniform(0, 730)
        ts = now - timedelta(days=age_days, seconds=rng.randint(0, 86400))
        if _is_date_type(col.data_type):
            return ts.date()
        if _is_timestamptz_type(col.data_type):
            return ts.replace(tzinfo=UTC)
        return ts.replace(tzinfo=None)

    if _is_text_type(col.data_type):
        if hint and hint.skew:
            return _skewed_choice(rng, hint.skew)
        # Low-cardinality default for status-like columns; otherwise unique-ish labels
        name_l = col.name.lower()
        if any(tok in name_l for tok in ("status", "state", "kind", "type")):
            return _skewed_choice(rng, _DEFAULT_STATUS_SKEW)
        if distinct_ratio is not None and distinct_ratio < 0.05:
            domain = max(2, int(n_rows * distinct_ratio))
            return f"v{rng.randint(1, domain)}"
        return f"{col.name}_{row_i}"

    # Fallback: text representation
    return f"{col.name}_{row_i}"


def _iter_bulk_rows(
    schema: ParsedSchema,
    *,
    n_rows: int,
    hints: Sequence[ColumnHint],
    seed: int,
) -> Iterator[tuple[Any, ...]]:
    rng = random.Random(seed)
    hint_by = _hint_map(hints)
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    cols = schema.columns
    for i in range(1, n_rows + 1):
        yield tuple(
            _value_for_column(
                c,
                row_i=i,
                n_rows=n_rows,
                hint=hint_by.get(c.name.lower()),
                rng=rng,
                now=now,
            )
            for c in cols
        )


def _adversarial_rows(
    schema: ParsedSchema,
    query: ParsedQuery | None,
    *,
    n_adv: int,
    hints: Sequence[ColumnHint],
    seed: int,
) -> list[tuple[Any, ...]]:
    """Roughly 1% of the table: nulls, boundaries, duplicates, tz midnights."""
    if n_adv <= 0:
        return []
    rng = random.Random(seed + 99)
    hint_by = _hint_map(hints)
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    cols = schema.columns
    filter_join = set()
    if query:
        filter_join = {c.lower() for c in (*query.filter_columns, *query.join_columns)}

    boundaries_by_col: dict[str, list[Any]] = {}
    if query:
        for b in query.range_boundaries:
            boundaries_by_col.setdefault(b.column.lower(), []).append(b.value)

    rows: list[tuple[Any, ...]] = []
    # Base template values
    base = list(
        next(
            _iter_bulk_rows(schema, n_rows=1, hints=hints, seed=seed)
        )
    )

    # 1) Nulls in every filtered/joined column (and nullable cols generally)
    null_target = filter_join or {c.name.lower() for c in cols if c.nullable}
    for _ in range(max(1, n_adv // 4)):
        row = list(base)
        for i, c in enumerate(cols):
            if c.name.lower() in null_target and c.nullable:
                row[i] = None
            else:
                row[i] = _value_for_column(
                    c,
                    row_i=10_000_000 + len(rows),
                    n_rows=10_000_000,
                    hint=hint_by.get(c.name.lower()),
                    rng=rng,
                    now=now,
                )
        rows.append(tuple(row))

    # 2) Values exactly on range boundaries
    for col_name, values in boundaries_by_col.items():
        for val in values:
            row = [
                _value_for_column(
                    c,
                    row_i=20_000_000 + len(rows),
                    n_rows=20_000_000,
                    hint=hint_by.get(c.name.lower()),
                    rng=rng,
                    now=now,
                )
                for c in cols
            ]
            for i, c in enumerate(cols):
                if c.name.lower() == col_name:
                    row[i] = val
            rows.append(tuple(row))

    # 3) Exact duplicate pairs for EXCEPT ALL multiset semantics
    dup_src = [
        tuple(
            _value_for_column(
                c,
                row_i=30_000_000 + k,
                n_rows=30_000_000,
                hint=hint_by.get(c.name.lower()),
                rng=rng,
                now=now,
            )
            for c in cols
        )
        for k in range(max(1, n_adv // 8))
    ]
    for r in dup_src:
        rows.append(r)
        rows.append(r)

    # 4) timestamptz midnight UTC and local midnight
    local_tz = ZoneInfo("America/New_York")
    for c_i, c in enumerate(cols):
        if not _is_timestamptz_type(c.data_type):
            continue
        for kind in ("utc", "local"):
            row = [
                _value_for_column(
                    col,
                    row_i=40_000_000 + len(rows),
                    n_rows=40_000_000,
                    hint=hint_by.get(col.name.lower()),
                    rng=rng,
                    now=now,
                )
                for col in cols
            ]
            day = datetime(2026, 6, 15, tzinfo=UTC)
            if kind == "utc":
                row[c_i] = day
            else:
                local_midnight = datetime(2026, 6, 15, 0, 0, 0, tzinfo=local_tz)
                row[c_i] = local_midnight.astimezone(UTC)
            rows.append(tuple(row))

    # Trim / pad to ~n_adv
    if len(rows) > n_adv:
        rows = rows[:n_adv]
    while len(rows) < n_adv:
        rows.append(
            tuple(
                _value_for_column(
                    c,
                    row_i=50_000_000 + len(rows),
                    n_rows=50_000_000,
                    hint=hint_by.get(c.name.lower()),
                    rng=rng,
                    now=now,
                )
                for c in cols
            )
        )
    return rows


def _copy_rows(
    conn: Connection[Any],
    table: str,
    columns: Sequence[str],
    rows: Iterator[tuple[Any, ...]] | Sequence[tuple[Any, ...]],
) -> None:
    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(table), col_list)
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            for row in rows:
                copy.write_row(row)


def apply_schema(conn: Connection[Any], schema: ParsedSchema) -> None:
    """Execute reconstructed DDL only — never raw user text."""
    for stmt in schema.reconstructed_ddl.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def seed_template(
    schema: ParsedSchema,
    *,
    hints: Sequence[ColumnHint] | None = None,
    query: ParsedQuery | None = None,
    settings: Settings | None = None,
    seed: int = 42,
) -> str:
    """
    Create (or replace) the seeded template database, load data, VACUUM ANALYZE,
    and mark it as a template. Returns the template database name.
    """
    s = settings or load_settings()
    hints = list(hints or [])
    db_name = s.sandbox.seeded_db
    n_rows = s.seed_rows
    n_adv = max(1, n_rows // 100)

    with connect(s, s.sandbox.admin_db, autocommit=True) as admin:
        # Clear template bit if a previous run left it set, then recreate.
        unmark_template(admin, db_name)
        create_database(admin, db_name)

    col_names = [c.name for c in schema.columns]

    with connect(s, db_name, autocommit=True) as conn:
        apply_schema(conn, schema)
        _copy_rows(
            conn,
            schema.table_name,
            col_names,
            _iter_bulk_rows(schema, n_rows=n_rows, hints=hints, seed=seed),
        )
        adv = _adversarial_rows(
            schema, query, n_adv=n_adv, hints=hints, seed=seed
        )
        if adv:
            _copy_rows(conn, schema.table_name, col_names, adv)

        # Visibility map must be populated for index-only scans later.
        conn.execute(
            sql.SQL("VACUUM ANALYZE {}").format(sql.Identifier(schema.table_name))
        )

    with connect(s, s.sandbox.admin_db, autocommit=True) as admin:
        terminate_backends(admin, db_name)
        mark_as_template(admin, db_name)

    return db_name
