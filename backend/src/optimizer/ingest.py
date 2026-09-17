from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from optimizer.contracts import (
    ColumnDef,
    IndexDef,
    NonSargablePredicate,
    ParsedQuery,
    ParsedSchema,
    RangeBoundary,
)

_FORBIDDEN_KINDS = {"FUNCTION", "PROCEDURE", "TRIGGER", "SCHEMA", "EXTENSION", "ROLE", "USER"}
_PG_CATALOG_RE = re.compile(r"\bpg_[a-zA-Z0-9_]+\b", re.IGNORECASE)


class IngestError(ValueError):
    """Raised when user DDL or SQL fails validation."""


def _reject_pg_catalog(sql: str) -> None:
    if _PG_CATALOG_RE.search(sql):
        raise IngestError("references to pg_ catalogs are not allowed")


def _parse_statements(sql: str) -> list[exp.Expression]:
    _reject_pg_catalog(sql)
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except ParseError as exc:
        raise IngestError(f"parse failed: {exc}") from exc
    return [s for s in statements if s is not None]


def _column_name(node: exp.Expression | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    if isinstance(node, exp.Ordered):
        return _column_name(node.this)
    return None


def _extract_index_columns(index: exp.Index) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cols: list[str] = []
    includes: list[str] = []
    # sqlglot represents columns under Index.params
    params = index.args.get("params")
    if isinstance(params, exp.IndexParameters):
        columns = params.args.get("columns")
        if isinstance(columns, list):
            for c in columns:
                name = _column_name(c)
                if name:
                    cols.append(name)
        include = params.args.get("include")
        if isinstance(include, list):
            for c in include:
                name = _column_name(c)
                if name:
                    includes.append(name)
    # Fallback: expressions on the Index itself
    if not cols:
        for e in index.expressions:
            name = _column_name(e)
            if name:
                cols.append(name)
    return tuple(cols), tuple(includes)


def _table_from_create(create: exp.Create) -> str:
    this = create.this
    if isinstance(this, exp.Schema):
        table = this.this
        if isinstance(table, exp.Table):
            return table.name
    if isinstance(this, exp.Table):
        return this.name
    if isinstance(this, exp.Index):
        table = this.args.get("table")
        if isinstance(table, exp.Table):
            return table.name
    raise IngestError("could not resolve table name from CREATE statement")


def _parse_create_table(create: exp.Create) -> tuple[str, list[ColumnDef], tuple[str, ...] | None]:
    kind = (create.args.get("kind") or "").upper()
    if kind and kind != "TABLE":
        raise IngestError(f"CREATE {kind} is not allowed")

    schema = create.this
    if not isinstance(schema, exp.Schema):
        raise IngestError("CREATE TABLE must define a schema")

    table = schema.this
    if not isinstance(table, exp.Table):
        raise IngestError("CREATE TABLE missing table name")
    table_name = table.name

    columns: list[ColumnDef] = []
    primary_key: tuple[str, ...] | None = None

    for expr in schema.expressions:
        if isinstance(expr, exp.ColumnDef):
            col_name = expr.name
            kind_node = expr.args.get("kind")
            data_type = kind_node.sql(dialect="postgres") if kind_node else "text"
            constraints = expr.args.get("constraints") or []
            nullable = True
            for c in constraints:
                if isinstance(c, exp.ColumnConstraint):
                    kind_c = c.args.get("kind")
                    if isinstance(kind_c, exp.NotNullColumnConstraint):
                        nullable = False
                    if isinstance(kind_c, exp.PrimaryKeyColumnConstraint):
                        nullable = False
                        primary_key = (col_name,)
            columns.append(ColumnDef(name=col_name, data_type=data_type, nullable=nullable))
        elif isinstance(expr, exp.PrimaryKey):
            pk_cols = tuple(
                n for e in expr.expressions if (n := _column_name(e)) is not None
            )
            primary_key = pk_cols or primary_key
        elif isinstance(expr, (exp.CheckColumnConstraint, exp.Constraint)):
            continue
        else:
            # Reject nested function/trigger definitions inside table DDL
            sql_frag = expr.sql(dialect="postgres")
            upper = sql_frag.upper()
            if any(tok in upper for tok in ("FUNCTION", "TRIGGER", "PROCEDURE", "LANGUAGE")):
                raise IngestError(f"forbidden construct in CREATE TABLE: {sql_frag[:80]}")

    if not columns:
        raise IngestError("CREATE TABLE has no columns")
    return table_name, columns, primary_key


def _parse_create_index(create: exp.Create) -> IndexDef:
    kind = (create.args.get("kind") or "").upper()
    if kind and kind != "INDEX":
        raise IngestError(f"CREATE {kind} is not allowed")

    this = create.this
    if isinstance(this, exp.Index):
        index = this
    else:
        raise IngestError("CREATE INDEX missing index definition")

    table_node = index.args.get("table")
    if not isinstance(table_node, exp.Table):
        raise IngestError("CREATE INDEX missing table")
    cols, includes = _extract_index_columns(index)
    if not cols:
        raise IngestError("CREATE INDEX has no columns")

    predicate = None
    params = index.args.get("params")
    if isinstance(params, exp.IndexParameters):
        where = params.args.get("where")
        if where is not None:
            # Store the condition without the WHERE keyword.
            cond = where.this if isinstance(where, exp.Where) else where
            predicate = cond.sql(dialect="postgres") if cond is not None else None

    return IndexDef(
        name=index.name,
        table=table_node.name,
        columns=cols,
        include=includes,
        predicate=predicate,
        unique=bool(create.args.get("unique")),
    )


def parse_ddl(ddl: str) -> ParsedSchema:
    """Parse and reconstruct user DDL. Never return the original pasted text for execution."""
    statements = _parse_statements(ddl)
    if not statements:
        raise IngestError("DDL is empty")

    table_name: str | None = None
    columns: list[ColumnDef] = []
    primary_key: tuple[str, ...] | None = None
    indexes: list[IndexDef] = []
    reconstructed: list[str] = []

    for stmt in statements:
        # sqlglot may emit trailing EndStatement after DO blocks
        if type(stmt).__name__ == "EndStatement":
            raise IngestError("DO blocks are not allowed")

        raw = stmt.sql(dialect="postgres")
        upper = raw.lstrip().upper()

        if isinstance(stmt, exp.Command) or type(stmt).__name__ == "Command":
            cmd = str(stmt.args.get("this") or "").upper()
            if cmd == "DO" or upper.startswith("DO"):
                raise IngestError("DO blocks are not allowed")
            if "FUNCTION" in upper:
                raise IngestError("functions are not allowed")
            raise IngestError(f"forbidden command: {raw[:80]}")

        if isinstance(stmt, exp.Copy) or type(stmt).__name__ == "Copy":
            raise IngestError("COPY is not allowed")

        if upper.startswith("DO ") or upper == "DO":
            raise IngestError("DO blocks are not allowed")
        if upper.startswith("COPY "):
            raise IngestError("COPY is not allowed")
        if "CREATE FUNCTION" in upper or "CREATE OR REPLACE FUNCTION" in upper:
            raise IngestError("functions are not allowed")
        if "CREATE TRIGGER" in upper:
            raise IngestError("triggers are not allowed")

        if not isinstance(stmt, exp.Create):
            raise IngestError(
                f"only CREATE TABLE and CREATE INDEX are allowed, got {type(stmt).__name__}"
            )

        kind = (stmt.args.get("kind") or "").upper()
        if kind in _FORBIDDEN_KINDS:
            raise IngestError(f"CREATE {kind} is not allowed")
        if kind == "TABLE":
            if table_name is not None:
                raise IngestError("only one CREATE TABLE is allowed per ingest")
            table_name, columns, primary_key = _parse_create_table(stmt)
            reconstructed.append(stmt.sql(dialect="postgres"))
        elif kind == "INDEX":
            indexes.append(_parse_create_index(stmt))
            reconstructed.append(stmt.sql(dialect="postgres"))
        else:
            raise IngestError(f"CREATE {kind or 'UNKNOWN'} is not allowed")

    if table_name is None:
        raise IngestError("DDL must include a CREATE TABLE")

    for idx in indexes:
        if idx.table.lower() != table_name.lower():
            raise IngestError(
                f"index on '{idx.table}' does not match table '{table_name}'"
            )

    return ParsedSchema(
        table_name=table_name,
        columns=columns,
        primary_key=primary_key,
        indexes=indexes,
        reconstructed_ddl=";\n".join(reconstructed) + ";",
    )


def _collect_columns_from(node: exp.Expression | None) -> list[str]:
    if node is None:
        return []
    names: list[str] = []
    for col in node.find_all(exp.Column):
        if col.name:
            names.append(col.name)
    return names


def _is_leading_wildcard_like(like: exp.Like | exp.ILike) -> bool:
    pattern = like.expression
    if isinstance(pattern, exp.Literal) and pattern.is_string:
        return pattern.this.startswith("%") or pattern.this.startswith("_")
    return False


def _detect_non_sargable(where: exp.Expression | None) -> list[NonSargablePredicate]:
    if where is None:
        return []
    found: list[NonSargablePredicate] = []

    for like in where.find_all(exp.Like, exp.ILike):
        if _is_leading_wildcard_like(like):
            found.append(
                NonSargablePredicate(
                    sql=like.sql(dialect="postgres"),
                    reason="leading_wildcard_like",
                )
            )

    for func in where.find_all(exp.Anonymous, exp.Func):
        # Column wrapped in a function: date_trunc('day', created_at), lower(name), etc.
        if any(isinstance(arg, exp.Column) for arg in func.expressions) or isinstance(
            func.this, exp.Column
        ):
            # Skip simple casts sometimes represented oddly; still flag real funcs
            if isinstance(func, exp.Cast):
                continue
            found.append(
                NonSargablePredicate(
                    sql=func.sql(dialect="postgres"),
                    reason="function_wrap",
                )
            )

    for arith in where.find_all(exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod):
        if any(isinstance(p, exp.Column) for p in (*arith.flatten(), arith.this, arith.expression) if p):
            cols = [p for p in (arith.this, arith.expression) if isinstance(p, exp.Column)]
            if cols:
                found.append(
                    NonSargablePredicate(
                        sql=arith.sql(dialect="postgres"),
                        reason="arithmetic",
                    )
                )

    # Deduplicate by sql text
    seen: set[str] = set()
    unique: list[NonSargablePredicate] = []
    for item in found:
        if item.sql not in seen:
            seen.add(item.sql)
            unique.append(item)
    return unique


def _extract_range_boundaries(where: exp.Expression | None) -> list[RangeBoundary]:
    if where is None:
        return []
    boundaries: list[RangeBoundary] = []
    for cmp_ in where.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ):
        left, right = cmp_.this, cmp_.expression
        op = cmp_.key.lower() if hasattr(cmp_, "key") else type(cmp_).__name__.lower()
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            boundaries.append(
                RangeBoundary(column=left.name, value=_literal_value(right), op=op)
            )
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            boundaries.append(
                RangeBoundary(column=right.name, value=_literal_value(left), op=op)
            )
    return boundaries


def _literal_value(lit: exp.Literal) -> Any:
    if lit.is_string:
        return lit.this
    if lit.is_int:
        return int(lit.this)
    if lit.is_number:
        return float(lit.this)
    return lit.this


def parse_query(sql: str) -> ParsedQuery:
    statements = _parse_statements(sql)
    if len(statements) != 1:
        raise IngestError("query must be a single SELECT statement")
    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        # WITH ... SELECT
        if isinstance(stmt, exp.With) or (
            isinstance(stmt, exp.Subquery) and isinstance(stmt.this, exp.Select)
        ):
            select = stmt.this if isinstance(stmt, exp.Subquery) else stmt
            if not isinstance(select, (exp.Select, exp.With)):
                raise IngestError("query must be a SELECT")
        elif isinstance(stmt, exp.Union):
            raise IngestError("UNION queries are not supported at ingest")
        else:
            raise IngestError(f"query must be a SELECT, got {type(stmt).__name__}")

    select = stmt
    if isinstance(stmt, exp.With):
        select = stmt.this
    if not isinstance(select, exp.Select):
        raise IngestError("query must be a SELECT")

    projection: list[str] = []
    for e in select.expressions:
        if isinstance(e, exp.Star):
            projection.append("*")
        elif isinstance(e, exp.Alias):
            projection.append(e.alias_or_name)
        elif isinstance(e, exp.Column):
            projection.append(e.name)
        else:
            projection.append(e.alias_or_name or e.sql(dialect="postgres"))

    where = select.args.get("where")
    where_expr = where.this if isinstance(where, exp.Where) else where

    filter_columns = _collect_columns_from(where_expr)

    join_columns: list[str] = []
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        join_columns.extend(_collect_columns_from(on))

    order_by: list[str] = []
    order = select.args.get("order")
    if isinstance(order, exp.Order):
        for e in order.expressions:
            name = _column_name(e)
            if name:
                order_by.append(name)

    group_by: list[str] = []
    group = select.args.get("group")
    if isinstance(group, exp.Group):
        for e in group.expressions:
            name = _column_name(e)
            if name:
                group_by.append(name)

    return ParsedQuery(
        reconstructed_sql=stmt.sql(dialect="postgres"),
        projection=projection,
        filter_columns=list(dict.fromkeys(filter_columns)),
        join_columns=list(dict.fromkeys(join_columns)),
        order_by=order_by,
        group_by=group_by,
        non_sargable=_detect_non_sargable(where_expr),
        range_boundaries=_extract_range_boundaries(where_expr),
    )
