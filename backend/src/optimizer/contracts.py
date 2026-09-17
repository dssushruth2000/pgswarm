from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Archetype = Literal["max_read", "min_write", "rewrite_only"]


class ColumnHint(BaseModel):
    name: str
    distinct_ratio: float | None = None
    null_fraction: float | None = None
    skew: dict[str, float] | None = None


class RunInput(BaseModel):
    ddl: str
    query: str
    read_write_ratio: float
    hints: list[ColumnHint] = Field(default_factory=list)


class Candidate(BaseModel):
    archetype: Archetype
    index_sql: str | None = None
    rewritten_query: str | None = None
    rationale: str
    proposed_by: str


class ScreenResult(BaseModel):
    candidate: Candidate
    planner_cost: float
    plan_uses_index: bool


class BenchResult(BaseModel):
    candidate: Candidate
    read_ms_median: float
    plan_node: str
    heap_fetches: int | None
    shared_hit_blocks: int
    wal_bytes: int
    insert_ms_median: float
    index_bytes: int


class BenchDelta(BaseModel):
    """Every figure relative to the no-index baseline (candidate − baseline)."""

    candidate: Candidate
    read_ms_delta: float
    wal_bytes_delta: int
    insert_ms_delta: float
    index_bytes: int
    plan_node: str
    heap_fetches: int | None
    shared_hit_blocks: int
    absolute: BenchResult


class BenchSuiteResult(BaseModel):
    baseline: BenchResult
    results: list[BenchResult]
    deltas: list[BenchDelta]


class Verdict(BaseModel):
    chosen: Candidate
    baseline: BenchResult
    results: list[BenchResult]
    decision_record: str
    migration_sql: str


class ColumnDef(BaseModel):
    name: str
    data_type: str
    nullable: bool = True


class IndexDef(BaseModel):
    name: str | None = None
    table: str
    columns: tuple[str, ...]
    include: tuple[str, ...] = ()
    predicate: str | None = None
    unique: bool = False


class ParsedSchema(BaseModel):
    table_name: str
    columns: list[ColumnDef]
    primary_key: tuple[str, ...] | None = None
    indexes: list[IndexDef] = Field(default_factory=list)
    reconstructed_ddl: str


class NonSargablePredicate(BaseModel):
    sql: str
    reason: Literal["function_wrap", "arithmetic", "leading_wildcard_like"]


class RangeBoundary(BaseModel):
    column: str
    value: Any
    op: str


class ParsedQuery(BaseModel):
    reconstructed_sql: str
    projection: list[str]
    filter_columns: list[str] = Field(default_factory=list)
    join_columns: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    non_sargable: list[NonSargablePredicate] = Field(default_factory=list)
    range_boundaries: list[RangeBoundary] = Field(default_factory=list)
