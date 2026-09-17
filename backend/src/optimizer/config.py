from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SandboxSettings(BaseModel):
    host: str = "localhost"
    port: int = 55432
    user: str = "postgres"
    password: str = "sandbox"
    admin_db: str = "bench"
    seeded_db: str = "bench_seeded"


class Settings(BaseModel):
    seed_rows: int
    insert_batch_size: int
    read_runs: int
    write_runs: int
    statement_timeout_ms: int
    run_timeout_s: int
    max_candidates_screened: int
    archetypes: list[str]
    model_dev: str
    model_final: str
    token_budget_usd_per_run: float
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


@lru_cache
def load_settings(path: str | None = None) -> Settings:
    cfg_path = Path(path) if path else _config_path()
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return Settings.model_validate(raw)
