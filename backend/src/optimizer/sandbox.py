from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import Connection

from optimizer.config import Settings, load_settings


def connect_dsn(settings: Settings | None = None, database: str | None = None) -> str:
    s = settings or load_settings()
    db = database or s.sandbox.admin_db
    return (
        f"host={s.sandbox.host} port={s.sandbox.port} "
        f"dbname={db} user={s.sandbox.user} password={s.sandbox.password}"
    )


@contextmanager
def connect(
    settings: Settings | None = None,
    database: str | None = None,
    *,
    autocommit: bool = False,
) -> Iterator[Connection[Any]]:
    conn = psycopg.connect(connect_dsn(settings, database), autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def terminate_backends(conn: Connection[Any], database: str) -> None:
    conn.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s AND pid <> pg_backend_pid()
        """,
        (database,),
    )


def drop_database(conn: Connection[Any], name: str) -> None:
    terminate_backends(conn, name)
    conn.execute(f'DROP DATABASE IF EXISTS "{name}"')


def create_database(conn: Connection[Any], name: str, *, template: str | None = None) -> None:
    drop_database(conn, name)
    if template:
        terminate_backends(conn, template)
        conn.execute(f'CREATE DATABASE "{name}" TEMPLATE "{template}"')
    else:
        conn.execute(f'CREATE DATABASE "{name}"')


def clone_from_template(
    settings: Settings | None = None,
    *,
    clone_name: str,
    template: str | None = None,
) -> str:
    """Clone the seeded template into a fresh database. Returns the clone name."""
    s = settings or load_settings()
    tpl = template or s.sandbox.seeded_db
    with connect(s, s.sandbox.admin_db, autocommit=True) as conn:
        create_database(conn, clone_name, template=tpl)
    return clone_name


def drop_clone(settings: Settings | None = None, *, clone_name: str) -> None:
    s = settings or load_settings()
    with connect(s, s.sandbox.admin_db, autocommit=True) as conn:
        drop_database(conn, clone_name)


def mark_as_template(conn: Connection[Any], database: str) -> None:
    conn.execute(
        "UPDATE pg_database SET datistemplate = true WHERE datname = %s",
        (database,),
    )


def unmark_template(conn: Connection[Any], database: str) -> None:
    conn.execute(
        "UPDATE pg_database SET datistemplate = false WHERE datname = %s",
        (database,),
    )
