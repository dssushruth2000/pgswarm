"""Quick ingest smoke checks (no database required)."""

from __future__ import annotations

import sys

from optimizer.ingest import IngestError, parse_ddl, parse_query


def expect_reject(ddl: str, needle: str) -> None:
    try:
        parse_ddl(ddl)
    except IngestError as exc:
        print(f"PASS reject ({needle}): {exc}")
        return
    raise SystemExit(f"FAIL: expected reject for {needle}")


def main() -> int:
    schema = parse_ddl(
        """
        CREATE TABLE orders (
          id integer NOT NULL,
          user_id integer NOT NULL,
          status text
        );
        CREATE INDEX ON orders (user_id) INCLUDE (status);
        """
    )
    assert schema.table_name == "orders"
    assert [c.name for c in schema.columns] == ["id", "user_id", "status"]
    assert len(schema.indexes) == 1
    assert schema.indexes[0].columns == ("user_id",)
    assert schema.indexes[0].include == ("status",)
    assert "CREATE TABLE" in schema.reconstructed_ddl.upper()
    print("PASS parse_ddl")

    q = parse_query(
        "SELECT id FROM orders WHERE lower(status) = 'x' AND name LIKE '%foo' AND amount + 1 > 10"
    )
    reasons = {n.reason for n in q.non_sargable}
    assert "function_wrap" in reasons
    assert "leading_wildcard_like" in reasons
    assert "arithmetic" in reasons
    print("PASS non-sargable detection:", reasons)

    expect_reject("CREATE FUNCTION f() RETURNS int AS $$ SELECT 1 $$ LANGUAGE sql;", "function")
    expect_reject("COPY orders FROM '/tmp/x';", "COPY")
    expect_reject("DO $$ BEGIN NULL; END $$;", "DO")
    expect_reject("CREATE TABLE t (id int); SELECT * FROM pg_user;", "pg_")
    expect_reject("CREATE TRIGGER tr AFTER INSERT ON t EXECUTE FUNCTION f();", "trigger")
    print("All ingest smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
