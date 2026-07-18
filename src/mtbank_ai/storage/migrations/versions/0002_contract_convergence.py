"""Converge applied foundation schemas with the domain contracts.

Revision ID: 0002_contract_convergence
Revises: 0001_foundation

This revision is intentionally online-only because it validates the live PostgreSQL
catalog and locks privacy-sensitive tables before changing their schema.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError

revision: str = "0002_contract_convergence"
down_revision: str | None = "0001_foundation"
branch_labels: str | None = None
depends_on: str | None = None

SCHEMA = "public"
TABLES = ("runs", "run_events", "analyses")
_POPULATED_ANALYSES_ERROR = (
    "0002_contract_convergence refuses populated analyses; export and remediate legacy data outside this migration"
)
_UNSUPPORTED_SCHEMA_ERROR = "0002_contract_convergence found an unsupported managed schema"
_MANAGED_UPDATE_ERROR = "0002_contract_convergence managed data transition failed"
_INVALID_EVENT_UNIQUE_ERROR = "uq_run_events_run_id_event_hash must use ordered columns (run_id, current_hash)"

_RUN_CHECKS = {
    "source": (
        "source IN ('openwebui', 'rest_file', 'rest_url', 'websocket', 'eval')",
        "ck_runs_ck_runs_source_allowed",
        "ck_runs_source_allowed",
    ),
    "status_baseline": (
        "status IN ('pending', 'running', 'completed', 'failed')",
        "ck_runs_ck_runs_status_allowed",
        "ck_runs_status_allowed",
    ),
    "status_head": (
        "status IN ('queued', 'processing', 'completed', 'failed')",
        "ck_runs_ck_runs_status_allowed",
        "ck_runs_status_allowed",
    ),
    "envelope": (
        "pg_catalog.jsonb_typeof(envelope) = 'object'",
        "ck_runs_ck_runs_envelope_object",
        "ck_runs_envelope_object",
    ),
}
_EVENT_CHECKS = {
    "sequence": (
        "sequence > 0",
        "ck_run_events_ck_run_events_sequence_positive",
        "ck_run_events_sequence_positive",
    ),
    "current_hash": (
        "current_hash ~ '^[0-9a-f]{64}$'",
        "ck_run_events_ck_run_events_current_hash_sha256",
        "ck_run_events_current_hash_sha256",
    ),
    "previous_hash": (
        "previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{64}$'",
        "ck_run_events_ck_run_events_previous_hash_sha256",
        "ck_run_events_previous_hash_sha256",
    ),
    "hash_chain": (
        "(sequence = 1 AND previous_hash IS NULL) OR (sequence > 1 AND previous_hash IS NOT NULL)",
        "ck_run_events_ck_run_events_hash_chain_position",
        "ck_run_events_hash_chain_position",
    ),
    "payload": (
        "pg_catalog.jsonb_typeof(redacted_payload) = 'object'",
        "ck_run_events_ck_run_events_redacted_payload_object",
        "ck_run_events_redacted_payload_object",
    ),
}
_ANALYSIS_CHECKS = {
    "result_object": (
        "pg_catalog.jsonb_typeof(result) = 'object'",
        "ck_analyses_ck_analyses_result_object",
        "ck_analyses_result_object",
    ),
    "result_digest": (
        "result_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_analyses_ck_analyses_result_sha256",
        "ck_analyses_result_sha256",
    ),
    "sanitized_object": (
        "pg_catalog.jsonb_typeof(sanitized_result) = 'object'",
        "ck_analyses_ck_analyses_result_object",
        "ck_analyses_sanitized_result_object",
    ),
    "sanitized_digest": (
        "sanitized_result_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_analyses_ck_analyses_result_sha256",
        "ck_analyses_sanitized_result_sha256",
    ),
}
_RUN_CHECK_NAMES = tuple(sorted({name for _, first, second in _RUN_CHECKS.values() for name in (first, second)}))
_ANALYSIS_CHECK_NAMES = tuple(
    sorted({name for _, first, second in _ANALYSIS_CHECKS.values() for name in (first, second)})
)


def _require_online_postgresql() -> None:
    context = op.get_context()
    if context.as_sql:
        raise RuntimeError("0002_contract_convergence does not support Alembic --sql; run it online against PostgreSQL")
    if op.get_bind().dialect.name != "postgresql":
        raise RuntimeError("0002_contract_convergence requires PostgreSQL")


def _set_trusted_search_path() -> None:
    """Keep 0002 expression parsing in pg_catalog without changing historical 0001 DDL."""
    op.execute(sa.text("SET LOCAL search_path TO pg_catalog"))


def _lock_managed_tables() -> None:
    """Lock managed relations in the documented global order: runs, run_events, analyses."""
    op.execute(sa.text("LOCK TABLE public.runs, public.run_events, public.analyses IN ACCESS EXCLUSIVE MODE"))


def _require_ordinary_unprotected_relations() -> None:
    rows = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT c.relname,
                c.relkind = 'r'::"char"
                    AND c.relpersistence = 'p'::"char"
                    AND NOT c.relispartition
                    AND c.reloftype = 0
                    AND NOT c.relrowsecurity
                    AND NOT c.relforcerowsecurity
                    AND NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_inherits AS inheritance
                        WHERE inheritance.inhrelid = c.oid OR inheritance.inhparent = c.oid
                    ) AS ordinary_unprotected
            FROM pg_catalog.pg_class AS c
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema AND c.relname IN :table_names
            """
            ).bindparams(sa.bindparam("table_names", expanding=True)),
            {"schema": SCHEMA, "table_names": TABLES},
        )
        .mappings()
    )
    relations = {str(row["relname"]): row for row in rows}
    if set(relations) != set(TABLES) or any(not row["ordinary_unprotected"] for row in relations.values()):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _lock_and_require_empty_analyses() -> None:
    _require_ordinary_unprotected_relations()
    _lock_managed_tables()
    _require_ordinary_unprotected_relations()
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM public.analyses)")):
        raise RuntimeError(_POPULATED_ANALYSES_ERROR)


def _require_inbound_foreign_key_and_internal_trigger_contract() -> None:
    has_unsafe_inbound_contract = op.get_bind().scalar(
        sa.text(
            """
            WITH inbound_foreign_keys AS (
                SELECT foreign_key.oid, foreign_key.conname, foreign_key.conrelid, foreign_key.confrelid,
                    foreign_key.conkey, foreign_key.confkey, foreign_key.condeferrable,
                    foreign_key.condeferred, foreign_key.convalidated, foreign_key.connoinherit,
                    foreign_key.coninhcount, foreign_key.conparentid, foreign_key.confdeltype,
                    foreign_key.confupdtype, foreign_key.confmatchtype,
                    child_namespace.nspname AS child_schema, child_relation.relname AS child_table,
                    parent_namespace.nspname AS parent_schema, parent_relation.relname AS parent_table,
                    child_attribute.attname AS child_column, parent_attribute.attname AS parent_column
                FROM pg_catalog.pg_constraint AS foreign_key
                JOIN pg_catalog.pg_class AS child_relation ON child_relation.oid = foreign_key.conrelid
                JOIN pg_catalog.pg_namespace AS child_namespace ON child_namespace.oid = child_relation.relnamespace
                JOIN pg_catalog.pg_class AS parent_relation ON parent_relation.oid = foreign_key.confrelid
                JOIN pg_catalog.pg_namespace AS parent_namespace
                    ON parent_namespace.oid = parent_relation.relnamespace
                LEFT JOIN pg_catalog.pg_attribute AS child_attribute
                    ON child_attribute.attrelid = foreign_key.conrelid
                    AND child_attribute.attnum = foreign_key.conkey[1]
                LEFT JOIN pg_catalog.pg_attribute AS parent_attribute
                    ON parent_attribute.attrelid = foreign_key.confrelid
                    AND parent_attribute.attnum = foreign_key.confkey[1]
                WHERE foreign_key.contype = 'f'::"char"
                  AND parent_namespace.nspname = :schema
                  AND parent_relation.relname IN :table_names
            ), allowed_foreign_keys AS (
                SELECT foreign_key.oid
                FROM inbound_foreign_keys AS foreign_key
                WHERE pg_catalog.array_length(foreign_key.conkey, 1) = 1
                  AND pg_catalog.array_length(foreign_key.confkey, 1) = 1
                  AND NOT foreign_key.condeferrable
                  AND NOT foreign_key.condeferred
                  AND foreign_key.convalidated
                  AND foreign_key.connoinherit
                  AND foreign_key.coninhcount = 0
                  AND foreign_key.conparentid = 0
                  AND foreign_key.confdeltype = 'r'::"char"
                  AND foreign_key.confupdtype = 'a'::"char"
                  AND foreign_key.confmatchtype = 's'::"char"
                  AND (
                      (
                          foreign_key.conname = 'fk_run_events_run_id_runs'
                          AND foreign_key.child_schema = :schema
                          AND foreign_key.child_table = 'run_events'
                          AND foreign_key.child_column = 'run_id'
                          AND foreign_key.parent_schema = :schema
                          AND foreign_key.parent_table = 'runs'
                          AND foreign_key.parent_column = 'run_id'
                      ) OR (
                          foreign_key.conname = 'fk_analyses_run_id_runs'
                          AND foreign_key.child_schema = :schema
                          AND foreign_key.child_table = 'analyses'
                          AND foreign_key.child_column = 'run_id'
                          AND foreign_key.parent_schema = :schema
                          AND foreign_key.parent_table = 'runs'
                          AND foreign_key.parent_column = 'run_id'
                      )
                  )
            )
            SELECT (
                (SELECT pg_catalog.count(*) FROM inbound_foreign_keys) <> 2
            ) OR EXISTS (
                SELECT 1
                FROM inbound_foreign_keys AS foreign_key
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM allowed_foreign_keys AS allowed_foreign_key
                    WHERE allowed_foreign_key.oid = foreign_key.oid
                )
            ) OR EXISTS (
                SELECT 1
                FROM allowed_foreign_keys AS allowed_foreign_key
                WHERE (
                    SELECT pg_catalog.count(*)
                    FROM pg_catalog.pg_trigger AS trigger_data
                    WHERE trigger_data.tgconstraint = allowed_foreign_key.oid
                      AND trigger_data.tgisinternal
                      AND trigger_data.tgenabled = 'O'::"char"
                ) <> 4
            ) OR EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger AS trigger_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_data.tgrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema
                  AND relation.relname IN :table_names
                  AND trigger_data.tgisinternal
                  AND (
                      trigger_data.tgenabled <> 'O'::"char"
                      OR NOT EXISTS (
                          SELECT 1
                          FROM allowed_foreign_keys AS allowed_foreign_key
                          WHERE allowed_foreign_key.oid = trigger_data.tgconstraint
                      )
                  )
            )
            """
        ).bindparams(sa.bindparam("table_names", expanding=True)),
        {"schema": SCHEMA, "table_names": TABLES},
    )
    if has_unsafe_inbound_contract:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _require_no_active_trigger_or_rule_contract() -> None:
    has_active_side_effects = op.get_bind().scalar(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger AS trigger_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_data.tgrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema
                  AND relation.relname IN :table_names
                  AND NOT trigger_data.tgisinternal
                  AND trigger_data.tgenabled <> 'D'::"char"
            ) OR EXISTS (
                SELECT 1
                FROM pg_catalog.pg_rewrite AS rule_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = rule_data.ev_class
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema
                  AND relation.relname IN :table_names
                  AND rule_data.ev_type <> '1'::"char"
                  AND rule_data.ev_enabled <> 'D'::"char"
            )
            """
        ).bindparams(sa.bindparam("table_names", expanding=True)),
        {"schema": SCHEMA, "table_names": TABLES},
    )
    if has_active_side_effects:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _columns(table_name: str) -> dict[str, dict[str, Any]]:
    inspector = sa.inspect(op.get_bind())
    inspected_columns = {
        str(column["name"]): dict(column) for column in inspector.get_columns(table_name, schema=SCHEMA)
    }
    rows = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT attribute.attname,
                pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS type_name,
                attribute.attnotnull, attribute.attidentity, attribute.attgenerated,
                attribute.attcollation = type_data.typcollation AS default_collation,
                pg_catalog.pg_get_expr(default_value.adbin, default_value.adrelid, false) AS default_definition
            FROM pg_catalog.pg_attribute AS attribute
            JOIN pg_catalog.pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_catalog.pg_type AS type_data ON type_data.oid = attribute.atttypid
            LEFT JOIN pg_catalog.pg_attrdef AS default_value
                ON default_value.adrelid = attribute.attrelid AND default_value.adnum = attribute.attnum
            WHERE namespace.nspname = :schema
              AND relation.relname = :table_name
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            """
            ),
            {"schema": SCHEMA, "table_name": table_name},
        )
        .mappings()
    )
    catalog_columns = {str(row["attname"]): dict(row) for row in rows}
    if set(inspected_columns) != set(catalog_columns):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    for name, column in inspected_columns.items():
        catalog = catalog_columns[name]
        column["_catalog_type_name"] = catalog["type_name"]
        column["_catalog_not_null"] = catalog["attnotnull"]
        column["_catalog_identity"] = catalog["attidentity"]
        column["_catalog_generated"] = catalog["attgenerated"]
        column["_catalog_default_collation"] = catalog["default_collation"]
        column["_catalog_default"] = catalog["default_definition"]
    return inspected_columns


def _constraints(table_name: str) -> dict[str, dict[str, Any]]:
    rows = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT con.conname, con.contype, con.condeferrable, con.condeferred,
                con.convalidated, con.connoinherit, con.coninhcount, con.conparentid,
                con.conindid AS index_oid, con.confdeltype, con.confupdtype, con.confmatchtype,
                referenced_namespace.nspname AS referenced_schema,
                referenced_relation.relname AS referenced_table,
                ARRAY(
                    SELECT attribute.attname
                    FROM pg_catalog.unnest(con.conkey) WITH ORDINALITY
                        AS key_attribute(attnum, position)
                    JOIN pg_catalog.pg_attribute AS attribute
                        ON attribute.attrelid = con.conrelid AND attribute.attnum = key_attribute.attnum
                    ORDER BY key_attribute.position
                ) AS key_columns,
                ARRAY(
                    SELECT attribute.attname
                    FROM pg_catalog.unnest(con.confkey) WITH ORDINALITY
                        AS key_attribute(attnum, position)
                    JOIN pg_catalog.pg_attribute AS attribute
                        ON attribute.attrelid = con.confrelid AND attribute.attnum = key_attribute.attnum
                    ORDER BY key_attribute.position
                ) AS referenced_columns,
                pg_catalog.pg_get_constraintdef(con.oid, false) AS definition
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS rel ON rel.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = rel.relnamespace
            LEFT JOIN pg_catalog.pg_class AS referenced_relation ON referenced_relation.oid = con.confrelid
            LEFT JOIN pg_catalog.pg_namespace AS referenced_namespace
                ON referenced_namespace.oid = referenced_relation.relnamespace
            WHERE namespace.nspname = :schema AND rel.relname = :table_name
            """
            ),
            {"schema": SCHEMA, "table_name": table_name},
        )
        .mappings()
    )
    return {str(row["conname"]): dict(row) for row in rows}


def _canonical_check_definition(expression: str) -> str:
    """Ask PostgreSQL 16 to deparse the static trusted expression on matching types."""
    bind = op.get_bind()
    savepoint = bind.begin_nested()
    try:
        bind.execute(sa.text("SET LOCAL search_path TO pg_catalog"))
        bind.execute(
            sa.text(
                """
                CREATE TEMP TABLE pg_temp.mtbank_check_probe (
                    source varchar(32), status varchar(32), envelope jsonb,
                    sequence integer, current_hash varchar(64), previous_hash varchar(64),
                    redacted_payload jsonb, result jsonb, result_sha256 varchar(64),
                    sanitized_result jsonb, sanitized_result_sha256 varchar(64)
                ) ON COMMIT DROP
                """
            )
        )
        bind.execute(
            sa.text(
                "ALTER TABLE pg_temp.mtbank_check_probe "
                f"ADD CONSTRAINT mtbank_check_probe_contract CHECK ({expression})"
            )
        )
        definition = bind.scalar(
            sa.text(
                """
                SELECT pg_catalog.pg_get_constraintdef(con.oid, false)
                FROM pg_catalog.pg_constraint AS con
                WHERE con.conrelid = 'pg_temp.mtbank_check_probe'::pg_catalog.regclass
                  AND con.conname = 'mtbank_check_probe_contract'
                """
            )
        )
        if not isinstance(definition, str):
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    except BaseException:
        try:
            savepoint.rollback()
        except Exception:
            pass
        raise
    savepoint.rollback()
    return definition


def _validate_check_group(table_name: str, checks: Iterable[tuple[str, str, str]]) -> None:
    constraints = _constraints(table_name)
    for expression, baseline_name, head_name in checks:
        expected_definition = _canonical_check_definition(expression)
        reserved_names = (baseline_name, head_name)
        matching_names = []
        for name in reserved_names:
            constraint = constraints.get(name)
            if constraint is None:
                continue
            if (
                not _ordinary_validated_constraint(constraint, "c", ())
                or constraint["definition"] != expected_definition
            ):
                raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
            matching_names.append(name)
        if len(matching_names) != 1:
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _validate_all_checks(payload_name: str, digest_name: str | None, *, head: bool) -> None:
    _validate_check_group(
        "runs",
        (
            _RUN_CHECKS["source"],
            _RUN_CHECKS["status_head" if head else "status_baseline"],
            _RUN_CHECKS["envelope"],
        ),
    )
    _validate_check_group("run_events", _EVENT_CHECKS.values())
    analysis_checks = (
        (_ANALYSIS_CHECKS["sanitized_object"], _ANALYSIS_CHECKS["sanitized_digest"])
        if payload_name == "sanitized_result"
        else (_ANALYSIS_CHECKS["result_object"], _ANALYSIS_CHECKS["result_digest"])
    )
    if digest_name is None:
        analysis_checks = analysis_checks[:1]
    _validate_check_group("analyses", analysis_checks)
    reserved_analysis_names = {name for _, first, second in _ANALYSIS_CHECKS.values() for name in (first, second)}
    active_analysis_names = {name for _, first, second in analysis_checks for name in (first, second)}
    existing_reserved = reserved_analysis_names.intersection(_constraints("analyses"))
    if not existing_reserved.issubset(active_analysis_names):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _has_catalog_type(column: dict[str, Any], expected_type: str) -> bool:
    return column.get("_catalog_type_name") == expected_type


def _normalised_default(value: str) -> str:
    return "".join(value.lower().split())


def _has_default(column: dict[str, Any], expected_default: str | None) -> bool:
    actual_default = column.get("_catalog_default")
    if expected_default is None:
        return actual_default is None
    return isinstance(actual_default, str) and _normalised_default(actual_default) == expected_default


def _is_catalog_empty_marker(value: object) -> bool:
    return value in {"", "\x00", b"", b"\x00"}


def _is_uuid(column: dict[str, Any]) -> bool:
    return isinstance(column["type"], postgresql.UUID) and _has_catalog_type(column, "uuid")


def _is_jsonb(column: dict[str, Any]) -> bool:
    return isinstance(column["type"], postgresql.JSONB) and _has_catalog_type(column, "jsonb")


def _is_varchar_sha256(column: dict[str, Any]) -> bool:
    column_type = column["type"]
    return (
        isinstance(column_type, sa.VARCHAR)
        and column_type.length == 64
        and _has_catalog_type(column, "character varying(64)")
    )


def _is_timestamptz_now(column: dict[str, Any]) -> bool:
    return _is_timestamptz(column) and _has_default(column, "now()")


def _is_varchar(length: int) -> Any:
    return lambda column: (
        isinstance(column["type"], sa.VARCHAR)
        and column["type"].length == length
        and _has_catalog_type(column, f"character varying({length})")
    )


def _is_timestamptz(column: dict[str, Any]) -> bool:
    return (
        isinstance(column["type"], sa.DateTime)
        and column["type"].timezone is True
        and _has_catalog_type(column, "timestamp with time zone")
    )


def _is_integer(column: dict[str, Any]) -> bool:
    return type(column["type"]).__name__ == "INTEGER" and _has_catalog_type(column, "integer")


def _require_columns(
    table_name: str,
    expected: dict[str, tuple[bool, Any, str | None]],
) -> None:
    columns = _columns(table_name)
    if set(columns) != set(expected):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    for name, (nullable, predicate, default) in expected.items():
        column = columns[name]
        if (
            column.get("nullable") is not nullable
            or column.get("_catalog_not_null") is not (not nullable)
            or not predicate(column)
            or column.get("computed") is not None
            or column.get("identity") is not None
            or not _is_catalog_empty_marker(column.get("_catalog_identity"))
            or not _is_catalog_empty_marker(column.get("_catalog_generated"))
            or column.get("_catalog_default_collation") is not True
            or (isinstance(column["type"], sa.String) and column["type"].collation is not None)
            or not _has_default(column, default)
        ):
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _constraint_has_type(constraint: dict[str, Any], expected_type: str) -> bool:
    return constraint.get("contype") in {expected_type, expected_type.encode()}


def _ordinary_validated_constraint(
    constraint: dict[str, Any],
    expected_type: str,
    expected_columns: tuple[str, ...],
) -> bool:
    return (
        _constraint_has_type(constraint, expected_type)
        and bool(constraint.get("convalidated"))
        and not constraint.get("condeferrable")
        and not constraint.get("condeferred")
        and (
            (expected_type == "c" and not constraint.get("connoinherit"))
            or (expected_type != "c" and bool(constraint.get("connoinherit")))
        )
        and constraint.get("coninhcount") == 0
        and constraint.get("conparentid") == 0
        and (expected_type == "c" or tuple(constraint.get("key_columns") or ()) == expected_columns)
    )


def _require_key_and_fk_contract(
    table_name: str,
    *,
    primary_key: tuple[str, ...],
    foreign_keys: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...],
    unique_names: dict[str, tuple[str, ...]],
) -> None:
    constraints = _constraints(table_name)
    primary_keys = [constraint for constraint in constraints.values() if _constraint_has_type(constraint, "p")]
    if len(primary_keys) != 1 or not _ordinary_validated_constraint(primary_keys[0], "p", primary_key):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    found_foreign_keys = [constraint for constraint in constraints.values() if _constraint_has_type(constraint, "f")]
    if len(found_foreign_keys) != len(foreign_keys):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    for local_columns, target_table, target_columns in foreign_keys:
        matches = [
            constraint
            for constraint in found_foreign_keys
            if _ordinary_validated_constraint(constraint, "f", local_columns)
            and constraint.get("referenced_schema") == SCHEMA
            and constraint.get("referenced_table") == target_table
            and tuple(constraint.get("referenced_columns") or ()) == target_columns
            and constraint.get("confdeltype") in {"r", b"r"}
            and constraint.get("confupdtype") in {"a", b"a"}
            and constraint.get("confmatchtype") in {"s", b"s"}
        ]
        if len(matches) != 1:
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    for name, columns in unique_names.items():
        constraint = constraints.get(name)
        if (
            constraint is None
            or not _ordinary_validated_constraint(constraint, "u", columns)
            or not constraint.get("index_oid")
        ):
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
        _require_index_contract(table_name, name, columns, unique=True, constraint_owned=True)


def _require_index_contract(
    table_name: str,
    name: str,
    columns: tuple[str, ...],
    *,
    unique: bool,
    constraint_owned: bool,
) -> None:
    row = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT index_relation.oid AS index_oid, index_relation.reloptions,
                index_data.indisvalid, index_data.indisready, index_data.indisunique,
                index_data.indisprimary, index_data.indisexclusion, index_data.indimmediate,
                index_data.indpred IS NULL AS no_predicate,
                index_data.indexprs IS NULL AS no_expressions,
                index_data.indnkeyatts, index_data.indnatts, access_method.amname,
                NOT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_constraint AS catalog_constraint
                    WHERE catalog_constraint.conindid = index_data.indexrelid
                ) AS not_constraint_owned,
                ARRAY(
                    SELECT attribute.attname
                    FROM pg_catalog.unnest(index_data.indkey) WITH ORDINALITY
                        AS key_attribute(attnum, position)
                    JOIN pg_catalog.pg_attribute AS attribute
                        ON attribute.attrelid = index_data.indrelid
                        AND attribute.attnum = key_attribute.attnum
                    WHERE key_attribute.position <= index_data.indnkeyatts
                    ORDER BY key_attribute.position
                ) AS key_columns,
                ARRAY(
                    SELECT key_option.option
                    FROM pg_catalog.unnest(index_data.indoption) WITH ORDINALITY
                        AS key_option(option, position)
                    WHERE key_option.position <= index_data.indnkeyatts
                    ORDER BY key_option.position
                ) AS key_options,
                pg_catalog.pg_get_indexdef(index_data.indexrelid) AS definition
            FROM pg_catalog.pg_index AS index_data
            JOIN pg_catalog.pg_class AS relation ON relation.oid = index_data.indrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_data.indexrelid
            JOIN pg_catalog.pg_am AS access_method ON access_method.oid = index_relation.relam
            WHERE namespace.nspname = :schema
              AND relation.relname = :table_name
              AND index_relation.relname = :index_name
            """
            ),
            {"schema": SCHEMA, "table_name": table_name, "index_name": name},
        )
        .mappings()
        .one_or_none()
    )
    expected_prefix = "CREATE UNIQUE INDEX" if unique else "CREATE INDEX"
    expected_definition = f"{expected_prefix} {name} ON {SCHEMA}.{table_name} USING btree ({', '.join(columns)})"
    if (
        row is None
        or row["indisvalid"] is not True
        or row["indisready"] is not True
        or row["indisunique"] is not unique
        or row["indisprimary"] is not False
        or row["indisexclusion"] is not False
        or row["indimmediate"] is not True
        or row["no_predicate"] is not True
        or row["no_expressions"] is not True
        or row["indnkeyatts"] != len(columns)
        or row["indnatts"] != len(columns)
        or row["amname"] != "btree"
        or row["not_constraint_owned"] is not (not constraint_owned)
        or tuple(row["key_columns"] or ()) != columns
        or tuple(row["key_options"] or ()) != (0,) * len(columns)
        or row["reloptions"]
        or row["definition"] != expected_definition
    ):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _require_indexes(table_name: str, expected: dict[str, tuple[str, ...]]) -> None:
    for name, columns in expected.items():
        _require_index_contract(table_name, name, columns, unique=False, constraint_owned=False)


def _require_all_index_contracts() -> None:
    rows = (
        op.get_bind()
        .execute(
            sa.text(
                """
                SELECT relation.relname AS table_name, index_relation.relname AS index_name,
                    index_relation.relkind, index_relation.relpersistence, index_relation.reloptions,
                    index_data.indisvalid, index_data.indisready, index_data.indislive,
                    index_data.indisexclusion, index_data.indimmediate, index_data.indcheckxmin,
                    index_data.indpred IS NULL AS no_predicate,
                    index_data.indexprs IS NULL AS no_expressions,
                    index_data.indnkeyatts, index_data.indnatts, access_method.amname,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indkey) AS key_attribute(attnum)
                        LEFT JOIN pg_catalog.pg_attribute AS attribute
                            ON attribute.attrelid = index_data.indrelid
                            AND attribute.attnum = key_attribute.attnum
                        WHERE key_attribute.attnum <= 0
                           OR attribute.attnum IS NULL
                           OR attribute.attisdropped
                    ) AS ordinary_key_columns,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indoption) AS key_option(option)
                        WHERE key_option.option <> 0
                    ) AS ordinary_key_options,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indclass) AS index_opclass(opclass_oid)
                        JOIN pg_catalog.pg_opclass AS catalog_opclass ON catalog_opclass.oid = index_opclass.opclass_oid
                        JOIN pg_catalog.pg_namespace AS opclass_namespace
                            ON opclass_namespace.oid = catalog_opclass.opcnamespace
                        WHERE opclass_namespace.nspname <> :trusted_schema
                    ) AS trusted_opclasses,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indcollation) AS collation_data(collation_oid)
                        JOIN pg_catalog.pg_collation AS catalog_collation
                            ON catalog_collation.oid = collation_data.collation_oid
                        JOIN pg_catalog.pg_namespace AS collation_namespace
                            ON collation_namespace.oid = catalog_collation.collnamespace
                        WHERE collation_data.collation_oid <> 0
                          AND collation_namespace.nspname <> :trusted_schema
                    ) AS trusted_collations,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indclass) AS index_opclass(opclass_oid)
                        JOIN pg_catalog.pg_opclass AS catalog_opclass ON catalog_opclass.oid = index_opclass.opclass_oid
                        JOIN pg_catalog.pg_namespace AS opclass_namespace
                            ON opclass_namespace.oid = catalog_opclass.opcnamespace
                        JOIN pg_catalog.pg_amproc AS support_data
                            ON support_data.amprocfamily = catalog_opclass.opcfamily
                            AND support_data.amproclefttype = catalog_opclass.opcintype
                            AND support_data.amprocrighttype = catalog_opclass.opcintype
                        JOIN pg_catalog.pg_proc AS support_procedure ON support_procedure.oid = support_data.amproc
                        JOIN pg_catalog.pg_namespace AS support_namespace
                            ON support_namespace.oid = support_procedure.pronamespace
                        WHERE opclass_namespace.nspname <> :trusted_schema
                           OR support_namespace.nspname <> :trusted_schema
                           OR support_procedure.provolatile <> 'i'::"char"
                    ) AS trusted_support_procedures,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.unnest(index_data.indclass) AS index_opclass(opclass_oid)
                        JOIN pg_catalog.pg_opclass AS catalog_opclass ON catalog_opclass.oid = index_opclass.opclass_oid
                        JOIN pg_catalog.pg_amop AS family_operator
                            ON family_operator.amopfamily = catalog_opclass.opcfamily
                            AND family_operator.amopmethod = catalog_opclass.opcmethod
                            AND family_operator.amoplefttype = catalog_opclass.opcintype
                            AND family_operator.amoprighttype = catalog_opclass.opcintype
                        JOIN pg_catalog.pg_operator AS catalog_operator
                            ON catalog_operator.oid = family_operator.amopopr
                        JOIN pg_catalog.pg_namespace AS operator_namespace
                            ON operator_namespace.oid = catalog_operator.oprnamespace
                        JOIN pg_catalog.pg_proc AS operator_procedure
                            ON operator_procedure.oid = catalog_operator.oprcode
                        JOIN pg_catalog.pg_namespace AS operator_procedure_namespace
                            ON operator_procedure_namespace.oid = operator_procedure.pronamespace
                        WHERE operator_namespace.nspname <> :trusted_schema
                           OR operator_procedure_namespace.nspname <> :trusted_schema
                           OR operator_procedure.provolatile <> 'i'::"char"
                    ) AS trusted_family_operators,
                    NOT EXISTS (
                        SELECT 1
                        FROM pg_catalog.pg_depend AS dependency
                        LEFT JOIN pg_catalog.pg_proc AS procedure
                            ON dependency.refclassid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                            AND procedure.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_namespace AS procedure_namespace
                            ON procedure_namespace.oid = procedure.pronamespace
                        LEFT JOIN pg_catalog.pg_operator AS catalog_operator
                            ON dependency.refclassid = 'pg_catalog.pg_operator'::pg_catalog.regclass
                            AND catalog_operator.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_namespace AS operator_namespace
                            ON operator_namespace.oid = catalog_operator.oprnamespace
                        LEFT JOIN pg_catalog.pg_proc AS operator_procedure
                            ON operator_procedure.oid = catalog_operator.oprcode
                        LEFT JOIN pg_catalog.pg_namespace AS operator_procedure_namespace
                            ON operator_procedure_namespace.oid = operator_procedure.pronamespace
                        LEFT JOIN pg_catalog.pg_opclass AS catalog_opclass
                            ON dependency.refclassid = 'pg_catalog.pg_opclass'::pg_catalog.regclass
                            AND catalog_opclass.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_namespace AS opclass_namespace
                            ON opclass_namespace.oid = catalog_opclass.opcnamespace
                        LEFT JOIN pg_catalog.pg_collation AS catalog_collation
                            ON dependency.refclassid = 'pg_catalog.pg_collation'::pg_catalog.regclass
                            AND catalog_collation.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_namespace AS collation_namespace
                            ON collation_namespace.oid = catalog_collation.collnamespace
                        LEFT JOIN pg_catalog.pg_type AS type_data
                            ON dependency.refclassid = 'pg_catalog.pg_type'::pg_catalog.regclass
                            AND type_data.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_namespace AS type_namespace
                            ON type_namespace.oid = type_data.typnamespace
                        LEFT JOIN pg_catalog.pg_cast AS cast_data
                            ON dependency.refclassid = 'pg_catalog.pg_cast'::pg_catalog.regclass
                            AND cast_data.oid = dependency.refobjid
                        LEFT JOIN pg_catalog.pg_proc AS cast_procedure ON cast_procedure.oid = cast_data.castfunc
                        LEFT JOIN pg_catalog.pg_namespace AS cast_procedure_namespace
                            ON cast_procedure_namespace.oid = cast_procedure.pronamespace
                        WHERE dependency.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
                          AND dependency.objid = index_relation.oid
                          AND (
                              (procedure.oid IS NOT NULL AND (
                                  procedure_namespace.nspname <> :trusted_schema
                                  OR procedure.provolatile <> 'i'::"char"
                              )) OR (catalog_operator.oid IS NOT NULL AND (
                                  operator_namespace.nspname <> :trusted_schema
                                  OR operator_procedure_namespace.nspname <> :trusted_schema
                                  OR operator_procedure.provolatile <> 'i'::"char"
                              )) OR (
                                  catalog_opclass.oid IS NOT NULL
                                  AND opclass_namespace.nspname <> :trusted_schema
                              ) OR (
                                  catalog_collation.oid IS NOT NULL
                                  AND collation_namespace.nspname <> :trusted_schema
                              ) OR (type_data.oid IS NOT NULL AND type_namespace.nspname <> :trusted_schema)
                                OR (cast_data.oid IS NOT NULL AND cast_procedure.oid IS NOT NULL AND (
                                    cast_procedure_namespace.nspname <> :trusted_schema
                                    OR cast_procedure.provolatile <> 'i'::"char"
                                ))
                          )
                    ) AS trusted_dependencies
                FROM pg_catalog.pg_index AS index_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = index_data.indrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_data.indexrelid
                JOIN pg_catalog.pg_am AS access_method ON access_method.oid = index_relation.relam
                WHERE namespace.nspname = :schema AND relation.relname IN :table_names
                """
            ).bindparams(sa.bindparam("table_names", expanding=True)),
            {"schema": SCHEMA, "table_names": TABLES, "trusted_schema": "pg_catalog"},
        )
        .mappings()
    )
    for row in rows:
        if (
            row["relkind"] not in {"i", b"i"}
            or row["relpersistence"] not in {"p", b"p"}
            or row["reloptions"]
            or row["indisvalid"] is not True
            or row["indisready"] is not True
            or row["indislive"] is not True
            or row["indisexclusion"] is not False
            or row["indimmediate"] is not True
            or row["indcheckxmin"] is not False
            or row["no_predicate"] is not True
            or row["no_expressions"] is not True
            or row["indnkeyatts"] != row["indnatts"]
            or row["amname"] != "btree"
            or row["ordinary_key_columns"] is not True
            or row["ordinary_key_options"] is not True
            or row["trusted_opclasses"] is not True
            or row["trusted_collations"] is not True
            or row["trusted_support_procedures"] is not True
            or row["trusted_family_operators"] is not True
            or row["trusted_dependencies"] is not True
        ):
            raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _require_static_schema(*, head: bool, payload_name: str, digest_name: str | None) -> None:
    _require_columns(
        "runs",
        {
            "run_id": (False, _is_uuid, None),
            "request_id": (False, _is_uuid, None),
            "correlation_id": (False, _is_uuid, None),
            "source": (False, _is_varchar(32), None),
            "status": (False, _is_varchar(32), None),
            "envelope": (False, _is_jsonb, None),
            "error_code": (True, _is_varchar(64), None),
            "created_at": (False, _is_timestamptz, "now()"),
            "updated_at": (False, _is_timestamptz, "now()"),
        },
    )
    _require_columns(
        "run_events",
        {
            "run_id": (False, _is_uuid, None),
            "sequence": (False, _is_integer, None),
            "event_type": (False, _is_varchar(64), None),
            "occurred_at": (False, _is_timestamptz, None),
            "component": (False, _is_varchar(256), None),
            "redacted_payload": (False, _is_jsonb, None),
            "previous_hash": (True, _is_varchar(64), None),
            "current_hash": (False, _is_varchar(64), None),
        },
    )
    analysis_columns = {
        "run_id": (False, _is_uuid, None),
        payload_name: (False, _is_jsonb, None),
        "created_at": (False, _is_timestamptz, "now()"),
    }
    if digest_name is not None:
        analysis_columns[digest_name] = (False, _is_varchar_sha256, None)
    _require_columns("analyses", analysis_columns)
    _require_key_and_fk_contract(
        "runs",
        primary_key=("run_id",),
        foreign_keys=(),
        unique_names={"uq_runs_request_id": ("request_id",)},
    )
    _require_key_and_fk_contract(
        "run_events",
        primary_key=("run_id", "sequence"),
        foreign_keys=((("run_id",), "runs", ("run_id",)),),
        unique_names=({"uq_run_events_run_id_event_hash": ("run_id", "current_hash")} if head else {}),
    )
    _require_key_and_fk_contract(
        "analyses",
        primary_key=("run_id",),
        foreign_keys=((("run_id",), "runs", ("run_id",)),),
        unique_names={},
    )
    _require_indexes(
        "runs",
        {"ix_runs_correlation_id": ("correlation_id",), "ix_runs_status_created_at": ("status", "created_at")},
    )
    _require_indexes("run_events", {"ix_run_events_run_id_event_type": ("run_id", "event_type")})
    _require_all_index_contracts()


def _require_non_nullable(column: dict[str, Any], predicate: bool) -> None:
    if (
        column.get("nullable") is not False
        or column.get("_catalog_not_null") is not True
        or not _is_catalog_empty_marker(column.get("_catalog_identity"))
        or not _is_catalog_empty_marker(column.get("_catalog_generated"))
        or column.get("_catalog_default_collation") is not True
        or not predicate
    ):
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _require_analysis_key_and_fk() -> None:
    _require_key_and_fk_contract(
        "analyses",
        primary_key=("run_id",),
        foreign_keys=((("run_id",), "runs", ("run_id",)),),
        unique_names={},
    )


def _validated_upgrade_analysis_shape() -> tuple[str, str | None]:
    columns = _columns("analyses")
    payload_columns = {"result", "sanitized_result"}.intersection(columns)
    digest_columns = {"result_sha256", "sanitized_result_sha256"}.intersection(columns)
    expected_columns = {"run_id", "created_at", *payload_columns, *digest_columns}
    if len(payload_columns) != 1 or len(digest_columns) > 1 or set(columns) != expected_columns:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)

    payload_name = next(iter(payload_columns))
    digest_name = next(iter(digest_columns), None)
    _require_non_nullable(columns["run_id"], _is_uuid(columns["run_id"]))
    _require_non_nullable(columns[payload_name], _is_jsonb(columns[payload_name]))
    _require_non_nullable(columns["created_at"], _is_timestamptz_now(columns["created_at"]))
    if digest_name is not None:
        _require_non_nullable(columns[digest_name], _is_varchar_sha256(columns[digest_name]))
    _require_analysis_key_and_fk()
    _validate_all_checks(payload_name, digest_name, head=payload_name == "sanitized_result")
    return payload_name, digest_name


def _require_head_analysis_shape() -> None:
    columns = _columns("analyses")
    if set(columns) != {"run_id", "sanitized_result", "sanitized_result_sha256", "created_at"}:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)
    _require_non_nullable(columns["run_id"], _is_uuid(columns["run_id"]))
    _require_non_nullable(columns["sanitized_result"], _is_jsonb(columns["sanitized_result"]))
    _require_non_nullable(
        columns["sanitized_result_sha256"],
        _is_varchar_sha256(columns["sanitized_result_sha256"]),
    )
    _require_non_nullable(columns["created_at"], _is_timestamptz_now(columns["created_at"]))
    _require_analysis_key_and_fk()
    _validate_all_checks("sanitized_result", "sanitized_result_sha256", head=True)


def _drop_check_names(table_name: str, names: Iterable[str]) -> None:
    existing = _constraints(table_name)
    seen_names: set[str] = set()
    for name in names:
        if name in seen_names:
            continue
        seen_names.add(name)
        if name in existing:
            op.drop_constraint(op.f(name), table_name, type_="check", schema=SCHEMA)


def _replace_checks(table_name: str, checks: Iterable[tuple[str, str, str]], *, baseline: bool) -> None:
    check_list = tuple(checks)
    _drop_check_names(table_name, (name for _, first, second in check_list for name in (first, second)))
    for expression, baseline_name, head_name in check_list:
        name = baseline_name if baseline else head_name
        op.create_check_constraint(op.f(name), table_name, expression, schema=SCHEMA)


def _reserved_public_relation_exists(name: str) -> bool:
    return bool(
        op.get_bind().scalar(
            sa.text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = :schema AND relation.relname = :name
                )
                """
            ),
            {"schema": SCHEMA, "name": name},
        )
    )


def _owned_event_unique_exists() -> bool:
    name = "uq_run_events_run_id_event_hash"
    constraints = _constraints("run_events")
    constraint = constraints.get(name)
    if constraint is None:
        if _reserved_public_relation_exists(name):
            raise RuntimeError(_INVALID_EVENT_UNIQUE_ERROR)
        return False
    if not _ordinary_validated_constraint(constraint, "u", ("run_id", "current_hash")) or not constraint.get(
        "index_oid"
    ):
        raise RuntimeError(_INVALID_EVENT_UNIQUE_ERROR)
    try:
        _require_index_contract(
            "run_events",
            name,
            ("run_id", "current_hash"),
            unique=True,
            constraint_owned=True,
        )
    except RuntimeError:
        raise RuntimeError(_INVALID_EVENT_UNIQUE_ERROR) from None
    return True


def _all_check_names(checks: Iterable[tuple[str, str, str]]) -> tuple[str, ...]:
    return tuple(name for _, first, second in checks for name in (first, second))


def _require_no_mutated_table_custom_checks() -> None:
    """Reject unknown CHECK constraints before changing rows or columns they can inspect."""
    has_custom_check = op.get_bind().scalar(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_constraint AS constraint_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = constraint_data.conrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema
                  AND constraint_data.contype = 'c'::"char"
                  AND (
                      (relation.relname = 'runs' AND constraint_data.conname NOT IN :run_check_names)
                      OR (
                          relation.relname = 'analyses'
                          AND constraint_data.conname NOT IN :analysis_check_names
                      )
                  )
            )
            """
        ).bindparams(
            sa.bindparam("run_check_names", expanding=True),
            sa.bindparam("analysis_check_names", expanding=True),
        ),
        {
            "schema": SCHEMA,
            "run_check_names": _RUN_CHECK_NAMES,
            "analysis_check_names": _ANALYSIS_CHECK_NAMES,
        },
    )
    if has_custom_check:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _require_trusted_expression_dependencies() -> None:
    """Reject unsafe dependencies from every managed CHECK and managed default expression."""
    has_untrusted_dependency = op.get_bind().scalar(
        sa.text(
            """
            WITH expression_objects AS (
                SELECT constraint_data.oid AS object_id,
                    'pg_catalog.pg_constraint'::pg_catalog.regclass AS class_id,
                    true AS is_check
                FROM pg_catalog.pg_constraint AS constraint_data
                JOIN pg_catalog.pg_class AS relation ON relation.oid = constraint_data.conrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema
                  AND relation.relname IN :table_names
                  AND constraint_data.contype = 'c'::"char"
                UNION ALL
                SELECT default_value.oid AS object_id,
                    'pg_catalog.pg_attrdef'::pg_catalog.regclass AS class_id,
                    false AS is_check
                FROM pg_catalog.pg_attrdef AS default_value
                JOIN pg_catalog.pg_class AS relation ON relation.oid = default_value.adrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = :schema AND relation.relname IN :table_names
            )
            SELECT EXISTS (
                SELECT 1
                FROM expression_objects AS expression_object
                JOIN pg_catalog.pg_depend AS dependency
                    ON dependency.classid = expression_object.class_id
                    AND dependency.objid = expression_object.object_id
                LEFT JOIN pg_catalog.pg_proc AS procedure
                    ON dependency.refclassid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                    AND procedure.oid = dependency.refobjid
                LEFT JOIN pg_catalog.pg_namespace AS procedure_namespace
                    ON procedure_namespace.oid = procedure.pronamespace
                LEFT JOIN pg_catalog.pg_operator AS catalog_operator
                    ON dependency.refclassid = 'pg_catalog.pg_operator'::pg_catalog.regclass
                    AND catalog_operator.oid = dependency.refobjid
                LEFT JOIN pg_catalog.pg_namespace AS operator_namespace
                    ON operator_namespace.oid = catalog_operator.oprnamespace
                LEFT JOIN pg_catalog.pg_proc AS operator_procedure
                    ON operator_procedure.oid = catalog_operator.oprcode
                LEFT JOIN pg_catalog.pg_namespace AS operator_procedure_namespace
                    ON operator_procedure_namespace.oid = operator_procedure.pronamespace
                LEFT JOIN pg_catalog.pg_type AS type_data
                    ON dependency.refclassid = 'pg_catalog.pg_type'::pg_catalog.regclass
                    AND type_data.oid = dependency.refobjid
                LEFT JOIN pg_catalog.pg_namespace AS type_namespace
                    ON type_namespace.oid = type_data.typnamespace
                LEFT JOIN pg_catalog.pg_collation AS collation_data
                    ON dependency.refclassid = 'pg_catalog.pg_collation'::pg_catalog.regclass
                    AND collation_data.oid = dependency.refobjid
                LEFT JOIN pg_catalog.pg_namespace AS collation_namespace
                    ON collation_namespace.oid = collation_data.collnamespace
                LEFT JOIN pg_catalog.pg_cast AS cast_data
                    ON dependency.refclassid = 'pg_catalog.pg_cast'::pg_catalog.regclass
                    AND cast_data.oid = dependency.refobjid
                LEFT JOIN pg_catalog.pg_proc AS cast_procedure ON cast_procedure.oid = cast_data.castfunc
                LEFT JOIN pg_catalog.pg_namespace AS cast_procedure_namespace
                    ON cast_procedure_namespace.oid = cast_procedure.pronamespace
                WHERE (
                    procedure.oid IS NOT NULL
                    AND (
                        procedure_namespace.nspname <> :trusted_schema
                        OR (expression_object.is_check AND procedure.provolatile <> 'i'::"char")
                    )
                ) OR (
                    catalog_operator.oid IS NOT NULL
                    AND (
                        operator_namespace.nspname <> :trusted_schema
                        OR operator_procedure_namespace.nspname <> :trusted_schema
                        OR (expression_object.is_check AND operator_procedure.provolatile <> 'i'::"char")
                    )
                ) OR (type_data.oid IS NOT NULL AND type_namespace.nspname <> :trusted_schema)
                  OR (collation_data.oid IS NOT NULL AND collation_namespace.nspname <> :trusted_schema)
                  OR (
                    cast_data.oid IS NOT NULL
                    AND cast_procedure.oid IS NOT NULL
                    AND (
                        cast_procedure_namespace.nspname <> :trusted_schema
                        OR (expression_object.is_check AND cast_procedure.provolatile <> 'i'::"char")
                    )
                )
            )
            """
        ).bindparams(sa.bindparam("table_names", expanding=True)),
        {"schema": SCHEMA, "table_names": TABLES, "trusted_schema": "pg_catalog"},
    )
    if has_untrusted_dependency:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def _create_checks(table_name: str, checks: Iterable[tuple[str, str, str]], *, baseline: bool) -> None:
    for expression, baseline_name, head_name in checks:
        op.create_check_constraint(
            op.f(baseline_name if baseline else head_name),
            table_name,
            expression,
            schema=SCHEMA,
        )


def _upgrade_analyses(payload_name: str, digest_name: str | None) -> None:
    _drop_check_names("analyses", _all_check_names(_ANALYSIS_CHECKS.values()))
    if payload_name == "result":
        op.alter_column(
            "analyses",
            "result",
            new_column_name="sanitized_result",
            existing_type=postgresql.JSONB(),
            existing_nullable=False,
            schema=SCHEMA,
        )
    if digest_name == "result_sha256":
        op.alter_column(
            "analyses",
            "result_sha256",
            new_column_name="sanitized_result_sha256",
            existing_type=sa.VARCHAR(length=64),
            existing_nullable=False,
            schema=SCHEMA,
        )
    elif digest_name is None:
        op.add_column(
            "analyses",
            sa.Column("sanitized_result_sha256", sa.VARCHAR(length=64), nullable=False),
            schema=SCHEMA,
        )
    op.execute(sa.text("ALTER TABLE public.analyses ALTER COLUMN created_at SET DEFAULT pg_catalog.now()"))
    _create_checks(
        "analyses",
        (_ANALYSIS_CHECKS["sanitized_object"], _ANALYSIS_CHECKS["sanitized_digest"]),
        baseline=False,
    )


def _execute_managed_update(statement: str) -> None:
    try:
        op.execute(sa.text(statement))
    except DBAPIError:
        raise RuntimeError(_MANAGED_UPDATE_ERROR) from None


def _upgrade_runs() -> None:
    _drop_check_names(
        "runs",
        _all_check_names((_RUN_CHECKS["source"], _RUN_CHECKS["status_baseline"], _RUN_CHECKS["envelope"])),
    )
    _execute_managed_update("UPDATE public.runs SET status = 'queued' WHERE status = 'pending'")
    _execute_managed_update("UPDATE public.runs SET status = 'processing' WHERE status = 'running'")
    _create_checks(
        "runs",
        (_RUN_CHECKS["source"], _RUN_CHECKS["status_head"], _RUN_CHECKS["envelope"]),
        baseline=False,
    )


def upgrade() -> None:
    _require_online_postgresql()
    _set_trusted_search_path()
    _lock_and_require_empty_analyses()
    event_unique_exists = _owned_event_unique_exists()
    _require_inbound_foreign_key_and_internal_trigger_contract()
    _require_no_active_trigger_or_rule_contract()
    _require_no_mutated_table_custom_checks()
    _require_trusted_expression_dependencies()
    payload_name, digest_name = _validated_upgrade_analysis_shape()
    _require_static_schema(
        head=payload_name == "sanitized_result",
        payload_name=payload_name,
        digest_name=digest_name,
    )

    _upgrade_analyses(payload_name, digest_name)
    _upgrade_runs()
    _replace_checks("run_events", _EVENT_CHECKS.values(), baseline=False)
    if not event_unique_exists:
        op.create_unique_constraint(
            op.f("uq_run_events_run_id_event_hash"),
            "run_events",
            ["run_id", "current_hash"],
            schema=SCHEMA,
        )
    op.execute(sa.text("SET LOCAL search_path TO public"))


def _downgrade_analyses() -> None:
    _drop_check_names("analyses", _all_check_names(_ANALYSIS_CHECKS.values()))
    op.alter_column(
        "analyses",
        "sanitized_result",
        new_column_name="result",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        schema=SCHEMA,
    )
    op.alter_column(
        "analyses",
        "sanitized_result_sha256",
        new_column_name="result_sha256",
        existing_type=sa.VARCHAR(length=64),
        existing_nullable=False,
        schema=SCHEMA,
    )
    op.execute(sa.text("ALTER TABLE public.analyses ALTER COLUMN created_at SET DEFAULT pg_catalog.now()"))
    _create_checks(
        "analyses",
        (_ANALYSIS_CHECKS["result_object"], _ANALYSIS_CHECKS["result_digest"]),
        baseline=True,
    )


def _downgrade_runs() -> None:
    _drop_check_names(
        "runs",
        _all_check_names((_RUN_CHECKS["source"], _RUN_CHECKS["status_head"], _RUN_CHECKS["envelope"])),
    )
    _execute_managed_update("UPDATE public.runs SET status = 'pending' WHERE status = 'queued'")
    _execute_managed_update("UPDATE public.runs SET status = 'running' WHERE status = 'processing'")
    _create_checks(
        "runs",
        (_RUN_CHECKS["source"], _RUN_CHECKS["status_baseline"], _RUN_CHECKS["envelope"]),
        baseline=True,
    )


def _require_head_contract() -> None:
    """Validate the already-head schema without reading or transforming managed data."""
    _require_online_postgresql()
    _set_trusted_search_path()
    _require_ordinary_unprotected_relations()
    _lock_managed_tables()
    _require_ordinary_unprotected_relations()
    event_unique_exists = _owned_event_unique_exists()
    _require_inbound_foreign_key_and_internal_trigger_contract()
    _require_no_active_trigger_or_rule_contract()
    _require_no_mutated_table_custom_checks()
    _require_trusted_expression_dependencies()
    _require_head_analysis_shape()
    _require_static_schema(
        head=True,
        payload_name="sanitized_result",
        digest_name="sanitized_result_sha256",
    )
    if not event_unique_exists:
        raise RuntimeError(_UNSUPPORTED_SCHEMA_ERROR)


def downgrade() -> None:
    _require_online_postgresql()
    _set_trusted_search_path()
    _lock_and_require_empty_analyses()
    event_unique_exists = _owned_event_unique_exists()
    _require_inbound_foreign_key_and_internal_trigger_contract()
    _require_no_active_trigger_or_rule_contract()
    _require_no_mutated_table_custom_checks()
    _require_trusted_expression_dependencies()
    _require_head_analysis_shape()
    _require_static_schema(
        head=True,
        payload_name="sanitized_result",
        digest_name="sanitized_result_sha256",
    )

    if event_unique_exists:
        op.drop_constraint(
            op.f("uq_run_events_run_id_event_hash"),
            "run_events",
            type_="unique",
            schema=SCHEMA,
        )
    _downgrade_analyses()
    _downgrade_runs()
    _replace_checks("run_events", _EVENT_CHECKS.values(), baseline=True)
    op.execute(sa.text("SET LOCAL search_path TO public"))
