from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import traceback
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from mtbank_ai.domain.events import EventAttribute, LifecycleEventType, RedactedPayload, RunEvent
from mtbank_ai.storage.canonical import canonical_json_sha256

ROOT = Path(__file__).parents[2]

_HEAD_CHECKS = {
    "runs": frozenset({"ck_runs_source_allowed", "ck_runs_status_allowed", "ck_runs_envelope_object"}),
    "run_events": frozenset(
        {
            "ck_run_events_sequence_positive",
            "ck_run_events_current_hash_sha256",
            "ck_run_events_previous_hash_sha256",
            "ck_run_events_hash_chain_position",
            "ck_run_events_redacted_payload_object",
        }
    ),
    "analyses": frozenset({"ck_analyses_sanitized_result_object", "ck_analyses_sanitized_result_sha256"}),
}
_BASELINE_CHECKS = {
    "runs": frozenset(
        {
            "ck_runs_ck_runs_source_allowed",
            "ck_runs_ck_runs_status_allowed",
            "ck_runs_ck_runs_envelope_object",
        }
    ),
    "run_events": frozenset(
        {
            "ck_run_events_ck_run_events_sequence_positive",
            "ck_run_events_ck_run_events_current_hash_sha256",
            "ck_run_events_ck_run_events_previous_hash_sha256",
            "ck_run_events_ck_run_events_hash_chain_position",
            "ck_run_events_ck_run_events_redacted_payload_object",
        }
    ),
    "analyses": frozenset(
        {
            "ck_analyses_ck_analyses_result_object",
            "ck_analyses_ck_analyses_result_sha256",
        }
    ),
}


class DestructiveDatabaseGuardError(ValueError):
    pass


_AMBIENT_TARGET_VARIABLES = (
    "PGHOST",
    "PGPORT",
    "PGUSER",
    "PGDATABASE",
    "PGSERVICE",
    "PGSERVICEFILE",
)


def _validate_destructive_database_target(value: str, opt_in: str | None) -> tuple[URL, str]:
    if opt_in != "1":
        raise DestructiveDatabaseGuardError("нужен явный MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS=1")
    if not isinstance(value, str) or not value or value != value.strip():
        raise DestructiveDatabaseGuardError("MTBANK_TEST_DATABASE_URL имеет недопустимый формат")
    try:
        parsed = urlsplit(value)
        if parsed.query:
            raise DestructiveDatabaseGuardError("MTBANK_TEST_DATABASE_URL не должен содержать query parameters")
        if parsed.fragment or parsed.netloc.count("@") != 1:
            raise DestructiveDatabaseGuardError("MTBANK_TEST_DATABASE_URL имеет недопустимый формат")
        url = make_url(value)
        port = url.port
    except DestructiveDatabaseGuardError:
        raise
    except Exception:
        raise DestructiveDatabaseGuardError("MTBANK_TEST_DATABASE_URL имеет недопустимый формат") from None
    if url.drivername != "postgresql+asyncpg":
        raise DestructiveDatabaseGuardError("migration test требует dialect postgresql+asyncpg")
    if not url.host or port is None or not url.username or url.password is None or not url.database:
        raise DestructiveDatabaseGuardError("MTBANK_TEST_DATABASE_URL требует explicit authority и database")
    database_name = url.database
    if not (database_name.startswith("mtbank_test_") or database_name.endswith("_test")):
        raise DestructiveDatabaseGuardError("migration test разрешён только для database mtbank_test_* или *_test")
    return url, database_name


def _database_target(environment: Mapping[str, str | None]) -> tuple[URL, str]:
    value_present = "MTBANK_TEST_DATABASE_URL" in environment
    opt_in_present = "MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS" in environment
    if not value_present and not opt_in_present:
        pytest.skip("MTBANK_TEST_DATABASE_URL и MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS не заданы")
    if not value_present or not opt_in_present:
        pytest.fail("destructive migration test требует URL и explicit opt-in", pytrace=False)
    if any(environment.get(name) is not None for name in _AMBIENT_TARGET_VARIABLES):
        pytest.fail("destructive migration test запрещает ambient PostgreSQL target settings", pytrace=False)
    try:
        return _validate_destructive_database_target(
            environment.get("MTBANK_TEST_DATABASE_URL") or "",
            environment.get("MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS"),
        )
    except DestructiveDatabaseGuardError as error:
        pytest.fail(str(error), pytrace=False)


async def _verify_current_database(url: URL, expected_database: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            actual_database = await connection.scalar(text("SELECT current_database()"))
    except Exception:
        raise DestructiveDatabaseGuardError("не удалось безопасно подтвердить имя disposable test database") from None
    finally:
        await engine.dispose()
    if actual_database != expected_database:
        raise DestructiveDatabaseGuardError(
            "current_database() не совпадает с database path из MTBANK_TEST_DATABASE_URL"
        )


def _alembic_config(url: URL, expected_database: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    rendered = url.render_as_string(hide_password=False)
    config.set_main_option("sqlalchemy.url", rendered.replace("%", "%%"))
    config.attributes["mtbank_expected_database"] = expected_database
    return config


def _plain_schema(connection: Any) -> dict[str, Any]:
    inspector = inspect(connection)
    return {
        "tables": tuple(sorted(inspector.get_table_names(schema="public"))),
        "analyses_columns": frozenset(column["name"] for column in inspector.get_columns("analyses", schema="public")),
        "analyses_defaults": {
            column["name"]: column.get("default") for column in inspector.get_columns("analyses", schema="public")
        },
        "checks": {
            table_name: frozenset(
                constraint["name"]
                for constraint in inspector.get_check_constraints(table_name, schema="public")
                if isinstance(constraint.get("name"), str)
            )
            for table_name in ("runs", "run_events", "analyses")
        },
        "check_definitions": {
            table_name: {
                str(constraint["name"]): str(constraint["sqltext"])
                for constraint in inspector.get_check_constraints(table_name, schema="public")
                if isinstance(constraint.get("name"), str)
            }
            for table_name in ("runs", "run_events", "analyses")
        },
        "event_uniques": tuple(
            sorted(
                (
                    constraint["name"],
                    tuple(constraint.get("column_names") or ()),
                )
                for constraint in inspector.get_unique_constraints("run_events")
                if isinstance(constraint.get("name"), str)
            )
        ),
    }


async def _schema(url: URL) -> dict[str, Any]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(_plain_schema)
    finally:
        await engine.dispose()


async def _revision(url: URL) -> str | None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return await connection.scalar(text("SELECT version_num FROM public.alembic_version"))
    finally:
        await engine.dispose()


_REJECTION_SECRET = "migration-secret-sentinel"


async def _managed_catalog_snapshot(url: URL) -> dict[str, tuple[tuple[Any, ...], ...]]:
    statements = {
        "relations": """
            SELECT relation.relname, relation.relkind::text, relation.relpersistence::text,
                relation.relispartition, relation.relrowsecurity, relation.relforcerowsecurity,
                relation.relhassubclass, relation.reloftype
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
            ORDER BY relation.relname
        """,
        "columns": """
            SELECT relation.relname, attribute.attnum, attribute.attname,
                pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
                attribute.attnotnull, attribute.attidentity, attribute.attgenerated,
                attribute.attcollation, pg_catalog.pg_get_expr(default_value.adbin, default_value.adrelid, false)
            FROM pg_catalog.pg_attribute AS attribute
            JOIN pg_catalog.pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            LEFT JOIN pg_catalog.pg_attrdef AS default_value
                ON default_value.adrelid = attribute.attrelid AND default_value.adnum = attribute.attnum
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            ORDER BY relation.relname, attribute.attnum
        """,
        "constraints": """
            SELECT relation.relname, catalog_constraint.conname, catalog_constraint.contype::text,
                catalog_constraint.condeferrable, catalog_constraint.condeferred, catalog_constraint.convalidated,
                catalog_constraint.connoinherit, catalog_constraint.coninhcount, catalog_constraint.conparentid,
                pg_catalog.pg_get_constraintdef(catalog_constraint.oid, false)
            FROM pg_catalog.pg_constraint AS catalog_constraint
            JOIN pg_catalog.pg_class AS relation ON relation.oid = catalog_constraint.conrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
            ORDER BY relation.relname, catalog_constraint.conname
        """,
        "indexes": """
            SELECT relation.relname, index_relation.relname, index_data.indisvalid,
                index_data.indisready, index_data.indisunique, index_data.indnkeyatts,
                index_data.indnatts, pg_catalog.pg_get_indexdef(index_data.indexrelid)
            FROM pg_catalog.pg_index AS index_data
            JOIN pg_catalog.pg_class AS relation ON relation.oid = index_data.indrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_data.indexrelid
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
            ORDER BY relation.relname, index_relation.relname
        """,
        "triggers": """
            SELECT relation.relname, trigger_data.tgname, trigger_data.tgenabled::text,
                trigger_data.tgisinternal, trigger_data.tgtype, trigger_data.tgfoid,
                pg_catalog.pg_get_triggerdef(trigger_data.oid, false)
            FROM pg_catalog.pg_trigger AS trigger_data
            JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_data.tgrelid
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
            ORDER BY relation.relname, trigger_data.tgname
        """,
        "rules": """
            SELECT relation.relname, rule_data.rulename, rule_data.ev_type::text,
                rule_data.ev_enabled::text, rule_data.is_instead,
                pg_catalog.pg_get_ruledef(rule_data.oid, false)
            FROM pg_catalog.pg_rewrite AS rule_data
            JOIN pg_catalog.pg_class AS relation ON relation.oid = rule_data.ev_class
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('runs', 'run_events', 'analyses')
            ORDER BY relation.relname, rule_data.rulename
        """,
        "inheritance": """
            SELECT parent_relation.relname, child_relation.relname
            FROM pg_catalog.pg_inherits AS inheritance
            JOIN pg_catalog.pg_class AS parent_relation ON parent_relation.oid = inheritance.inhparent
            JOIN pg_catalog.pg_namespace AS parent_namespace ON parent_namespace.oid = parent_relation.relnamespace
            JOIN pg_catalog.pg_class AS child_relation ON child_relation.oid = inheritance.inhrelid
            JOIN pg_catalog.pg_namespace AS child_namespace ON child_namespace.oid = child_relation.relnamespace
            WHERE parent_namespace.nspname = 'public'
              AND child_namespace.nspname = 'public'
              AND (parent_relation.relname IN ('runs', 'run_events', 'analyses')
                   OR child_relation.relname IN ('runs', 'run_events', 'analyses'))
            ORDER BY parent_relation.relname, child_relation.relname
        """,
    }
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            await connection.execute(text("SET LOCAL search_path TO pg_catalog"))
            snapshot: dict[str, tuple[tuple[Any, ...], ...]] = {}
            for name, statement in statements.items():
                rows = await connection.execute(text(statement))
                snapshot[name] = tuple(tuple(row) for row in rows)
            return snapshot
    finally:
        await engine.dispose()


def _assert_rejected_without_managed_ddl(
    url: URL,
    config: Config,
    operation: Callable[[], None],
    *,
    error_match: str = "unsupported managed schema",
) -> None:
    before_revision = asyncio.run(_revision(url))
    before_catalog = asyncio.run(_managed_catalog_snapshot(url))

    with pytest.raises(RuntimeError, match=error_match) as error:
        operation()

    error_text = str(error.value)
    assert _REJECTION_SECRET not in error_text
    assert asyncio.run(_revision(url)) == before_revision
    assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog


def _assert_rejected_without_side_effects(
    url: URL,
    config: Config,
    operation: Callable[[], None],
    *,
    error_match: str = "unsupported managed schema",
) -> None:
    _assert_rejected_without_managed_ddl(url, config, operation, error_match=error_match)
    assert asyncio.run(_execute_and_scalar(url, "SELECT calls FROM public.mtbank_migration_side_effects", {})) == 0


async def _execute(url: URL, statement: str, parameters: dict[str, object] | None = None) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(statement), parameters or {})
    finally:
        await engine.dispose()


async def _execute_with_search_path(url: URL, statement: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET LOCAL search_path TO public, pg_catalog"))
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def _execute_with_trusted_search_path(url: URL, statement: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET LOCAL search_path TO pg_catalog, public"))
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def _execute_autocommit(url: URL, statement: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            await connection.execution_options(isolation_level="AUTOCOMMIT")
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _insert_run(
    connection: AsyncConnection,
    run_id: UUID,
    *,
    status: str = "pending",
    correlation_id: UUID | None = None,
    envelope: str = "{}",
) -> None:
    await connection.execute(
        text(
            """
            INSERT INTO public.runs(run_id, request_id, correlation_id, source, status, envelope)
            VALUES (:run_id, :request_id, :correlation_id, 'rest_file', :status, CAST(:envelope AS jsonb))
            """
        ),
        {
            "run_id": run_id,
            "request_id": uuid4(),
            "correlation_id": correlation_id or uuid4(),
            "status": status,
            "envelope": envelope,
        },
    )


async def _insert_pending_run(url: URL) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await _insert_run(connection, uuid4(), status="pending")
    finally:
        await engine.dispose()


async def _insert_event(
    connection: AsyncConnection,
    run_id: UUID,
    *,
    sequence: int,
    current_hash: str,
    previous_hash: str | None,
) -> None:
    event = RunEvent(
        run_id=run_id,
        sequence=sequence,
        event_type=(LifecycleEventType.RUN_STARTED if sequence == 1 else LifecycleEventType.RUN_COMPLETED),
        occurred_at=datetime.now(UTC),
        component="workflow",
        payload=RedactedPayload(fields=(EventAttribute(key="attempt", value=sequence),)),
        previous_hash=previous_hash,
        current_hash=current_hash,
    )
    python_values = event.model_dump()
    payload = json.dumps(event.model_dump(mode="json")["payload"])
    await connection.execute(
        text(
            """
            INSERT INTO public.run_events(
                run_id, sequence, event_type, occurred_at, component,
                redacted_payload, previous_hash, current_hash
            ) VALUES (
                :run_id, :sequence, :event_type, :occurred_at, :component,
                CAST(:payload AS jsonb), :previous_hash, :current_hash
            )
            """
        ),
        {**python_values, "payload": payload},
    )


async def _insert_legacy_analysis(url: URL, run_id: UUID, payload: dict[str, object]) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await _insert_run(connection, run_id)
            await connection.execute(
                text(
                    """
                    INSERT INTO public.analyses(run_id, result, result_sha256)
                    VALUES (:run_id, CAST(:result AS jsonb), :digest)
                    """
                ),
                {
                    "run_id": run_id,
                    "result": json.dumps(payload),
                    "digest": canonical_json_sha256(payload),
                },
            )
    finally:
        await engine.dispose()


async def _insert_head_analysis(url: URL, run_id: UUID, payload: dict[str, object]) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await _insert_run(connection, run_id, status="queued")
            await connection.execute(
                text(
                    """
                    INSERT INTO public.analyses(run_id, sanitized_result, sanitized_result_sha256)
                    VALUES (:run_id, CAST(:result AS jsonb), :digest)
                    """
                ),
                {
                    "run_id": run_id,
                    "result": json.dumps(payload),
                    "digest": canonical_json_sha256(payload),
                },
            )
    finally:
        await engine.dispose()


async def _analysis_payload(url: URL, column_name: str, run_id: UUID) -> object:
    if column_name not in {"result", "sanitized_result"}:
        raise AssertionError("unexpected analysis payload column")
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return await connection.scalar(
                text(f"SELECT {column_name} FROM public.analyses WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
    finally:
        await engine.dispose()


def _reset(config: Config, revision: str) -> None:
    command.downgrade(config, "base")
    command.upgrade(config, revision)


async def _restore_baseline_version_table(url: URL) -> None:
    await _execute_with_trusted_search_path(
        url,
        """
        CREATE TABLE public.alembic_version (
            version_num varchar(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
        """,
    )
    await _execute_with_trusted_search_path(
        url,
        "INSERT INTO public.alembic_version(version_num) VALUES ('0001_foundation')",
    )


@pytest.fixture(scope="module")
def database() -> Iterator[tuple[URL, Config]]:
    url, expected_database = _database_target(os.environ)
    try:
        asyncio.run(_verify_current_database(url, expected_database))
    except DestructiveDatabaseGuardError as error:
        pytest.fail(str(error), pytrace=False)
    config = _alembic_config(url, expected_database)
    command.downgrade(config, "base")
    yield url, config
    command.downgrade(config, "base")


@pytest.mark.integration
def test_fresh_schema_and_empty_roundtrip(database: tuple[URL, Config]) -> None:
    url, config = database
    command.upgrade(config, "head")

    head = asyncio.run(_schema(url))
    assert {"runs", "run_events", "analyses"}.issubset(head["tables"])
    assert head["analyses_columns"] == {
        "run_id",
        "sanitized_result",
        "sanitized_result_sha256",
        "created_at",
    }
    assert head["checks"] == _HEAD_CHECKS
    assert (
        "uq_run_events_run_id_event_hash",
        ("run_id", "current_hash"),
    ) in head["event_uniques"]
    command.check(config)

    command.downgrade(config, "0001_foundation")
    baseline = asyncio.run(_schema(url))
    assert baseline["analyses_columns"] == {"run_id", "result", "result_sha256", "created_at"}
    assert baseline["checks"] == _BASELINE_CHECKS
    assert not any(name == "uq_run_events_run_id_event_hash" for name, _ in baseline["event_uniques"])

    command.upgrade(config, "head")
    assert asyncio.run(_revision(url)) == "0002_contract_convergence"


@pytest.mark.integration
def test_empty_missing_digest_drift_converges(database: tuple[URL, Config]) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    asyncio.run(_execute(url, "ALTER TABLE public.analyses DROP COLUMN result_sha256"))

    command.upgrade(config, "head")

    assert asyncio.run(_schema(url))["analyses_columns"] == {
        "run_id",
        "sanitized_result",
        "sanitized_result_sha256",
        "created_at",
    }
    command.check(config)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("revision", "statement"),
    (
        ("0001_foundation", "ALTER TABLE public.analyses ALTER COLUMN created_at DROP DEFAULT"),
        (
            "head",
            "ALTER TABLE public.analyses ALTER COLUMN created_at DROP DEFAULT",
        ),
        (
            "0001_foundation",
            "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_status_allowed, "
            "ADD CONSTRAINT ck_runs_ck_runs_status_allowed CHECK (status <> '')",
        ),
    ),
)
def test_schema_and_reserved_constraint_drift_fail_before_ddl(
    database: tuple[URL, Config],
    revision: str,
    statement: str,
) -> None:
    url, config = database
    _reset(config, revision)
    before_revision = asyncio.run(_revision(url))
    before_schema = asyncio.run(_schema(url))
    asyncio.run(_execute(url, statement))
    drifted_schema = asyncio.run(_schema(url))

    target = "head" if revision == "0001_foundation" else "0001_foundation"
    with pytest.raises(RuntimeError, match="unsupported managed schema"):
        (command.upgrade if target == "head" else command.downgrade)(config, target)

    assert asyncio.run(_revision(url)) == before_revision
    assert asyncio.run(_schema(url)) == drifted_schema
    assert asyncio.run(_schema(url)) != before_schema
    if "ck_runs_ck_runs_status_allowed" in statement:
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_status_allowed, "
                "ADD CONSTRAINT ck_runs_ck_runs_status_allowed "
                "CHECK (status IN ('pending', 'running', 'completed', 'failed'))",
            )
        )
    else:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses ALTER COLUMN created_at SET DEFAULT now()"))


@pytest.mark.integration
def test_rls_managed_relation_fails_closed_before_emptiness(database: tuple[URL, Config]) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    asyncio.run(_execute(url, "ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY"))
    try:
        with pytest.raises(RuntimeError, match="unsupported managed schema"):
            command.upgrade(config, "head")
        assert asyncio.run(_revision(url)) == "0001_foundation"
    finally:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses DISABLE ROW LEVEL SECURITY"))


@pytest.mark.integration
def test_populated_legacy_upgrade_fails_atomically_without_reading_payload(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    run_id = uuid4()
    payload = {"opaque": "legacy-private-sentinel", "nested": {"value": 7}}
    asyncio.run(_insert_legacy_analysis(url, run_id, payload))

    with pytest.raises(RuntimeError, match="refuses populated analyses") as error:
        command.upgrade(config, "head")

    assert "legacy-private-sentinel" not in str(error.value)
    assert asyncio.run(_revision(url)) == "0001_foundation"
    assert asyncio.run(_schema(url))["analyses_columns"] == {
        "run_id",
        "result",
        "result_sha256",
        "created_at",
    }
    assert asyncio.run(_analysis_payload(url, "result", run_id)) == payload


@pytest.mark.integration
def test_populated_head_downgrade_fails_atomically(database: tuple[URL, Config]) -> None:
    url, config = database
    asyncio.run(_execute(url, "TRUNCATE public.analyses, public.run_events, public.runs"))
    command.upgrade(config, "head")
    run_id = uuid4()
    payload: dict[str, object] = {"sanitized": True}
    asyncio.run(_insert_head_analysis(url, run_id, payload))

    with pytest.raises(RuntimeError, match="refuses populated analyses"):
        command.downgrade(config, "0001_foundation")

    assert asyncio.run(_revision(url)) == "0002_contract_convergence"
    assert asyncio.run(_schema(url))["analyses_columns"] == {
        "run_id",
        "sanitized_result",
        "sanitized_result_sha256",
        "created_at",
    }
    assert asyncio.run(_analysis_payload(url, "sanitized_result", run_id)) == payload


@pytest.mark.integration
def test_populated_runs_events_and_custom_run_event_constraints_survive_roundtrip(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    asyncio.run(_execute(url, "TRUNCATE public.analyses, public.run_events, public.runs"))
    command.downgrade(config, "0001_foundation")

    async def prepare() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                pending_id = uuid4()
                running_id = uuid4()
                await _insert_run(connection, pending_id, status="pending")
                await _insert_run(connection, running_id, status="running")
                await _insert_event(
                    connection,
                    pending_id,
                    sequence=1,
                    current_hash="a" * 64,
                    previous_hash=None,
                )
                await connection.execute(
                    text(
                        "ALTER TABLE public.run_events ADD CONSTRAINT ck_custom_run_events_event_type "
                        "CHECK (event_type <> 'never')"
                    )
                )
                await connection.execute(
                    text(
                        "ALTER TABLE public.run_events ADD CONSTRAINT uq_custom_run_events_hash "
                        "UNIQUE (run_id, current_hash)"
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(prepare())
    command.upgrade(config, "head")
    head = asyncio.run(_schema(url))
    assert head["checks"]["runs"] == _HEAD_CHECKS["runs"]
    assert head["checks"]["run_events"] == _HEAD_CHECKS["run_events"] | {"ck_custom_run_events_event_type"}
    assert head["checks"]["analyses"] == _HEAD_CHECKS["analyses"]
    assert ("uq_custom_run_events_hash", ("run_id", "current_hash")) in head["event_uniques"]
    assert (
        "uq_run_events_run_id_event_hash",
        ("run_id", "current_hash"),
    ) in head["event_uniques"]

    command.downgrade(config, "0001_foundation")
    baseline = asyncio.run(_schema(url))
    assert baseline["checks"]["runs"] == _BASELINE_CHECKS["runs"]
    assert baseline["checks"]["run_events"] == _BASELINE_CHECKS["run_events"] | {"ck_custom_run_events_event_type"}
    assert baseline["checks"]["analyses"] == _BASELINE_CHECKS["analyses"]
    assert ("uq_custom_run_events_hash", ("run_id", "current_hash")) in baseline["event_uniques"]
    assert not any(name == "uq_run_events_run_id_event_hash" for name, _ in baseline["event_uniques"])

    engine = create_async_engine(url)
    try:

        async def statuses() -> tuple[str, ...]:
            async with engine.connect() as connection:
                rows = await connection.execute(text("SELECT status FROM public.runs ORDER BY status"))
                return tuple(rows.scalars())

        assert asyncio.run(statuses()) == ("pending", "running")
    finally:
        asyncio.run(engine.dispose())


@pytest.mark.integration
def test_duplicate_event_hashes_block_unique_creation_transactionally(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    run_id = uuid4()

    async def prepare() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await _insert_run(connection, run_id)
                await _insert_event(
                    connection,
                    run_id,
                    sequence=1,
                    current_hash="b" * 64,
                    previous_hash=None,
                )
                await _insert_event(
                    connection,
                    run_id,
                    sequence=2,
                    current_hash="b" * 64,
                    previous_hash="b" * 64,
                )
        finally:
            await engine.dispose()

    asyncio.run(prepare())
    with pytest.raises(IntegrityError):
        command.upgrade(config, "head")

    assert asyncio.run(_revision(url)) == "0001_foundation"
    assert asyncio.run(_schema(url))["analyses_columns"] == {
        "run_id",
        "result",
        "result_sha256",
        "created_at",
    }
    status = asyncio.run(
        _execute_and_scalar(
            url,
            "SELECT status FROM public.runs WHERE run_id = :run_id",
            {"run_id": run_id},
        )
    )
    assert status == "pending"


async def _execute_and_scalar(url: URL, statement: str, parameters: dict[str, object]) -> object:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return await connection.scalar(text(statement), parameters)
    finally:
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statement", "case_id"),
    (
        (
            "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_status_allowed, "
            "ADD CONSTRAINT ck_runs_ck_runs_status_allowed CHECK "
            "(status = 'pending' OR status = 'running' AND status = 'completed' OR status = 'failed')",
            "boolean_regrouping",
        ),
        (
            "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_status_allowed, "
            "ADD CONSTRAINT ck_runs_ck_runs_status_allowed "
            "CHECK (status IN ('Pending', 'running', 'completed', 'failed'))",
            "status_literal_case",
        ),
        (
            "ALTER TABLE public.run_events DROP CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256, "
            "ADD CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256 "
            "CHECK (current_hash ~ '^[0-9a-f]{64} $')",
            "regex_literal_whitespace",
        ),
        (
            "ALTER TABLE public.run_events DROP CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256, "
            "ADD CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256 "
            "CHECK (current_hash ~ '^[0-9A-F]{64}$')",
            "regex_literal_case",
        ),
        ("ALTER TABLE public.runs ALTER COLUMN status DROP NOT NULL", "nullable_status"),
        ("ALTER TABLE public.run_events ALTER COLUMN sequence TYPE smallint", "sequence_smallint"),
        ("ALTER TABLE public.run_events ALTER COLUMN sequence TYPE bigint", "sequence_bigint"),
        ("ALTER TABLE public.run_events ALTER COLUMN current_hash TYPE text", "current_hash_text"),
        (
            "ALTER TABLE public.analyses ALTER COLUMN result_sha256 SET DEFAULT '" + "a" * 64 + "'",
            "digest_default",
        ),
        (
            'ALTER TABLE public.runs ALTER COLUMN status TYPE varchar(32) COLLATE "C"',
            "nondefault_collation",
        ),
        (
            "ALTER TABLE public.run_events ALTER COLUMN sequence ADD GENERATED ALWAYS AS IDENTITY",
            "identity_column",
        ),
    ),
    ids=lambda value: (
        value
        if isinstance(value, str)
        and value.startswith(
            ("boolean", "status", "regex", "nullable", "sequence", "current", "digest", "nondefault", "identity")
        )
        else None
    ),
)
def test_guard_rejects_check_and_column_contract_drift_without_managed_ddl(
    database: tuple[URL, Config],
    statement: str,
    case_id: str,
) -> None:
    del case_id
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, statement))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statement", "case_id"),
    (
        ("ALTER TABLE public.run_events DROP CONSTRAINT pk_run_events", "missing_run_events_pk"),
        (
            "ALTER TABLE public.run_events DROP CONSTRAINT pk_run_events, "
            "ADD CONSTRAINT pk_run_events PRIMARY KEY (run_id, sequence) DEFERRABLE INITIALLY DEFERRED",
            "deferrable_run_events_pk",
        ),
        ("ALTER TABLE public.run_events DROP CONSTRAINT fk_run_events_run_id_runs", "missing_run_events_fk"),
        (
            "ALTER TABLE public.run_events DROP CONSTRAINT fk_run_events_run_id_runs, "
            "ADD CONSTRAINT fk_run_events_run_id_runs FOREIGN KEY (run_id) "
            "REFERENCES public.runs(run_id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED",
            "deferrable_run_events_fk",
        ),
        (
            "ALTER TABLE public.run_events DROP CONSTRAINT fk_run_events_run_id_runs, "
            "ADD CONSTRAINT fk_run_events_run_id_runs FOREIGN KEY (run_id) "
            "REFERENCES public.runs(run_id) ON DELETE RESTRICT NOT VALID",
            "not_valid_run_events_fk",
        ),
        (
            "ALTER TABLE public.analyses DROP CONSTRAINT pk_analyses, "
            "ADD CONSTRAINT pk_analyses PRIMARY KEY (run_id) DEFERRABLE INITIALLY DEFERRED",
            "deferrable_analyses_pk",
        ),
        (
            "ALTER TABLE public.analyses DROP CONSTRAINT fk_analyses_run_id_runs, "
            "ADD CONSTRAINT fk_analyses_run_id_runs FOREIGN KEY (run_id) "
            "REFERENCES public.runs(run_id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED",
            "deferrable_analyses_fk",
        ),
    ),
    ids=lambda value: (
        value if isinstance(value, str) and value.startswith(("missing", "deferrable", "not_valid")) else None
    ),
)
def test_guard_rejects_key_and_foreign_key_drift_without_managed_ddl(
    database: tuple[URL, Config],
    statement: str,
    case_id: str,
) -> None:
    del case_id
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, statement))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_exact_integer_sequence_is_accepted(database: tuple[URL, Config]) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "ALTER TABLE public.run_events ALTER COLUMN sequence TYPE integer"))
        command.upgrade(config, "head")
        assert asyncio.run(_revision(url)) == "0002_contract_convergence"
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("drop_statement", "create_statement", "case_id"),
    (
        (
            "DROP INDEX public.ix_runs_correlation_id",
            "CREATE INDEX ix_runs_correlation_id ON public.runs (correlation_id) WHERE source = 'rest_file'",
            "partial",
        ),
        (
            "DROP INDEX public.ix_runs_correlation_id",
            "CREATE INDEX ix_runs_correlation_id ON public.runs USING hash (correlation_id)",
            "hash",
        ),
        (
            "DROP INDEX public.ix_runs_status_created_at",
            "CREATE INDEX ix_runs_status_created_at ON public.runs (status, created_at) INCLUDE (correlation_id)",
            "include",
        ),
        (
            "DROP INDEX public.ix_run_events_run_id_event_type",
            "CREATE INDEX ix_run_events_run_id_event_type ON public.run_events (run_id, (lower(event_type)))",
            "expression",
        ),
        (
            "DROP INDEX public.ix_runs_status_created_at",
            "CREATE INDEX ix_runs_status_created_at ON public.runs (status DESC NULLS FIRST, created_at)",
            "noncanonical",
        ),
    ),
    ids=lambda value: value if value in {"partial", "hash", "include", "expression", "noncanonical"} else None,
)
def test_guard_rejects_reserved_index_shape_drift_without_managed_ddl(
    database: tuple[URL, Config],
    drop_statement: str,
    create_statement: str,
    case_id: str,
) -> None:
    del case_id
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, drop_statement))
        asyncio.run(_execute(url, create_statement))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_missing_reserved_index_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "DROP INDEX public.ix_runs_correlation_id"))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(
            _execute(
                url,
                "CREATE INDEX IF NOT EXISTS ix_runs_correlation_id ON public.runs (correlation_id)",
            )
        )
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_reserved_unique_index_alias_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE INDEX uq_run_events_run_id_event_hash ON public.run_events (run_id, current_hash)",
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="uq_run_events_run_id_event_hash",
        )
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_deferrable_reserved_unique_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.run_events ADD CONSTRAINT uq_run_events_run_id_event_hash "
                "UNIQUE (run_id, current_hash) DEFERRABLE INITIALLY DEFERRED",
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="uq_run_events_run_id_event_hash",
        )
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_invalid_reserved_index_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")

    async def prepare_duplicate_correlations() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                correlation_id = uuid4()
                await _insert_run(connection, uuid4(), correlation_id=correlation_id)
                await connection.execute(
                    text(
                        """
                        INSERT INTO public.runs(
                            run_id, request_id, correlation_id, source, status, envelope
                        ) VALUES (
                            :run_id, :request_id, :correlation_id, 'rest_file', 'pending', CAST('{}' AS jsonb)
                        )
                        """
                    ),
                    {"run_id": uuid4(), "request_id": uuid4(), "correlation_id": correlation_id},
                )
        finally:
            await engine.dispose()

    try:
        asyncio.run(prepare_duplicate_correlations())
        asyncio.run(_execute(url, "DROP INDEX public.ix_runs_correlation_id"))
        with pytest.raises(DBAPIError):
            asyncio.run(
                _execute_autocommit(
                    url,
                    "CREATE UNIQUE INDEX CONCURRENTLY ix_runs_correlation_id ON public.runs (correlation_id)",
                )
            )
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_extra_expression_index_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.mtbank_index_expression(value varchar) RETURNS varchar "
                "LANGUAGE sql IMMUTABLE AS $$ SELECT value $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE INDEX mtbank_extra_runs_expression ON public.runs (public.mtbank_index_expression(status))",
            )
        )
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP INDEX IF EXISTS public.mtbank_extra_runs_expression"))
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.mtbank_index_expression(varchar)"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_extra_partial_index_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE INDEX mtbank_extra_runs_partial ON public.runs (status) WHERE status = 'pending'",
            )
        )
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP INDEX IF EXISTS public.mtbank_extra_runs_partial"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_safe_extra_column_index_survives_catalog_validation(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "CREATE INDEX mtbank_safe_runs_status ON public.runs (status)"))
        command.upgrade(config, "head")
        assert asyncio.run(_revision(url)) == "0002_contract_convergence"
        assert (
            asyncio.run(
                _execute_and_scalar(
                    url,
                    "SELECT pg_catalog.to_regclass('public.mtbank_safe_runs_status')",
                    {},
                )
            )
            == "mtbank_safe_runs_status"
        )
    finally:
        asyncio.run(_execute(url, "DROP INDEX IF EXISTS public.mtbank_safe_runs_status"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_unlogged_managed_relation_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses SET UNLOGGED"))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_inherited_managed_relation_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses RENAME TO mtbank_analyses_original"))
        asyncio.run(
            _execute(
                url,
                "CREATE TABLE public.mtbank_analyses_parent "
                "(run_id uuid, result jsonb, result_sha256 varchar(64), created_at timestamptz)",
            )
        )
        asyncio.run(_execute(url, "CREATE TABLE public.analyses () INHERITS (public.mtbank_analyses_parent)"))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.analyses"))
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_analyses_parent"))
        asyncio.run(_execute(url, "ALTER TABLE public.mtbank_analyses_original RENAME TO analyses"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_partitioned_managed_relation_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses RENAME TO mtbank_analyses_original"))
        asyncio.run(_execute(url, "CREATE TABLE public.analyses (run_id uuid) PARTITION BY LIST (run_id)"))
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.analyses"))
        asyncio.run(_execute(url, "ALTER TABLE public.mtbank_analyses_original RENAME TO analyses"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_active_statement_trigger_before_managed_dml(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "CREATE TABLE public.mtbank_migration_side_effects (calls integer NOT NULL)"))
        asyncio.run(_execute(url, "INSERT INTO public.mtbank_migration_side_effects VALUES (0)"))
        asyncio.run(_insert_pending_run(url))
        asyncio.run(
            _execute(
                url,
                """
                CREATE FUNCTION public.mtbank_runs_update_counter() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                    UPDATE public.mtbank_migration_side_effects SET calls = calls + 1;
                    RETURN NULL;
                END;
                $$
                """,
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE TRIGGER mtbank_runs_update_counter AFTER UPDATE ON public.runs "
                "FOR EACH STATEMENT EXECUTE FUNCTION public.mtbank_runs_update_counter()",
            )
        )
        asyncio.run(_execute(url, "UPDATE public.mtbank_migration_side_effects SET calls = 0"))
        _assert_rejected_without_side_effects(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP TRIGGER IF EXISTS mtbank_runs_update_counter ON public.runs"))
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.mtbank_runs_update_counter()"))
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_migration_side_effects"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_active_update_rule_before_managed_dml(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "CREATE TABLE public.mtbank_migration_side_effects (calls integer NOT NULL)"))
        asyncio.run(_execute(url, "INSERT INTO public.mtbank_migration_side_effects VALUES (0)"))
        asyncio.run(_insert_pending_run(url))
        asyncio.run(
            _execute(
                url,
                "CREATE RULE mtbank_runs_update_counter AS ON UPDATE TO public.runs "
                "DO ALSO UPDATE public.mtbank_migration_side_effects SET calls = calls + 1",
            )
        )
        asyncio.run(_execute(url, "UPDATE public.mtbank_migration_side_effects SET calls = 0"))
        _assert_rejected_without_side_effects(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP RULE IF EXISTS mtbank_runs_update_counter ON public.runs"))
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_migration_side_effects"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_expression_dependency_guard_rejects_volatile_custom_check_without_evaluation(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "CREATE TABLE public.mtbank_migration_side_effects (calls integer NOT NULL)"))
        asyncio.run(_execute(url, "INSERT INTO public.mtbank_migration_side_effects VALUES (0)"))
        asyncio.run(_insert_pending_run(url))
        asyncio.run(
            _execute(
                url,
                """
                CREATE FUNCTION public.mtbank_custom_check_counter(value varchar) RETURNS boolean
                LANGUAGE plpgsql VOLATILE AS $$
                BEGIN
                    UPDATE public.mtbank_migration_side_effects SET calls = calls + 1;
                    RETURN true;
                END;
                $$
                """,
            )
        )
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.runs ADD CONSTRAINT ck_custom_runs_side_effect "
                "CHECK (public.mtbank_custom_check_counter(status))",
            )
        )
        asyncio.run(_execute(url, "UPDATE public.mtbank_migration_side_effects SET calls = 0"))
        _assert_rejected_without_side_effects(
            url,
            config,
            lambda: asyncio.run(_require_convergence_expression_dependencies(url)),
        )
    finally:
        asyncio.run(_execute(url, "ALTER TABLE public.runs DROP CONSTRAINT IF EXISTS ck_custom_runs_side_effect"))
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.mtbank_custom_check_counter(varchar)"))
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_migration_side_effects"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_postgresql_equivalent_check_definition_is_accepted(database: tuple[URL, Config]) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_status_allowed, "
                "ADD CONSTRAINT ck_runs_ck_runs_status_allowed "
                "CHECK (((status IN ('pending', 'running', 'completed', 'failed'))))",
            )
        )
        command.upgrade(config, "head")
        assert asyncio.run(_revision(url)) == "0002_contract_convergence"
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_migration_uses_pg_catalog_functions_with_public_shadows(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.jsonb_typeof(value jsonb) RETURNS text LANGUAGE sql IMMUTABLE "
                "AS $$ SELECT 'array'::text $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.now() RETURNS timestamptz LANGUAGE sql STABLE "
                "AS $$ SELECT TIMESTAMPTZ '1970-01-01 00:00:00+00' $$",
            )
        )
        command.upgrade(config, "head")

        async def insert_with_trusted_contracts() -> datetime:
            engine = create_async_engine(url)
            try:
                async with engine.begin() as connection:
                    run_id = uuid4()
                    await _insert_run(connection, run_id, status="queued")
                    await connection.execute(
                        text(
                            """
                            INSERT INTO public.analyses(run_id, sanitized_result, sanitized_result_sha256)
                            VALUES (:run_id, CAST('{}' AS jsonb), :digest)
                            """
                        ),
                        {"run_id": run_id, "digest": "a" * 64},
                    )
                    created_at = await connection.scalar(
                        text("SELECT created_at FROM public.analyses WHERE run_id = :run_id"),
                        {"run_id": run_id},
                    )
                    assert isinstance(created_at, datetime)
                    return created_at
            finally:
                await engine.dispose()

        assert asyncio.run(insert_with_trusted_contracts()) > datetime(2000, 1, 1, tzinfo=UTC)
    finally:
        asyncio.run(_execute(url, "TRUNCATE public.analyses, public.run_events, public.runs"))
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.now()"))
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.jsonb_typeof(jsonb)"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_migration_uses_explicit_public_schema_with_current_user_shadow(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    current_user = str(asyncio.run(_execute_and_scalar(url, "SELECT current_user", {})))
    schema_identifier = _quote_identifier(current_user)
    existing_schema = asyncio.run(
        _execute_and_scalar(
            url,
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = :schema)",
            {"schema": current_user},
        )
    )
    if existing_schema:
        pytest.skip("current-user schema already exists in disposable database")
    try:
        asyncio.run(_execute(url, f"CREATE SCHEMA {schema_identifier}"))
        asyncio.run(_execute(url, f"CREATE TABLE {schema_identifier}.runs (run_id uuid)"))
        asyncio.run(
            _execute(
                url,
                f"CREATE TABLE {schema_identifier}.alembic_version (version_num varchar(32) NOT NULL)",
            )
        )
        command.upgrade(config, "head")
        assert asyncio.run(_revision(url)) == "0002_contract_convergence"
        assert (
            asyncio.run(_execute_and_scalar(url, f"SELECT count(*) FROM {schema_identifier}.alembic_version", {})) == 0
        )
    finally:
        asyncio.run(_execute(url, f"DROP SCHEMA IF EXISTS {schema_identifier} CASCADE"))
        _reset(config, "0001_foundation")


def _convergence_module() -> Any:
    path = ROOT / "src" / "mtbank_ai" / "storage" / "migrations" / "versions" / "0002_contract_convergence.py"
    spec = importlib.util.spec_from_file_location("mtbank_check_probe_regression", path)
    if spec is None or spec.loader is None:
        raise AssertionError("convergence migration module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _require_convergence_expression_dependencies(url: URL) -> None:
    module = _convergence_module()
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET LOCAL search_path TO pg_catalog"))

            def validate(sync_connection: Any) -> None:
                migration_context = MigrationContext.configure(sync_connection)
                with Operations.context(migration_context):
                    module._require_trusted_expression_dependencies()

            await connection.run_sync(validate)
    finally:
        await engine.dispose()


async def _require_ordinary_convergence_relations(url: URL) -> None:
    module = _convergence_module()
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:

            def validate(sync_connection: Any) -> None:
                migration_context = MigrationContext.configure(sync_connection)
                with Operations.context(migration_context):
                    module._set_trusted_search_path()
                    module._lock_managed_tables()
                    module._require_ordinary_unprotected_relations()

            await connection.run_sync(validate)
    finally:
        await engine.dispose()


@pytest.mark.integration
def test_canonicalization_probe_preserves_original_postgresql_error_and_atomicity(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    before_revision = asyncio.run(_revision(url))
    before_catalog = asyncio.run(_managed_catalog_snapshot(url))
    module = _convergence_module()

    async def exercise_probe() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:

                def probe(sync_connection: Any) -> None:
                    migration_context = MigrationContext.configure(sync_connection)
                    with Operations.context(migration_context):
                        with pytest.raises(DBAPIError) as error:
                            module._canonical_check_definition("pg_catalog.no_such_probe_function(source)")
                        error_text = str(error.value)
                        assert "InFailedSqlTransaction" not in error_text
                        assert _REJECTION_SECRET not in error_text
                        assert sync_connection.scalar(text("SELECT 1")) == 1
                        assert (
                            sync_connection.scalar(text("SELECT pg_catalog.to_regclass('pg_temp.mtbank_check_probe')"))
                            is None
                        )

                await connection.run_sync(probe)
        finally:
            await engine.dispose()

    try:
        asyncio.run(exercise_probe())
        assert asyncio.run(_revision(url)) == before_revision
        assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog
    finally:
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_public_operator_dependency_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.mtbank_public_varchar_regex(left_value varchar, right_value varchar) "
                "RETURNS boolean LANGUAGE sql IMMUTABLE AS $$ SELECT true $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE OPERATOR public.~ (LEFTARG = varchar, RIGHTARG = varchar, "
                "PROCEDURE = public.mtbank_public_varchar_regex)",
            )
        )
        asyncio.run(
            _execute_with_search_path(
                url,
                "ALTER TABLE public.run_events "
                "DROP CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256, "
                "ADD CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256 "
                "CHECK (current_hash ~ '^[0-9a-f]{64}$')",
            )
        )
        assert asyncio.run(
            _execute_and_scalar(
                url,
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_constraint AS constraint_data
                    JOIN pg_catalog.pg_depend AS dependency
                        ON dependency.classid = 'pg_catalog.pg_constraint'::pg_catalog.regclass
                        AND dependency.objid = constraint_data.oid
                    JOIN pg_catalog.pg_operator AS catalog_operator
                        ON dependency.refclassid = 'pg_catalog.pg_operator'::pg_catalog.regclass
                        AND catalog_operator.oid = dependency.refobjid
                    JOIN pg_catalog.pg_namespace AS namespace
                        ON namespace.oid = catalog_operator.oprnamespace
                    WHERE constraint_data.conname = 'ck_run_events_ck_run_events_current_hash_sha256'
                      AND namespace.nspname = 'public'
                )
                """,
                {},
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: asyncio.run(_require_convergence_expression_dependencies(url)),
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic migration operator is unsafe",
        )
    finally:
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "ALTER TABLE public.run_events DROP CONSTRAINT ck_run_events_ck_run_events_current_hash_sha256",
            )
        )
        asyncio.run(_execute_with_trusted_search_path(url, "DROP OPERATOR IF EXISTS public.~ (varchar, varchar)"))
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "DROP FUNCTION IF EXISTS public.mtbank_public_varchar_regex(varchar, varchar)",
            )
        )
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_public_default_dependency_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.mtbank_public_now() RETURNS timestamptz LANGUAGE sql STABLE "
                "AS $$ SELECT pg_catalog.now() $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.analyses ALTER COLUMN created_at SET DEFAULT public.mtbank_public_now()",
            )
        )
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        command.downgrade(config, "base")
        asyncio.run(_execute(url, "DROP FUNCTION IF EXISTS public.mtbank_public_now()"))
        command.upgrade(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_public_equality_operator_dependency_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.mtbank_public_varchar_equal(left_value varchar, right_value varchar) "
                "RETURNS boolean LANGUAGE sql IMMUTABLE AS $$ SELECT true $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE OPERATOR public.= (LEFTARG = varchar, RIGHTARG = varchar, "
                "PROCEDURE = public.mtbank_public_varchar_equal)",
            )
        )
        asyncio.run(
            _execute_with_search_path(
                url,
                "ALTER TABLE public.runs "
                "DROP CONSTRAINT ck_runs_ck_runs_source_allowed, "
                "ADD CONSTRAINT ck_runs_ck_runs_source_allowed "
                "CHECK (source IN ('openwebui', 'rest_file', 'rest_url', 'websocket', 'eval'))",
            )
        )
        assert asyncio.run(
            _execute_and_scalar(
                url,
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_constraint AS constraint_data
                    JOIN pg_catalog.pg_depend AS dependency
                        ON dependency.classid = 'pg_catalog.pg_constraint'::pg_catalog.regclass
                        AND dependency.objid = constraint_data.oid
                    JOIN pg_catalog.pg_operator AS catalog_operator
                        ON dependency.refclassid = 'pg_catalog.pg_operator'::pg_catalog.regclass
                        AND catalog_operator.oid = dependency.refobjid
                    JOIN pg_catalog.pg_namespace AS namespace
                        ON namespace.oid = catalog_operator.oprnamespace
                    WHERE constraint_data.conname = 'ck_runs_ck_runs_source_allowed'
                      AND namespace.nspname = 'public'
                )
                """,
                {},
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: asyncio.run(_require_convergence_expression_dependencies(url)),
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic migration operator is unsafe",
        )
    finally:
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "ALTER TABLE public.runs DROP CONSTRAINT ck_runs_ck_runs_source_allowed",
            )
        )
        asyncio.run(_execute_with_trusted_search_path(url, "DROP OPERATOR IF EXISTS public.= (varchar, varchar)"))
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "DROP FUNCTION IF EXISTS public.mtbank_public_varchar_equal(varchar, varchar)",
            )
        )
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_cross_type_public_equality_before_version_table_dml(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "CREATE TABLE public.mtbank_migration_side_effects (calls integer NOT NULL)"))
        asyncio.run(_execute(url, "INSERT INTO public.mtbank_migration_side_effects VALUES (0)"))
        asyncio.run(_execute(url, "CREATE TYPE public.mtbank_operator_enum AS ENUM ('value')"))
        asyncio.run(
            _execute(
                url,
                """
                CREATE FUNCTION public.mtbank_bare_varchar_equal(
                    left_value varchar,
                    right_value public.mtbank_operator_enum
                )
                RETURNS boolean LANGUAGE plpgsql VOLATILE AS $$
                BEGIN
                    UPDATE public.mtbank_migration_side_effects SET calls = calls + 1;
                    RETURN false;
                END;
                $$
                """,
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE OPERATOR public.= (LEFTARG = varchar, RIGHTARG = public.mtbank_operator_enum, "
                "PROCEDURE = public.mtbank_bare_varchar_equal)",
            )
        )
        _assert_rejected_without_side_effects(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic migration operator is unsafe",
        )
    finally:
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "DROP OPERATOR IF EXISTS public.= (varchar, public.mtbank_operator_enum)",
            )
        )
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "DROP FUNCTION IF EXISTS public.mtbank_bare_varchar_equal(varchar, public.mtbank_operator_enum)",
            )
        )
        asyncio.run(_execute_with_trusted_search_path(url, "DROP TYPE IF EXISTS public.mtbank_operator_enum"))
        asyncio.run(_execute_with_trusted_search_path(url, "DROP TABLE IF EXISTS public.mtbank_migration_side_effects"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_spoofed_version_table_view_before_alembic_reads(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "DROP TABLE public.alembic_version"))
        asyncio.run(
            _execute(
                url,
                "CREATE VIEW public.alembic_version AS "
                "SELECT CAST('0002_contract_convergence' AS varchar(32)) AS version_num",
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic version table is unsafe",
        )
    finally:
        asyncio.run(_execute_with_trusted_search_path(url, "DROP VIEW IF EXISTS public.alembic_version"))
        asyncio.run(_restore_baseline_version_table(url))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_wrong_owner_version_table_before_alembic_reads(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    current_user = str(asyncio.run(_execute_and_scalar(url, "SELECT current_user", {})))
    current_user_identifier = _quote_identifier(current_user)
    spoofed_owner = f"mtbank_version_owner_{uuid4().hex}"
    spoofed_owner_identifier = _quote_identifier(spoofed_owner)
    try:
        asyncio.run(_execute(url, f"CREATE ROLE {spoofed_owner_identifier} NOLOGIN"))
        asyncio.run(_execute(url, f"ALTER TABLE public.alembic_version OWNER TO {spoofed_owner_identifier}"))
        before_catalog = asyncio.run(_managed_catalog_snapshot(url))

        with pytest.raises(RuntimeError, match="Alembic version table is unsafe") as error:
            command.upgrade(config, "head")

        assert _REJECTION_SECRET not in str(error.value)
        assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog
    finally:
        asyncio.run(_execute(url, f"ALTER TABLE public.alembic_version OWNER TO {current_user_identifier}"))
        asyncio.run(_execute(url, f"DROP ROLE IF EXISTS {spoofed_owner_identifier}"))
        assert asyncio.run(_revision(url)) == "0001_foundation"
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_wrong_shape_version_table_before_alembic_reads(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.alembic_version ALTER COLUMN version_num TYPE varchar(64)",
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic version table is unsafe",
        )
    finally:
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.alembic_version ALTER COLUMN version_num TYPE varchar(32)",
            )
        )
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_inbound_version_table_foreign_key_before_alembic_reads(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                """
                CREATE TABLE public.mtbank_version_table_reference (
                    version_num varchar(32) NOT NULL,
                    CONSTRAINT mtbank_version_table_reference_fk
                        FOREIGN KEY (version_num) REFERENCES public.alembic_version(version_num)
                )
                """,
            )
        )
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match="Alembic version table is unsafe",
        )
    finally:
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_version_table_reference"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_bootstrap_with_untrusted_public_create_role(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    untrusted_role = _quote_identifier(f"mtbank_public_create_{uuid4().hex}")
    command.downgrade(config, "base")
    asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.alembic_version"))
    before_catalog = asyncio.run(_managed_catalog_snapshot(url))
    try:
        asyncio.run(_execute(url, f"CREATE ROLE {untrusted_role} NOLOGIN NOSUPERUSER"))
        asyncio.run(_execute(url, f"GRANT CREATE ON SCHEMA public TO {untrusted_role}"))

        with pytest.raises(RuntimeError, match="Alembic bootstrap schema is unsafe") as error:
            command.upgrade(config, "0001_foundation")

        assert _REJECTION_SECRET not in str(error.value)
        assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog
        assert asyncio.run(
            _execute_and_scalar(
                url,
                "SELECT pg_catalog.to_regclass('public.alembic_version') IS NULL",
                {},
            )
        )
    finally:
        asyncio.run(_execute(url, f"REVOKE CREATE ON SCHEMA public FROM {untrusted_role}"))
        asyncio.run(_execute(url, f"DROP ROLE IF EXISTS {untrusted_role}"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_custom_mutated_table_check_rejects_before_update_without_payload_leakage(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    payload_marker = "full-row-payload-sentinel"

    async def prepare() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await _insert_run(
                    connection,
                    uuid4(),
                    envelope=json.dumps({"opaque": _REJECTION_SECRET, "payload": payload_marker}),
                )
        finally:
            await engine.dispose()

    try:
        asyncio.run(prepare())
        asyncio.run(
            _execute(
                url,
                "ALTER TABLE public.runs ADD CONSTRAINT ck_custom_runs_no_queued CHECK (status <> 'queued')",
            )
        )
        before_revision = asyncio.run(_revision(url))
        before_catalog = asyncio.run(_managed_catalog_snapshot(url))

        with pytest.raises(RuntimeError, match="unsupported managed schema") as error:
            command.upgrade(config, "head")

        error_text = str(error.value)
        traceback_text = "".join(traceback.format_exception(error.type, error.value, error.tb))
        assert _REJECTION_SECRET not in error_text
        assert payload_marker not in error_text
        assert _REJECTION_SECRET not in traceback_text
        assert payload_marker not in traceback_text
        assert asyncio.run(_revision(url)) == before_revision
        assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog
        assert asyncio.run(_execute_and_scalar(url, "SELECT status FROM public.runs", {})) == "pending"
    finally:
        asyncio.run(_execute(url, "ALTER TABLE public.runs DROP CONSTRAINT IF EXISTS ck_custom_runs_no_queued"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_ordinary_relation_guard_rejects_typed_managed_relation_without_managed_ddl(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, "ALTER TABLE public.analyses RENAME TO mtbank_analyses_original"))
        asyncio.run(_execute(url, "CREATE TYPE public.mtbank_analyses_typed_contract AS (run_id uuid)"))
        asyncio.run(_execute(url, "CREATE TABLE public.analyses OF public.mtbank_analyses_typed_contract"))
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: asyncio.run(_require_ordinary_convergence_relations(url)),
        )
    finally:
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.analyses"))
        asyncio.run(_execute(url, "DROP TYPE IF EXISTS public.mtbank_analyses_typed_contract"))
        asyncio.run(_execute(url, "ALTER TABLE public.mtbank_analyses_original RENAME TO analyses"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statement", "restore_statement", "error_match"),
    (
        (
            "UPDATE public.alembic_version SET version_num = 'mtbank_invalid_revision'",
            "UPDATE public.alembic_version SET version_num = '0001_foundation'",
            "Alembic version table is unsafe",
        ),
        (
            "INSERT INTO public.alembic_version(version_num) VALUES ('0002_contract_convergence')",
            "DELETE FROM public.alembic_version WHERE version_num = '0002_contract_convergence'",
            "Alembic version table is unsafe",
        ),
        (
            "UPDATE public.alembic_version SET version_num = '0002_contract_convergence'",
            "UPDATE public.alembic_version SET version_num = '0001_foundation'",
            "unsupported managed schema",
        ),
    ),
    ids=("unknown_revision", "multiple_revisions", "claimed_head"),
)
def test_env_rejects_noncanonical_version_table_content(
    database: tuple[URL, Config],
    statement: str,
    restore_statement: str,
    error_match: str,
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(_execute(url, statement))
        _assert_rejected_without_managed_ddl(
            url,
            config,
            lambda: command.upgrade(config, "head"),
            error_match=error_match,
        )
    finally:
        asyncio.run(_execute(url, restore_statement))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_guard_rejects_unexpected_inbound_foreign_key_before_managed_dml(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    _reset(config, "0001_foundation")
    try:
        asyncio.run(
            _execute(
                url,
                """
                CREATE TABLE public.mtbank_inbound_runs_reference (
                    run_id uuid NOT NULL,
                    CONSTRAINT mtbank_inbound_runs_reference_fk
                        FOREIGN KEY (run_id) REFERENCES public.runs(run_id) ON DELETE RESTRICT
                )
                """,
            )
        )
        assert asyncio.run(
            _execute_and_scalar(
                url,
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_trigger AS trigger_data
                    JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_data.tgrelid
                    WHERE relation.oid = 'public.runs'::pg_catalog.regclass
                      AND trigger_data.tgisinternal
                      AND trigger_data.tgconstraint = (
                          SELECT foreign_key.oid
                          FROM pg_catalog.pg_constraint AS foreign_key
                          WHERE foreign_key.conname = 'mtbank_inbound_runs_reference_fk'
                      )
                )
                """,
                {},
            )
        )
        _assert_rejected_without_managed_ddl(url, config, lambda: command.upgrade(config, "head"))
    finally:
        asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.mtbank_inbound_runs_reference"))
        _reset(config, "0001_foundation")


@pytest.mark.integration
def test_env_rejects_public_regex_during_foundation_bootstrap(
    database: tuple[URL, Config],
) -> None:
    url, config = database
    command.downgrade(config, "base")
    asyncio.run(_execute(url, "DROP TABLE IF EXISTS public.alembic_version"))
    before_catalog = asyncio.run(_managed_catalog_snapshot(url))
    try:
        asyncio.run(
            _execute(
                url,
                "CREATE FUNCTION public.mtbank_bare_varchar_regex(left_value varchar, right_value varchar) "
                "RETURNS boolean LANGUAGE sql IMMUTABLE AS $$ SELECT true $$",
            )
        )
        asyncio.run(
            _execute(
                url,
                "CREATE OPERATOR public.~ (LEFTARG = varchar, RIGHTARG = varchar, "
                "PROCEDURE = public.mtbank_bare_varchar_regex)",
            )
        )

        with pytest.raises(RuntimeError, match="Alembic migration operator is unsafe") as error:
            command.upgrade(config, "0001_foundation")

        assert _REJECTION_SECRET not in str(error.value)
        assert asyncio.run(_managed_catalog_snapshot(url)) == before_catalog
        assert asyncio.run(
            _execute_and_scalar(
                url,
                "SELECT pg_catalog.to_regclass('public.alembic_version') IS NULL",
                {},
            )
        )
    finally:
        asyncio.run(_execute_with_trusted_search_path(url, "DROP OPERATOR IF EXISTS public.~ (varchar, varchar)"))
        asyncio.run(
            _execute_with_trusted_search_path(
                url,
                "DROP FUNCTION IF EXISTS public.mtbank_bare_varchar_regex(varchar, varchar)",
            )
        )
        _reset(config, "0001_foundation")
