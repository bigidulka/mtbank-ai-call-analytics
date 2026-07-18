"""Async Alembic environment для PostgreSQL projection."""

from __future__ import annotations

import asyncio
import importlib.util
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from mtbank_ai.config import load_database_settings
from mtbank_ai.storage.models import metadata
from mtbank_ai.storage.postgres import build_postgres_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata
_VERSION_TABLE_REVISIONS = frozenset({"0001_foundation", "0002_contract_convergence"})


def _online_database_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    database = load_database_settings()
    return build_postgres_url(database).render_as_string(hide_password=False)


def run_migrations_offline() -> None:
    raise RuntimeError(
        "offline Alembic SQL generation is unsupported because 0002_contract_convergence "
        "requires live PostgreSQL catalog validation"
    )


def _include_object(object_: object, name: str | None, type_: str, reflected: bool, compare_to: object) -> bool:
    del object_, reflected, compare_to
    return not (type_ == "table" and name == "alembic_version")


def _require_safe_version_table_operators(connection: Connection) -> None:
    """Reject public operators that can affect historical or Alembic migration SQL."""
    connection.execute(text("SET LOCAL search_path TO pg_catalog"))
    has_unsafe_operator = connection.scalar(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_operator AS operator_data
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = operator_data.oprnamespace
                WHERE namespace.nspname = 'public'
                  AND operator_data.oprkind = 'b'::"char"
                  AND operator_data.oprname IN ('=', '~')
            )
            """
        )
    )
    if has_unsafe_operator:
        raise RuntimeError("Alembic migration operator is unsafe")


def _require_safe_version_table_contract(connection: Connection) -> None:
    """Validate an existing public.alembic_version before Alembic can read it."""
    with connection.begin_nested():
        connection.execute(text("SET LOCAL search_path TO pg_catalog"))
        has_safe_version_table = connection.scalar(
            text(
                """
                WITH version_table AS (
                    SELECT relation.*
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = 'public'
                      AND relation.relname = 'alembic_version'
                )
                SELECT NOT EXISTS (SELECT 1 FROM version_table) OR EXISTS (
                    SELECT 1
                    FROM version_table AS relation
                    JOIN pg_catalog.pg_attribute AS attribute
                        ON attribute.attrelid = relation.oid AND attribute.attnum = 1
                    JOIN pg_catalog.pg_type AS type_data ON type_data.oid = attribute.atttypid
                    JOIN pg_catalog.pg_collation AS collation_data ON collation_data.oid = attribute.attcollation
                    JOIN pg_catalog.pg_namespace AS collation_namespace
                        ON collation_namespace.oid = collation_data.collnamespace
                    JOIN pg_catalog.pg_constraint AS primary_key
                        ON primary_key.conrelid = relation.oid
                        AND primary_key.conname = 'alembic_version_pkc'
                    JOIN pg_catalog.pg_index AS index_data ON index_data.indexrelid = primary_key.conindid
                    JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_data.indexrelid
                    JOIN pg_catalog.pg_am AS access_method ON access_method.oid = index_relation.relam
                    WHERE relation.relkind = 'r'::"char"
                      AND relation.relpersistence = 'p'::"char"
                      AND NOT relation.relispartition
                      AND relation.reloftype = 0
                      AND NOT relation.relrowsecurity
                      AND NOT relation.relforcerowsecurity
                      AND relation.relowner = (
                          SELECT role.oid
                          FROM pg_catalog.pg_roles AS role
                          WHERE role.rolname = CURRENT_USER
                      )
                      AND relation.relacl IS NULL
                      AND relation.reloptions IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_inherits AS inheritance
                          WHERE inheritance.inhrelid = relation.oid OR inheritance.inhparent = relation.oid
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_trigger AS trigger_data
                          WHERE trigger_data.tgrelid = relation.oid
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_rewrite AS rule_data
                          WHERE rule_data.ev_class = relation.oid
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_constraint AS foreign_key
                          WHERE foreign_key.contype = 'f'::"char"
                            AND (foreign_key.conrelid = relation.oid OR foreign_key.confrelid = relation.oid)
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_depend AS dependency
                          JOIN pg_catalog.pg_class AS dependent_relation
                              ON dependency.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
                              AND dependent_relation.oid = dependency.objid
                          WHERE dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
                            AND dependency.refobjid = relation.oid
                            AND dependent_relation.oid <> relation.oid
                            AND dependent_relation.oid <> primary_key.conindid
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_depend AS dependency
                          JOIN pg_catalog.pg_rewrite AS dependent_rule
                              ON dependency.classid = 'pg_catalog.pg_rewrite'::pg_catalog.regclass
                              AND dependent_rule.oid = dependency.objid
                          WHERE dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
                            AND dependency.refobjid = relation.oid
                            AND dependent_rule.ev_class <> relation.oid
                      )
                      AND attribute.attname = 'version_num'
                      AND attribute.atttypid = 'pg_catalog.varchar'::pg_catalog.regtype
                      AND attribute.atttypmod = 36
                      AND attribute.attnotnull
                      AND attribute.attidentity NOT IN ('a'::"char", 'd'::"char")
                      AND attribute.attgenerated <> 's'::"char"
                      AND attribute.attcollation = type_data.typcollation
                      AND collation_namespace.nspname = 'pg_catalog'
                      AND attribute.attacl IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_attribute AS other_attribute
                          WHERE other_attribute.attrelid = relation.oid
                            AND other_attribute.attnum > 0
                            AND (
                                other_attribute.attisdropped
                                OR other_attribute.attname <> 'version_num'
                                OR other_attribute.attnum <> 1
                            )
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.pg_attrdef AS default_value
                          WHERE default_value.adrelid = relation.oid
                      )
                      AND (
                          SELECT pg_catalog.count(*)
                          FROM pg_catalog.pg_constraint AS other_constraint
                          WHERE other_constraint.conrelid = relation.oid
                      ) = 1
                      AND primary_key.contype = 'p'::"char"
                      AND NOT primary_key.condeferrable
                      AND NOT primary_key.condeferred
                      AND primary_key.convalidated
                      AND primary_key.connoinherit
                      AND primary_key.coninhcount = 0
                      AND primary_key.conparentid = 0
                      AND pg_catalog.array_length(primary_key.conkey, 1) = 1
                      AND primary_key.conkey[1] = attribute.attnum
                      AND index_data.indrelid = relation.oid
                      AND index_relation.relname = 'alembic_version_pkc'
                      AND index_relation.relkind = 'i'::"char"
                      AND index_relation.relpersistence = 'p'::"char"
                      AND index_relation.relowner = relation.relowner
                      AND index_relation.relacl IS NULL
                      AND index_relation.reloptions IS NULL
                      AND index_data.indisvalid
                      AND index_data.indisready
                      AND index_data.indislive
                      AND index_data.indisunique
                      AND index_data.indisprimary
                      AND NOT index_data.indisexclusion
                      AND index_data.indimmediate
                      AND NOT index_data.indcheckxmin
                      AND NOT index_data.indnullsnotdistinct
                      AND NOT index_data.indisreplident
                      AND NOT index_data.indisclustered
                      AND index_data.indpred IS NULL
                      AND index_data.indexprs IS NULL
                      AND index_data.indnkeyatts = 1
                      AND index_data.indnatts = 1
                      AND access_method.amname = 'btree'
                      AND (
                          SELECT pg_catalog.count(*)
                          FROM pg_catalog.pg_index AS other_index
                          WHERE other_index.indrelid = relation.oid
                      ) = 1
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.unnest(index_data.indkey) WITH ORDINALITY
                              AS key_attribute(attnum, position)
                          WHERE key_attribute.position <> 1 OR key_attribute.attnum <> attribute.attnum
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.unnest(index_data.indoption) AS key_option(option)
                          WHERE key_option.option <> 0
                      )
                      AND EXISTS (
                          SELECT 1
                          FROM pg_catalog.unnest(index_data.indclass) WITH ORDINALITY
                              AS index_opclass(opclass_oid, position)
                          JOIN pg_catalog.pg_opclass AS catalog_opclass
                              ON catalog_opclass.oid = index_opclass.opclass_oid
                          JOIN pg_catalog.pg_namespace AS opclass_namespace
                              ON opclass_namespace.oid = catalog_opclass.opcnamespace
                          WHERE index_opclass.position = 1
                            AND opclass_namespace.nspname = 'pg_catalog'
                            AND catalog_opclass.opcname = 'text_ops'
                            AND catalog_opclass.opcintype = 'pg_catalog.text'::pg_catalog.regtype
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM pg_catalog.unnest(index_data.indcollation) WITH ORDINALITY
                              AS index_collation(collation_oid, position)
                          WHERE index_collation.position <> 1
                             OR index_collation.collation_oid <> attribute.attcollation
                      )
                )
                """
            )
        )
        if has_safe_version_table is not True:
            raise RuntimeError("Alembic version table is unsafe")


def _version_table_exists(connection: Connection) -> bool:
    return bool(
        connection.scalar(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = 'public'
                      AND relation.relname = 'alembic_version'
                )
                """
            )
        )
    )


def _require_safe_bootstrap_public_schema(connection: Connection) -> None:
    connection.execute(text("SET LOCAL search_path TO pg_catalog"))
    has_untrusted_create_privilege = connection.scalar(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_namespace AS namespace
                JOIN pg_catalog.pg_roles AS schema_owner ON schema_owner.oid = namespace.nspowner
                LEFT JOIN LATERAL pg_catalog.aclexplode(
                    COALESCE(namespace.nspacl, pg_catalog.acldefault('n'::"char", namespace.nspowner))
                ) AS acl_data(grantor, grantee, privilege_type, is_grantable) ON true
                LEFT JOIN pg_catalog.pg_roles AS role_data ON role_data.oid = acl_data.grantee
                WHERE namespace.nspname = 'public'
                  AND (
                      (
                          schema_owner.rolname <> CURRENT_USER
                          AND schema_owner.rolname <> 'pg_database_owner'
                          AND NOT schema_owner.rolsuper
                      )
                      OR (
                          acl_data.privilege_type = 'CREATE'
                          AND (
                              acl_data.grantee = 0
                              OR role_data.oid IS NULL
                              OR (
                                  role_data.rolname <> CURRENT_USER
                                  AND role_data.rolname <> 'pg_database_owner'
                                  AND NOT role_data.rolsuper
                              )
                          )
                      )
                  )
            )
            """
        )
    )
    if has_untrusted_create_privilege:
        raise RuntimeError("Alembic bootstrap schema is unsafe")


def _lock_and_require_safe_version_table(connection: Connection) -> tuple[object, ...]:
    _require_safe_version_table_contract(connection)
    if not _version_table_exists(connection):
        return ()
    try:
        connection.execute(text("LOCK TABLE public.alembic_version IN ACCESS EXCLUSIVE MODE"))
    except DBAPIError:
        raise RuntimeError("Alembic version table is unsafe") from None
    _require_safe_version_table_contract(connection)
    revisions = tuple(connection.execute(text("SELECT version_num FROM public.alembic_version")).scalars())
    if len(revisions) > 1 or any(revision not in _VERSION_TABLE_REVISIONS for revision in revisions):
        raise RuntimeError("Alembic version table is unsafe")
    return revisions


def _require_head_contract_for_version_table(connection: Connection, revisions: tuple[object, ...]) -> None:
    if revisions != ("0002_contract_convergence",):
        return
    path = Path(__file__).parent / "versions" / "0002_contract_convergence.py"
    spec = importlib.util.spec_from_file_location("mtbank_version_table_head_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Alembic version table is unsafe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require_head_contract = getattr(module, "_require_head_contract", None)
    if not callable(require_head_contract):
        raise RuntimeError("Alembic version table is unsafe")
    migration_context = MigrationContext.configure(connection)
    with Operations.context(migration_context):
        require_head_contract()


def _run_migrations(connection: Connection) -> None:
    connection.execute(text("SET search_path TO public"))
    connection.commit()
    expected_database = config.attributes.get("mtbank_expected_database")
    if expected_database is not None:
        actual_database = connection.scalar(text("SELECT pg_catalog.current_database()"))
        if actual_database != expected_database:
            raise RuntimeError("Alembic connection database does not match the guarded test database")
        connection.commit()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        include_object=_include_object,
        version_table_schema="public",
        transaction_per_migration=False,
    )
    with context.begin_transaction():
        _require_safe_version_table_operators(connection)
        if _version_table_exists(connection):
            _lock_and_require_safe_version_table(connection)
        else:
            _require_safe_bootstrap_public_schema(connection)
        connection.execute(text("SET LOCAL search_path TO public"))
        context.run_migrations()
        final_version_table_revisions = _lock_and_require_safe_version_table(connection)
        _require_head_contract_for_version_table(connection, final_version_table_revisions)


async def run_migrations_online() -> None:
    connectable = create_async_engine(_online_database_url(), poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
