from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from mtbank_ai.domain.agents import ComplianceSeverity
from mtbank_ai.domain.analysis import (
    AnalysisVersions,
    SanitizedAnalysisRecord,
    SanitizedComplianceIssue,
    SanitizedQualityChecklist,
)
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.storage.canonical import canonical_json_sha256
from mtbank_ai.storage.models import analyses, metadata, run_events, runs
from mtbank_ai.storage.repositories import AnalysisRepository, EventRepository, RunRepository

ROOT = Path(__file__).parents[2]


def _versions() -> AnalysisVersions:
    revision = ComponentRevision(
        package="speech-package",
        package_version="1.0.0",
        model_id="speech-model",
        model_revision="revision-1",
    )
    return AnalysisVersions(
        code_sha="abcdef0",
        prompt_bundle_hash="a" * 64,
        taxonomy_version="taxonomy/v1",
        quality_rubric_version="quality/v1",
        compliance_policy_version="compliance/v1",
        asr=revision,
        alignment=revision,
        diarization=revision,
    )


def _record(**changes: object) -> SanitizedAnalysisRecord:
    values: dict[str, object] = {
        "run_id": UUID("11111111-1111-4111-8111-111111111111"),
        "classification_topic_id": "credits",
        "classification_priority_id": "medium",
        "classification_confidence": 0.9,
        "classification_evidence_segment_ids": (UUID("22222222-2222-4222-8222-222222222222"),),
        "quality_total": 75.0,
        "quality_checklist": SanitizedQualityChecklist(
            greeting=True,
            need_detection=True,
            solution_provided=True,
            farewell=False,
        ),
        "quality_evidence_segment_ids": (UUID("22222222-2222-4222-8222-222222222222"),),
        "compliance_passed": True,
        "compliance_issues": (
            SanitizedComplianceIssue(
                rule_id="disclaimer",
                severity=ComplianceSeverity.WARNING,
                evidence_segment_ids=(UUID("22222222-2222-4222-8222-222222222222"),),
            ),
        ),
        "action_item_count": 1,
        "needs_review": False,
        "processing_ms": 125,
        "trusted_versions": _versions(),
    }
    values.update(changes)
    return SanitizedAnalysisRecord.model_validate(values)


def test_storage_metadata_has_only_foundation_tables_and_no_raw_content_columns() -> None:
    assert set(metadata.tables) == {"runs", "run_events", "analyses"}
    forbidden_fragments = ("audio", "transcript", "prompt", "public_response")
    for table in (runs, run_events, analyses):
        for column in table.columns:
            assert not any(fragment in column.name for fragment in forbidden_fragments)
    assert set(analyses.c.keys()) == {"run_id", "sanitized_result", "sanitized_result_sha256", "created_at"}


def test_storage_uses_uuid_jsonb_timezone_hash_checks_indexes_and_restrict_fks() -> None:
    assert isinstance(runs.c.run_id.type, PgUUID)
    assert isinstance(runs.c.envelope.type, JSONB)
    for column_type in (
        runs.c.created_at.type,
        runs.c.updated_at.type,
        run_events.c.occurred_at.type,
        analyses.c.created_at.type,
    ):
        assert isinstance(column_type, DateTime)
        assert column_type.timezone is True
    assert isinstance(run_events.c.redacted_payload.type, JSONB)
    assert isinstance(analyses.c.sanitized_result.type, JSONB)
    assert isinstance(analyses.c.sanitized_result_sha256.type, String)
    assert analyses.c.sanitized_result_sha256.nullable is False

    checks = "\n".join(
        str(constraint.sqltext)
        for table in (runs, run_events, analyses)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    )
    assert "sequence > 0" in checks
    assert "^[0-9a-f]{64}$" in checks
    assert "jsonb_typeof" in checks
    assert "queued" in checks and "processing" in checks

    event_uniques = {
        tuple(column.name for column in constraint.columns)
        for constraint in run_events.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("run_id", "current_hash") in event_uniques

    foreign_keys = [foreign_key for table in (run_events, analyses) for foreign_key in table.foreign_keys]
    assert len(foreign_keys) == 2
    assert all(foreign_key.ondelete == "RESTRICT" for foreign_key in foreign_keys)
    assert {index.name for index in runs.indexes} == {
        "ix_runs_correlation_id",
        "ix_runs_status_created_at",
    }
    assert {index.name for index in run_events.indexes} == {"ix_run_events_run_id_event_type"}


def test_sanitized_record_rejects_sensitive_full_result_markers_and_has_stable_digest() -> None:
    record = _record()
    assert record.model_dump(mode="json")["schema_version"] == "1"
    reversed_payload = dict(reversed(record.model_dump(mode="json").items()))
    assert canonical_json_sha256(record) == canonical_json_sha256(reversed_payload)

    for marker in ("transcript", "summary", "action_items", "prompt", "raw_provider_response", "audio"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            SanitizedAnalysisRecord.model_validate({**record.model_dump(), marker: "must-not-persist"})


def test_repository_ports_expose_only_typed_sanitized_operations() -> None:
    assert hasattr(RunRepository, "create")
    assert hasattr(RunRepository, "get")
    assert hasattr(RunRepository, "set_status")
    assert hasattr(EventRepository, "append")
    assert hasattr(EventRepository, "list")
    assert not hasattr(EventRepository, "delete")
    assert not hasattr(EventRepository, "update")
    assert hasattr(AnalysisRepository, "save_sanitized")
    assert hasattr(AnalysisRepository, "get")
    assert not hasattr(AnalysisRepository, "put")
    assert "SanitizedAnalysisRecord" in str(inspect.signature(AnalysisRepository.save_sanitized))
    assert "AnalyzeResponse" not in str(inspect.signature(AnalysisRepository.save_sanitized))
    assert "hash" not in str(inspect.signature(AnalysisRepository.save_sanitized))


def test_migration_history_preserves_immutable_baseline_and_fail_closed_0002() -> None:
    migrations = ROOT / "src" / "mtbank_ai" / "storage" / "migrations" / "versions"
    foundation_bytes = (migrations / "0001_foundation.py").read_bytes()
    convergence = (migrations / "0002_contract_convergence.py").read_text(encoding="utf-8")
    migration_environment = (migrations.parent / "env.py").read_text(encoding="utf-8")

    assert hashlib.sha256(foundation_bytes).hexdigest() == (
        "eefcad1fc7f6b0eb3baee24229dedf2874c89dd9bf7b6b9eade8c2f81d0864e1"
    )
    assert 'down_revision: str | None = "0001_foundation"' in convergence
    assert "LOCK TABLE public.runs, public.run_events, public.analyses IN ACCESS EXCLUSIVE MODE" in convergence
    assert "SELECT EXISTS (SELECT 1 FROM public.analyses)" in convergence
    assert 'SCHEMA = "public"' in convergence
    assert "relrowsecurity" in convergence and "relforcerowsecurity" in convergence
    assert 'version_table_schema="public"' in migration_environment
    assert "transaction_per_migration=False" in migration_environment
    assert "SET search_path TO public" in migration_environment
    assert "SET search_path TO public, pg_catalog" not in migration_environment
    assert "SET search_path TO pg_catalog, public" not in migration_environment
    assert "compare_server_default=True" in migration_environment
    assert "SELECT pg_catalog.current_database()" in migration_environment
    assert "refuses populated analyses" in convergence
    assert "does not support Alembic --sql" in convergence
    assert "offline Alembic SQL generation is unsupported" in migration_environment
    offline_body = migration_environment.split("def run_migrations_offline() -> None:", maxsplit=1)[1].split(
        "def _run_migrations", maxsplit=1
    )[0]
    assert "context.run_migrations()" not in offline_body
    assert "SELECT run_id" not in convergence
    assert "DELETE FROM analyses" not in convergence
    assert "UPDATE analyses SET" not in convergence
    assert "hashlib" not in convergence
    assert "json.loads" not in convergence
    assert "uq_run_events_run_id_event_hash" in convergence
    assert "pg_catalog.jsonb_typeof" in convergence
    assert "SET DEFAULT pg_catalog.now()" in convergence
    assert "relpersistence" in convergence and "relispartition" in convergence and "reloftype" in convergence
    assert "pg_catalog.pg_inherits" in convergence
    assert "indisvalid" in convergence and "pg_catalog.pg_get_indexdef" in convergence
    assert "pg_catalog.pg_depend" in convergence and "pg_catalog.pg_operator" in convergence
    assert "pg_catalog.pg_trigger" in convergence and "pg_catalog.pg_rewrite" in convergence
    assert "pg_catalog.pg_index" in convergence
    assert "_require_head_contract" in convergence
    assert "_require_inbound_foreign_key_and_internal_trigger_contract" in convergence
    assert "_require_no_active_trigger_or_rule_contract" in convergence
    assert "_require_all_index_contracts" in convergence
    assert "SET LOCAL search_path TO pg_catalog" in convergence
    assert "bind.begin_nested()" in convergence
    assert "DROP TABLE IF EXISTS pg_temp.mtbank_check_probe" not in convergence
    assert "_execute_managed_update" in convergence
    assert "log_error_verbosity" not in convergence
    assert "managed data transition failed" in convergence
    assert "_require_no_mutated_table_custom_checks" in convergence
    assert "_require_safe_version_table_operators(connection)" in migration_environment
    assert "_lock_and_require_safe_version_table(connection)" in migration_environment
    assert "_require_safe_bootstrap_public_schema(connection)" in migration_environment
    assert "acl_data(grantor, grantee, privilege_type, is_grantable)" in migration_environment
    assert "_require_head_contract_for_version_table" in migration_environment
    assert "LOCK TABLE public.alembic_version IN ACCESS EXCLUSIVE MODE" in migration_environment
    assert "_VERSION_TABLE_REVISIONS" in migration_environment
    assert "_require_safe_version_table_contract(connection)" in migration_environment
    assert "alembic_version_pkc" in migration_environment
    assert "Alembic migration operator is unsafe" in migration_environment
    assert "Alembic version table is unsafe" in migration_environment
    preflight_call = migration_environment.index("_require_safe_version_table_operators(connection)")
    version_table_preflight = migration_environment.index("_lock_and_require_safe_version_table(connection)")
    migration_execution = migration_environment.index("context.run_migrations()")
    final_version_table_preflight = migration_environment.index(
        "final_version_table_revisions = _lock_and_require_safe_version_table(connection)"
    )
    head_contract_validation = migration_environment.index(
        "_require_head_contract_for_version_table(connection, final_version_table_revisions)"
    )
    assert preflight_call < migration_execution and version_table_preflight < migration_execution
    assert migration_execution < final_version_table_preflight < head_contract_validation
