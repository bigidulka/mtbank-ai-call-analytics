"""SQLAlchemy Core metadata для PostgreSQL projection."""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)

runs = Table(
    "runs",
    metadata,
    Column("run_id", UUID(as_uuid=True), primary_key=True),
    Column("request_id", UUID(as_uuid=True), nullable=False),
    Column("correlation_id", UUID(as_uuid=True), nullable=False),
    Column("source", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("envelope", JSONB, nullable=False),
    Column("error_code", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("request_id", name="uq_runs_request_id"),
    CheckConstraint(
        "source IN ('openwebui', 'rest_file', 'rest_url', 'websocket', 'eval')",
        name="source_allowed",
    ),
    CheckConstraint(
        "status IN ('queued', 'processing', 'completed', 'failed')",
        name="status_allowed",
    ),
    CheckConstraint("jsonb_typeof(envelope) = 'object'", name="envelope_object"),
)
Index("ix_runs_status_created_at", runs.c.status, runs.c.created_at)
Index("ix_runs_correlation_id", runs.c.correlation_id)

run_events = Table(
    "run_events",
    metadata,
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey("runs.run_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("sequence", Integer, primary_key=True),
    Column("event_type", String(64), nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("component", String(256), nullable=False),
    Column("redacted_payload", JSONB, nullable=False),
    Column("previous_hash", String(64), nullable=True),
    Column("current_hash", String(64), nullable=False),
    UniqueConstraint("run_id", "current_hash", name="uq_run_events_run_id_event_hash"),
    CheckConstraint("sequence > 0", name="sequence_positive"),
    CheckConstraint("current_hash ~ '^[0-9a-f]{64}$'", name="current_hash_sha256"),
    CheckConstraint(
        "previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{64}$'",
        name="previous_hash_sha256",
    ),
    CheckConstraint(
        "(sequence = 1 AND previous_hash IS NULL) OR (sequence > 1 AND previous_hash IS NOT NULL)",
        name="hash_chain_position",
    ),
    CheckConstraint("jsonb_typeof(redacted_payload) = 'object'", name="redacted_payload_object"),
)
Index("ix_run_events_run_id_event_type", run_events.c.run_id, run_events.c.event_type)

analyses = Table(
    "analyses",
    metadata,
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey("runs.run_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    Column("sanitized_result", JSONB, nullable=False),
    Column("sanitized_result_sha256", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("jsonb_typeof(sanitized_result) = 'object'", name="sanitized_result_object"),
    CheckConstraint(
        "sanitized_result_sha256 ~ '^[0-9a-f]{64}$'",
        name="sanitized_result_sha256",
    ),
)
