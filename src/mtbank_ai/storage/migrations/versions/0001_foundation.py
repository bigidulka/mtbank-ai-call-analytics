"""Создаёт исходную foundation schema.

Revision ID: 0001_foundation
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_foundation"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("envelope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "source IN ('openwebui', 'rest_file', 'rest_url', 'websocket', 'eval')",
            name="ck_runs_source_allowed",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_runs_status_allowed",
        ),
        sa.CheckConstraint("jsonb_typeof(envelope) = 'object'", name="ck_runs_envelope_object"),
        sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
        sa.UniqueConstraint("request_id", name="uq_runs_request_id"),
    )
    op.create_index("ix_runs_correlation_id", "runs", ["correlation_id"], unique=False)
    op.create_index("ix_runs_status_created_at", "runs", ["status", "created_at"], unique=False)

    op.create_table(
        "run_events",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("component", sa.String(length=256), nullable=False),
        sa.Column("redacted_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=True),
        sa.Column("current_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint("sequence > 0", name="ck_run_events_sequence_positive"),
        sa.CheckConstraint(
            "current_hash ~ '^[0-9a-f]{64}$'",
            name="ck_run_events_current_hash_sha256",
        ),
        sa.CheckConstraint(
            "previous_hash IS NULL OR previous_hash ~ '^[0-9a-f]{64}$'",
            name="ck_run_events_previous_hash_sha256",
        ),
        sa.CheckConstraint(
            "(sequence = 1 AND previous_hash IS NULL) OR (sequence > 1 AND previous_hash IS NOT NULL)",
            name="ck_run_events_hash_chain_position",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(redacted_payload) = 'object'",
            name="ck_run_events_redacted_payload_object",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_run_events_run_id_runs", ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("run_id", "sequence", name="pk_run_events"),
    )
    op.create_index(
        "ix_run_events_run_id_event_type",
        "run_events",
        ["run_id", "event_type"],
        unique=False,
    )

    op.create_table(
        "analyses",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("jsonb_typeof(result) = 'object'", name="ck_analyses_result_object"),
        sa.CheckConstraint(
            "result_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_analyses_result_sha256",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name="fk_analyses_run_id_runs", ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("run_id", name="pk_analyses"),
    )


def downgrade() -> None:
    op.drop_table("analyses")
    op.drop_index("ix_run_events_run_id_event_type", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("ix_runs_status_created_at", table_name="runs")
    op.drop_index("ix_runs_correlation_id", table_name="runs")
    op.drop_table("runs")
