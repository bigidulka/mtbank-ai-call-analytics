#!/usr/bin/env python3
"""Perform bounded, static-only competitor evidence collection.

This command never downloads, imports, builds, or executes a candidate repository.
Sources must be supplied explicitly as pre-fetched directories named owner__repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.competitor_common import (
        ManifestError,
        load_json_yaml,
        repository_directory_name,
        validate_manifest,
    )
except ModuleNotFoundError:  # Direct invocation: python scripts/analyze_competitors.py
    from competitor_common import ManifestError, load_json_yaml, repository_directory_name, validate_manifest

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "evals" / "competitors" / "manifest.yaml"
DEFAULT_RUBRIC = ROOT / "evals" / "competitors" / "rubric.yaml"
TRUSTED_CANDIDATE_ROOT = ROOT.resolve()
MAX_FILE_BYTES = 1_048_576
MAX_FILES_PER_TREE = 5_000
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".dockerfile",
    ".env",
    ".go",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".pytest_cache",
    ".pyright",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "tmp",
    "vendor",
}

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pipeline_entry", re.compile(r"\b(Pipeline|pipeline|OpenWebUI|pipe\s*\()", re.IGNORECASE)),
    (
        "attachment_flow",
        re.compile(r"\b(attachment|UploadFile|multipart|file_context|uploaded?\s+file)\b", re.IGNORECASE),
    ),
    (
        "asr",
        re.compile(r"\b(faster[-_ ]?whisper|whisper|transcrib\w*|speech[-_ ]?to[-_ ]?text)\b", re.IGNORECASE),
    ),
    ("diarization", re.compile(r"\b(diariz\w*|pyannote)\b", re.IGNORECASE)),
    (
        "role_resolution",
        re.compile(r"\b(speaker\s*(role|label|mapping)|role\s*resolution|operator|customer)\b", re.IGNORECASE),
    ),
    ("llm_call", re.compile(r"\b(openai|chatcompletion|responses\.create|langgraph|agent)\b", re.IGNORECASE)),
    ("tool_trajectory", re.compile(r"\b(tool_calls?|function_call|bind_tools?|tools?\s*=)\b", re.IGNORECASE)),
    ("api_surface", re.compile(r"\b(FastAPI|APIRouter|@app\.(get|post|put|delete)|flask)\b", re.IGNORECASE)),
    ("compose", re.compile(r"\b(services:|docker-compose|compose\.ya?ml|Dockerfile)\b", re.IGNORECASE)),
    ("secrets_egress", re.compile(r"\b(secret|api[_-]?key|token|allowlist|egress|trusted[_-]?host)\b", re.IGNORECASE)),
    ("privacy_redaction", re.compile(r"\b(redact\w*|privacy|PII|personal data|mask\w*)\b", re.IGNORECASE)),
    ("persistence", re.compile(r"\b(sqlalchemy|postgres|sqlite|database|redis|persist\w*)\b", re.IGNORECASE)),
    ("observability", re.compile(r"\b(opentelemetry|prometheus|metrics?|tracing|structured log)\b", re.IGNORECASE)),
    ("test_definition", re.compile(r"\b(pytest|def test_|describe\(|it\()", re.IGNORECASE)),
    (
        "documentation_artifact",
        re.compile(r"\b(architecture|deployment|operation|runbook|installation)\b", re.IGNORECASE),
    ),
    ("evaluation_artifact", re.compile(r"\b(eval(?:uation)?|benchmark|fixture|golden)\b", re.IGNORECASE)),
    ("resilience_controls", re.compile(r"\b(timeout|retry|fail[-_ ]?closed|circuit breaker)\b", re.IGNORECASE)),
)

CRITERION_RULES: dict[str, tuple[str, ...]] = {
    "pipeline_entry": ("pipeline_entry",),
    "attachment_flow": ("attachment_flow",),
    "asr": ("asr",),
    "diarization_roles": ("diarization", "role_resolution"),
    "llm_tool_trajectories": ("llm_call", "tool_trajectory"),
    "api_compose": ("api_surface", "compose"),
    "security_privacy": ("secrets_egress", "privacy_redaction"),
    "persistence": ("persistence",),
    "observability": ("observability",),
    "tests": ("test_definition",),
    "documentation": (),
}
BONUS_RULES = {
    "evaluation_artifacts": ("evaluation_artifact",),
    "resilience_controls": ("resilience_controls",),
    "release_evidence": (),
}


def excerpt_hash(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def _is_claim_only_path(relative: Path) -> bool:
    return relative.suffix.casefold() in {".md", ".rst", ".txt"} or "docs" in relative.parts


def _safe_relative(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _file_kind(path: Path) -> str | None:
    if path.name in {"Dockerfile", "Makefile"} or path.suffix.casefold() in TEXT_SUFFIXES:
        return "text"
    return None


def _skipped(repo: str, sha: str | None, path: str, status: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "sha": sha,
        "path": path,
        "line": None,
        "rule_id": "scan_guard",
        "excerpt_hash": None,
        "status": status,
    }


def scan_tree(repo: str, sha: str | None, root: Path) -> dict[str, Any]:
    """Scan only bounded regular text files under root; never follow symlinks."""
    evidence: list[dict[str, Any]] = []
    files_seen = 0
    source_evidence_found = False
    if root.is_symlink():
        return {
            "scan_status": "source_symlink_rejected",
            "evidence": [_skipped(repo, sha, str(root), "skipped_symlink")],
            "files_seen": files_seen,
        }
    root = root.resolve()

    if not root.is_dir():
        return {"scan_status": "source_not_provided", "evidence": evidence, "files_seen": files_seen}

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        if current != root and (current / ".git").is_file():
            relative = current.relative_to(root)
            evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_submodule"))
            directory_names[:] = []
            continue
        kept_directories: list[str] = []
        for name in directory_names:
            child = current / name
            relative = child.relative_to(root)
            if child.is_symlink():
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_symlink"))
            elif name in IGNORED_DIRECTORIES:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_untrusted_metadata"))
            elif _safe_relative(child, root) is None:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_path_outside_root"))
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            path = current / name
            relative = path.relative_to(root)
            if path.is_symlink():
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_symlink"))
                continue
            if _safe_relative(path, root) is None:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_path_outside_root"))
                continue
            if _file_kind(path) is None:
                continue
            files_seen += 1
            if files_seen > MAX_FILES_PER_TREE:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_file_limit"))
                return {"scan_status": "completed_with_file_limit", "evidence": evidence, "files_seen": files_seen}
            try:
                size = path.stat().st_size
            except OSError:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_unreadable"))
                continue
            if size > MAX_FILE_BYTES:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_oversized"))
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_unreadable"))
                continue
            if b"\x00" in raw:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_binary"))
                continue
            if raw.startswith(b"version https://git-lfs.github.com/spec/"):
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_lfs_pointer"))
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                evidence.append(_skipped(repo, sha, relative.as_posix(), "skipped_non_utf8"))
                continue

            claim_only = _is_claim_only_path(relative)
            for line_number, line in enumerate(text.splitlines(), start=1):
                for rule_id, pattern in RULES:
                    if pattern.search(line) is None:
                        continue
                    status = "claim_only" if claim_only else "verified"
                    evidence.append(
                        {
                            "repo": repo,
                            "sha": sha,
                            "path": relative.as_posix(),
                            "line": line_number,
                            "rule_id": rule_id,
                            "excerpt_hash": excerpt_hash(line),
                            "status": status,
                        }
                    )
                    source_evidence_found = source_evidence_found or status == "verified"
    scan_status = "completed" if source_evidence_found else "completed_without_verified_signals"
    return {"scan_status": scan_status, "evidence": evidence, "files_seen": files_seen}


def _rule_statuses(evidence: list[dict[str, Any]], rule_ids: tuple[str, ...]) -> set[str]:
    return {item["rule_id"] for item in evidence if item["status"] == "verified" and item["rule_id"] in rule_ids}


def score_evidence(
    scan: dict[str, Any], rubric: dict[str, Any], *, candidate_release_ready: bool = False
) -> dict[str, Any]:
    """Calculate only observed verified points; never turn unknown into a zero."""
    evidence = scan["evidence"]
    completed = scan["scan_status"].startswith("completed")
    criteria: list[dict[str, Any]] = []
    bonuses: list[dict[str, Any]] = []
    verified_points = 0
    verified_bonus = 0

    for criterion in rubric["criteria"]:
        rule_ids = CRITERION_RULES[criterion["id"]]
        found = _rule_statuses(evidence, rule_ids) if completed else set()
        if not completed:
            status, points = "unknown", None
        elif not found:
            status, points = "unknown", None
        elif len(rule_ids) == 1:
            status, points = "verified", criterion["weight"]
        elif len(found) == len(rule_ids):
            status, points = "verified", criterion["weight"]
        else:
            status, points = "partial_verified", criterion["weight"] // len(rule_ids)
        if isinstance(points, int):
            verified_points += points
        criteria.append({"id": criterion["id"], "status": status, "verified_points": points})

    for bonus in rubric["bonuses"]:
        rule_ids = BONUS_RULES[bonus["id"]]
        found = _rule_statuses(evidence, rule_ids) if completed else set()
        if bonus["id"] == "release_evidence":
            status, points = ("verified", bonus["weight"]) if candidate_release_ready else ("unknown", None)
        elif found:
            status, points = "verified", bonus["weight"]
        else:
            status, points = "unknown", None
        if isinstance(points, int):
            verified_bonus += points
        bonuses.append({"id": bonus["id"], "status": status, "verified_points": points})

    complete = all(item["status"] == "verified" for item in criteria) and candidate_release_ready
    return {
        "score_status": "verified_complete" if complete else "unknown",
        "comparative_score": verified_points + verified_bonus if complete else None,
        "verified_points_observed": verified_points,
        "verified_bonus_observed": verified_bonus,
        "criteria": criteria,
        "bonuses": bonuses,
    }


def _trusted_git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }


def _trusted_git_command(*arguments: str) -> list[str]:
    return [
        "git",
        "-c",
        "core.fsmonitor=",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        f"safe.directory={TRUSTED_CANDIDATE_ROOT}",
        "-C",
        str(TRUSTED_CANDIDATE_ROOT),
        *arguments,
    ]


def candidate_identity(root: Path) -> dict[str, Any]:
    """Inspect Git only for the fixed local project, never an arbitrary candidate root."""
    if root.resolve() != TRUSTED_CANDIDATE_ROOT:
        return {"immutable_sha": None, "commit_status": "untrusted_candidate_requires_explicit_identity"}
    try:
        dirty = subprocess.run(
            _trusted_git_command("status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=_trusted_git_environment(),
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"immutable_sha": None, "commit_status": "unfrozen_no_git"}
    if dirty.strip():
        return {"immutable_sha": None, "commit_status": "uncommitted"}
    try:
        sha = subprocess.run(
            _trusted_git_command("rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=_trusted_git_environment(),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"immutable_sha": None, "commit_status": "unfrozen_no_git"}
    return {"immutable_sha": sha, "commit_status": "clean_git_sha"}


def _candidate_report(root: Path, rubric: dict[str, Any]) -> dict[str, Any]:
    identity = candidate_identity(root)
    scan = scan_tree("local-candidate", identity["immutable_sha"], root)
    scoring = score_evidence(scan, rubric, candidate_release_ready=False)
    return {
        "root": str(root.resolve()),
        **identity,
        "release_image_evidence": "missing",
        "comparative_score_blocked": "An immutable release SHA and image evidence are required.",
        "scan": scan,
        "scoring": scoring,
    }


def analyze(
    manifest: dict[str, Any],
    rubric: dict[str, Any],
    sources_directory: Path | None = None,
    candidate_root: Path | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest)
    reports: list[dict[str, Any]] = []
    for record in manifest["records"]:
        source = None if sources_directory is None else sources_directory / repository_directory_name(record["repo"])
        scan = (
            {"scan_status": "source_not_provided", "evidence": [], "files_seen": 0}
            if source is None
            else scan_tree(record["repo"], record["sha"], source)
        )
        reports.append(
            {
                "repo": record["repo"],
                "sha": record["sha"],
                "scan": scan,
                "scoring": score_evidence(scan, rubric),
            }
        )
    return {
        "schema": "competitor-analysis/v1",
        "execution_policy": "static-only; no download, import, build, test, Docker, or binary execution",
        "cohort_reconstruction": manifest["reconstruction"],
        "competitors": reports,
        "candidate": _candidate_report(candidate_root, rubric) if candidate_root else None,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--sources-dir", type=Path, help="Explicit pre-fetched owner__repository source directories")
    parser.add_argument("--candidate-root", type=Path, help="Explicit local candidate root to scan statically")
    parser.add_argument("--output", type=Path, help="Write JSON report here instead of stdout")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        manifest = load_json_yaml(arguments.manifest)
        rubric = load_json_yaml(arguments.rubric)
        report = analyze(manifest, rubric, arguments.sources_dir, arguments.candidate_root)
    except ManifestError as error:
        print(f"manifest error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
