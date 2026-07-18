from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import scripts.analyze_competitors as competitor_analysis
from scripts.analyze_competitors import (
    MAX_FILE_BYTES,
    analyze,
    candidate_identity,
    excerpt_hash,
    scan_tree,
    score_evidence,
)
from scripts.competitor_common import ManifestError, load_json_yaml, validate_manifest

ROOT = Path(__file__).parents[2]
MANIFEST_PATH = ROOT / "evals" / "competitors" / "manifest.yaml"
RUBRIC_PATH = ROOT / "evals" / "competitors" / "rubric.yaml"


def test_manifest_is_exact_historical_cohort() -> None:
    manifest = load_json_yaml(MANIFEST_PATH)

    validate_manifest(manifest)

    assert len(manifest["records"]) == 44
    assert manifest["checked_at_utc"] is None
    assert manifest["excluded_after_original_request"]["repo"] == "vbuyel/mtbank-ai-hiring"
    assert "vbuyel/mtbank-ai-hiring" not in {record["repo"] for record in manifest["records"]}
    assert next(record for record in manifest["records"] if record["repo"] == "AlexeyShakal/MTBank")["sha"] is None


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "invalid_sha", "empty_missing_reason", "order", "replacement", "vbuyel", "unexpected_null"],
)
def test_manifest_validator_rejects_invalid_records(mutation: str) -> None:
    manifest = copy.deepcopy(load_json_yaml(MANIFEST_PATH))
    if mutation == "duplicate":
        manifest["records"][1]["repo"] = manifest["records"][0]["repo"]
    elif mutation == "invalid_sha":
        manifest["records"][0]["sha"] = "not-a-commit"
    elif mutation == "empty_missing_reason":
        empty = next(record for record in manifest["records"] if record["repo"] == "AlexeyShakal/MTBank")
        empty.pop("unavailable_reason")
    elif mutation == "order":
        manifest["records"][0], manifest["records"][1] = manifest["records"][1], manifest["records"][0]
    elif mutation == "replacement":
        manifest["records"][0]["repo"] = "replacement/not-in-cohort"
        manifest["records"][0]["url"] = "https://github.com/replacement/not-in-cohort"
    elif mutation == "vbuyel":
        manifest["excluded_after_original_request"]["reason"] = "different reason"
    else:
        manifest["records"][0]["sha"] = None
        manifest["records"][0]["status"] = "empty_repository_no_immutable_commit"
        manifest["records"][0]["unavailable_reason"] = "not allowed"

    with pytest.raises(ManifestError):
        validate_manifest(manifest)


def test_scanner_collects_hashed_static_evidence_without_execution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "pipeline.py"
    source_line = "class Pipeline: pass"
    source_file.write_text(f"{source_line}\nraise RuntimeError('must not execute')\n", encoding="utf-8")

    scan = scan_tree("example/repository", "a" * 40, source)

    item = next(evidence for evidence in scan["evidence"] if evidence["rule_id"] == "pipeline_entry")
    assert item == {
        "repo": "example/repository",
        "sha": "a" * 40,
        "path": "pipeline.py",
        "line": 1,
        "rule_id": "pipeline_entry",
        "excerpt_hash": hashlib.sha256(source_line.encode("utf-8")).hexdigest(),
        "status": "verified",
    }
    assert excerpt_hash(f"  {source_line}  ") == item["excerpt_hash"]


def test_scanner_rejects_symlink_submodule_binary_lfs_and_large_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("class Pipeline: pass\n", encoding="utf-8")
    (source / "linked.py").symlink_to(outside)
    (source / "binary.py").write_bytes(b"\x00class Pipeline")
    (source / "large.py").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    (source / "pointer.py").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    submodule = source / "dependency"
    submodule.mkdir()
    (submodule / ".git").write_text("gitdir: ../.git/modules/dependency\n", encoding="utf-8")
    (submodule / "nested.py").write_text("class Pipeline: pass\n", encoding="utf-8")

    scan = scan_tree("example/repository", "b" * 40, source)
    statuses = {item["status"] for item in scan["evidence"]}

    assert {
        "skipped_symlink",
        "skipped_binary",
        "skipped_oversized",
        "skipped_lfs_pointer",
        "skipped_submodule",
    } <= statuses
    assert not any(item["path"] == "linked.py" and item["rule_id"] == "pipeline_entry" for item in scan["evidence"])
    assert not any(item["path"] == "dependency/nested.py" for item in scan["evidence"])


def test_readme_claims_do_not_create_verified_points() -> None:
    rubric = load_json_yaml(RUBRIC_PATH)
    claim_only_scan = {
        "scan_status": "completed_without_verified_signals",
        "files_seen": 1,
        "evidence": [
            {
                "repo": "example/repository",
                "sha": "c" * 40,
                "path": "README.md",
                "line": 1,
                "rule_id": "pipeline_entry",
                "excerpt_hash": "x",
                "status": "claim_only",
            }
        ],
    }

    score = score_evidence(claim_only_scan, rubric)

    assert score["comparative_score"] is None
    assert score["verified_points_observed"] == 0
    assert next(item for item in score["criteria"] if item["id"] == "pipeline_entry")["status"] == "unknown"


def test_prose_and_docs_spoofs_do_not_create_verified_points(tmp_path: Path) -> None:
    rubric = load_json_yaml(RUBRIC_PATH)
    source = tmp_path / "source"
    source.mkdir()
    (source / "architecture.md").write_text(
        "class Pipeline: pass; pytest; secret; timeout; benchmark\n", encoding="utf-8"
    )
    docs = source / "docs"
    docs.mkdir()
    (docs / "implementation.py").write_text("class Pipeline: pass\n", encoding="utf-8")
    (source / "guide.rst").write_text("class Pipeline: pass\n", encoding="utf-8")
    (source / "notes.txt").write_text("class Pipeline: pass\n", encoding="utf-8")

    scan = scan_tree("example/repository", "e" * 40, source)
    score = score_evidence(scan, rubric)

    assert scan["scan_status"] == "completed_without_verified_signals"
    assert all(item["status"] == "claim_only" for item in scan["evidence"])
    assert score["verified_points_observed"] == 0
    assert score["verified_bonus_observed"] == 0


def test_verified_static_evidence_is_observed_but_not_a_comparative_score() -> None:
    rubric = load_json_yaml(RUBRIC_PATH)
    verified_scan = {
        "scan_status": "completed",
        "files_seen": 1,
        "evidence": [
            {
                "repo": "example/repository",
                "sha": "d" * 40,
                "path": "pipeline.py",
                "line": 1,
                "rule_id": "pipeline_entry",
                "excerpt_hash": "x",
                "status": "verified",
            }
        ],
    }

    score = score_evidence(verified_scan, rubric)

    assert score["score_status"] == "unknown"
    assert score["comparative_score"] is None
    assert score["verified_points_observed"] == 8


def test_trusted_candidate_dirty_tree_has_no_immutable_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    (tmp_path / "uncommitted.py").write_text("class Pipeline: pass\n", encoding="utf-8")
    monkeypatch.setattr(competitor_analysis, "TRUSTED_CANDIDATE_ROOT", tmp_path.resolve())

    identity = candidate_identity(tmp_path)

    assert identity == {"immutable_sha": None, "commit_status": "uncommitted"}


def test_untrusted_candidate_never_invokes_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git_directory = tmp_path / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text("[alias]\nstatus = !false\n", encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("untrusted candidate must not invoke git")

    monkeypatch.setattr(competitor_analysis.subprocess, "run", fail_if_called)

    identity = candidate_identity(tmp_path)

    assert identity == {"immutable_sha": None, "commit_status": "untrusted_candidate_requires_explicit_identity"}


def test_trusted_candidate_git_uses_sanitized_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Result:
        stdout = "?? uncommitted.py\n"

    def record_call(command: list[str], **kwargs: object) -> Result:
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(competitor_analysis, "TRUSTED_CANDIDATE_ROOT", tmp_path.resolve())
    monkeypatch.setattr(competitor_analysis.subprocess, "run", record_call)

    identity = candidate_identity(tmp_path)

    assert identity == {"immutable_sha": None, "commit_status": "uncommitted"}
    command, options = calls[0]
    assert command[:7] == [
        "git",
        "-c",
        "core.fsmonitor=",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
    ]
    assert options["env"] == {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": "/bin:/usr/bin",
    }


def test_manifest_only_analysis_keeps_all_scores_unknown() -> None:
    report = analyze(load_json_yaml(MANIFEST_PATH), load_json_yaml(RUBRIC_PATH))

    assert len(report["competitors"]) == 44
    assert all(item["scan"]["scan_status"] == "source_not_provided" for item in report["competitors"])
    assert all(item["scoring"]["comparative_score"] is None for item in report["competitors"])


def test_report_and_workflow_keep_static_only_contract() -> None:
    document = (ROOT / "docs" / "competitive-analysis.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "competitive-benchmark.yml").read_text(encoding="utf-8")

    assert "evals/competitors/manifest.yaml" in document
    assert "unknown" in document
    assert "--write-reconstruction" in document
    assert "permissions:\n  contents: read" in workflow
    assert "fetch_competitor_metadata.py --refresh" in workflow
    assert "analyze_competitors.py --output" in workflow
    assert "--write-reconstruction" not in workflow
    assert "docker" not in workflow.casefold()
    assert "pytest" not in workflow.casefold()


def test_manifests_are_json_compatible_yaml() -> None:
    for path in (MANIFEST_PATH, RUBRIC_PATH):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
