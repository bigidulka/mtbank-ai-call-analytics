from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
EXPECTED_COHORT_SIZE = 44
CANONICAL_RECORDS_SHA256 = "26c3dc79168d96a00d16838146ccd5ffff079c8bad89e7f2b40ebd0acdd0886c"
CANONICAL_REPOSITORIES = (
    "Carcajo/mtbank-ai-call-analytics",
    "PchelentsovRoman/mtbank-test-assignment",
    "JustiZzZz/mtbank-ai-transcription",
    "devAsmodeus/mtbank-ai-hiring",
    "ib0gdan/speech-analytics",
    "kutsydanil/AI-Contact-Analytics",
    "antonsokol1542-beep/MTBank-AI-Engineer-",
    "ZubikIT/mtbank-ai-hiring",
    "Prime-Publisher/mtbank-game-proposal",
    "rdammala-org/MTBank_Senior-SRE-Engineer-Observability",
    "revus05/mtbank",
    "fakysim7/mtbank_game_backend",
    "anxthercode/mtbank",
    "fedosikser/mtbank_coop",
    "weblov33/mtbank_coop",
    "MarkCesium/mtbank_hackathon",
    "Fuz483/MTBankGame",
    "Fuz483/mtbank-final-web",
    "Fuz483/mtbankgame-server",
    "ILYUXXXA/MTBankWeb",
    "Kurumilog/mtbank",
    "iluusha/MTbank",
    "Temkinn/MTBankGame",
    "ILYUXXXA/MTBank-Car-Merge",
    "ILYUXXXA/MTBankMerge",
    "maturelion/mtbank",
    "mirra-games/mtbank-game-proposal",
    "meghatiw/MTBank",
    "mtbanks/mtbanks.github.io",
    "majortomdev/MTBankInc",
    "motesctf/mtbank.by",
    "motesctf/mtblog.mtbank.by",
    "dimasnytin/mtbank-media-stock",
    "marinchi03/mtbank",
    "MastaWorks/UBHacking-MTBank-Challenge",
    "skazmasters/mtbank",
    "sashabely221100/mtbank",
    "krishnamca100/MTBANK",
    "exxt505/MTBankFails",
    "YouAreNext/mtbank",
    "geoffreywaynehall/MTBank",
    "zankoav/html-mtb",
    "peryomin/ORRMSchedule",
    "AlexeyShakal/MTBank",
)
CANONICAL_EXCLUSION = {
    "repo": "vbuyel/mtbank-ai-hiring",
    "created_at_utc": "2026-07-15T12:14:03Z",
    "reason": "created after original transcript request",
}
EMPTY_REPOSITORY = "AlexeyShakal/MTBank"
EMPTY_REPOSITORY_STATUS = "empty_repository_no_immutable_commit"


class ManifestError(ValueError):
    """The frozen competitor cohort does not satisfy its safety contract."""


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load the repository's JSON-compatible YAML without a runtime YAML dependency."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a mapping")
    return value


def dump_json_yaml(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def canonical_records_digest(records: list[dict[str, Any]]) -> str:
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "competitor-cohort/v1":
        raise ManifestError("schema must be competitor-cohort/v1")
    if manifest.get("original_query") != "q=mtbank&sort=updated&order=desc":
        raise ManifestError("original_query is not the frozen search query")
    if manifest.get("original_total_count_claim") != EXPECTED_COHORT_SIZE:
        raise ManifestError("original_total_count_claim must be 44")
    if manifest.get("checked_at_utc") is not None:
        raise ManifestError("checked_at_utc must remain null because no clock evidence exists")
    if manifest.get("excluded_after_original_request") != CANONICAL_EXCLUSION:
        raise ManifestError("vbuyel exclusion must match the frozen historical reconstruction")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_COHORT_SIZE:
        raise ManifestError("records must contain exactly 44 entries")

    repositories: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ManifestError("each record must be a mapping")
        repo = record.get("repo")
        if not isinstance(repo, str) or repo.count("/") != 1:
            raise ManifestError("record repo must be owner/name")
        if repo in repositories:
            raise ManifestError(f"duplicate repository: {repo}")
        repositories.add(repo)
        if record.get("url") != f"https://github.com/{repo}":
            raise ManifestError(f"record URL does not match repo: {repo}")
        if not isinstance(record.get("branch"), str) or not record["branch"]:
            raise ManifestError(f"record branch missing: {repo}")

        sha = record.get("sha")
        if sha is None:
            if repo != EMPTY_REPOSITORY:
                raise ManifestError(f"only {EMPTY_REPOSITORY} may lack an immutable SHA")
            if record.get("status") != EMPTY_REPOSITORY_STATUS or not record.get("unavailable_reason"):
                raise ManifestError(f"unavailable SHA needs the frozen empty-repository status: {repo}")
        elif not isinstance(sha, str) or SHA_PATTERN.fullmatch(sha) is None:
            raise ManifestError(f"SHA must be a 40-character lowercase commit: {repo}")

        for name in ("archived",):
            if not isinstance(record.get(name), bool):
                raise ManifestError(f"record {name} must be boolean: {repo}")
        for name in ("stars", "forks"):
            if not isinstance(record.get(name), int) or record[name] < 0:
                raise ManifestError(f"record {name} must be a non-negative integer: {repo}")
        if not isinstance(record.get("license_status"), str):
            raise ManifestError(f"record license_status missing: {repo}")

    if tuple(record["repo"] for record in records) != CANONICAL_REPOSITORIES:
        raise ManifestError("record repositories must match the canonical frozen order")
    if canonical_records_digest(records) != CANONICAL_RECORDS_SHA256:
        raise ManifestError("record content does not match the canonical frozen cohort digest")

    empty = next(record for record in records if record["repo"] == EMPTY_REPOSITORY)
    if empty.get("sha") is not None or empty.get("status") != EMPTY_REPOSITORY_STATUS:
        raise ManifestError("AlexeyShakal/MTBank must remain the documented empty repository")


def repository_directory_name(repo: str) -> str:
    return repo.replace("/", "__")
