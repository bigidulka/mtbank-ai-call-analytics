#!/usr/bin/env python3
"""Provisioning закреплённых local speech artifacts; runtime остаётся offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from pydantic import StringConstraints, ValidationError, field_validator, model_validator

from mtbank_ai.domain.base import NonEmptyId, Sha256, StrictFrozenModel
from services.speech.manifest import ModelArtifact, SpeechModelManifest, artifact_tree_sha256

_COMPONENTS = ("diarization",)
_PROVENANCE_FILENAME = ".mtbank-speech-provenance.json"
_ARTIFACT_CONTENT_DIGEST_SCHEMA = "v2"
_ARTIFACT_CONTENT_DIGEST_DOMAIN = b"mtbank-ai:speech-artifact-content:v2\x00"
_TOKEN_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class ProvisioningError(RuntimeError):
    """Provisioning блокируется без раскрытия token или деталей remote response."""


class ModelPackage(StrictFrozenModel):
    name: NonEmptyId
    version: NonEmptyId


class ModelSource(StrictFrozenModel):
    repo_id: NonEmptyId
    expected_resolved_repo_id: NonEmptyId | None = None
    model_id: NonEmptyId
    revision: GitRevision
    expected_artifact_content_sha256: Sha256 | None = None
    license: NonEmptyId
    gated: bool
    relative_path: NonEmptyId
    package: ModelPackage

    @field_validator("relative_path")
    @classmethod
    def require_clean_top_level_directory(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
            raise ValueError("relative_path должен именовать одну директорию внутри artifact_root")
        return value

    @model_validator(mode="after")
    def require_reviewed_public_artifact_digest(self) -> ModelSource:
        if not self.gated and self.expected_artifact_content_sha256 is None:
            raise ValueError("public model source requires reviewed artifact content SHA-256")
        return self


class ModelSources(StrictFrozenModel):
    schema_version: Literal[2] = 2
    artifact_content_digest_schema: Literal["v2"] = _ARTIFACT_CONTENT_DIGEST_SCHEMA
    sources: dict[str, ModelSource]

    @model_validator(mode="after")
    def require_exact_components(self) -> ModelSources:
        if set(self.sources) != set(_COMPONENTS):
            raise ValueError("model sources должны содержать ровно diarization")
        paths = tuple(self.sources[name].relative_path for name in _COMPONENTS)
        if len(set(paths)) != len(paths):
            raise ValueError("model source paths должны быть уникальны")
        return self


class ArtifactProvenance(StrictFrozenModel):
    schema_version: Literal[2] = 2
    artifact_content_digest_schema: Literal["v2"] = _ARTIFACT_CONTENT_DIGEST_SCHEMA
    repo_id: NonEmptyId
    expected_resolved_repo_id: NonEmptyId | None = None
    model_id: NonEmptyId
    model_revision: GitRevision
    package: NonEmptyId
    package_version: NonEmptyId
    license: NonEmptyId
    artifact_content_sha256: Sha256


class LegacyArtifactProvenance(StrictFrozenModel):
    """Предыдущий script-owned marker, допустимый только для atomic migration."""

    schema_version: Literal[1]
    repo_id: NonEmptyId
    expected_resolved_repo_id: NonEmptyId | None = None
    model_id: NonEmptyId
    model_revision: GitRevision
    package: NonEmptyId
    package_version: NonEmptyId
    license: NonEmptyId
    artifact_content_sha256: Sha256


class HubApi(Protocol):
    def model_info(self, *, repo_id: str, revision: str, token: str | None) -> object: ...


@dataclass(frozen=True)
class HubFunctions:
    api_factory: Callable[[], HubApi]
    snapshot_download: Callable[..., str | Path]


@dataclass
class _ManifestPublication:
    staging_path: Path | None = None
    created: bool = False


def load_model_sources(path: Path) -> ModelSources:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ModelSources.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ProvisioningError("model sources are unavailable or invalid") from error


def load_hub_functions() -> HubFunctions:
    """Лениво импортирует transitive Hub dependency только для explicit provisioning."""

    try:
        import huggingface_hub  # pyright: ignore[reportMissingImports]
        from huggingface_hub import HfApi, snapshot_download  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ProvisioningError("huggingface-hub 0.36.x is unavailable in the speech environment") from error
    version = getattr(huggingface_hub, "__version__", "")
    if not isinstance(version, str) or not version.startswith("0.36."):
        raise ProvisioningError("huggingface-hub 0.36.x is required for provisioning")
    return HubFunctions(api_factory=HfApi, snapshot_download=snapshot_download)


def provision(
    *,
    sources_path: Path,
    artifact_root: Path,
    output_manifest: Path,
    cache_dir: Path,
    components: Sequence[str] | None = None,
    token_env_name: str = "HF_TOKEN",
    environment: Mapping[str, str | None] | None = None,
    hub: HubFunctions | None = None,
) -> SpeechModelManifest | None:
    """Provision selected components and publish a manifest only for a complete verified set."""

    sources = load_model_sources(sources_path)
    selected = _selected_components(components)
    _, output, resolved_cache_dir, targets = _preflight_destinations(
        artifact_root,
        output_manifest,
        cache_dir,
        sources,
    )
    existing = {
        name: _verify_component_artifact(name, sources.sources[name], targets[name])
        for name in _COMPONENTS
        if _exists_or_symlink(targets[name])
    }
    if _exists_or_symlink(output) and len(existing) != len(_COMPONENTS):
        raise ProvisioningError("model manifest exists while configured artifacts are incomplete")

    missing = tuple(name for name in selected if name not in existing)
    for name in missing:
        _require_reviewed_artifact_digest(name, sources.sources[name])
    token = _read_token(token_env_name, os.environ if environment is None else environment)
    if any(sources.sources[name].gated for name in missing) and token is None:
        raise ProvisioningError(f"gated model requires token environment variable {token_env_name}")

    published_targets: tuple[Path, ...] = ()
    manifest_publication = _ManifestPublication()
    committed = False
    try:
        provisioned, published_targets = _provision_missing_components(
            names=missing,
            sources=sources,
            targets=targets,
            cache_dir=resolved_cache_dir,
            token=token,
            hub=hub,
        )
        artifacts = {**existing, **provisioned}
        if len(artifacts) != len(_COMPONENTS):
            return None
        manifest = SpeechModelManifest(diarization=artifacts["diarization"])
        persisted = _write_or_verify_manifest(output, manifest, publication=manifest_publication)
        committed = True
        return persisted
    except BaseException:
        if not committed:
            _cleanup_manifest_publication(output, manifest_publication)
            _remove_current_invocation_targets(published_targets)
        raise


def _selected_components(components: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(_COMPONENTS if components is None else components)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(_COMPONENTS):
        raise ProvisioningError("components must be unique values from diarization")
    return selected


def _provision_missing_components(
    *,
    names: tuple[str, ...],
    sources: ModelSources,
    targets: Mapping[str, Path],
    cache_dir: Path,
    token: str | None,
    hub: HubFunctions | None,
) -> tuple[dict[str, ModelArtifact], tuple[Path, ...]]:
    if not names:
        return {}, ()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_functions = hub or load_hub_functions()
    api = hub_functions.api_factory()
    snapshots = {
        name: _verified_snapshot(
            name=name,
            source=sources.sources[name],
            api=api,
            snapshot_download=hub_functions.snapshot_download,
            cache_dir=cache_dir,
            token=token,
        )
        for name in names
    }

    staging_directories: list[Path] = []
    published_targets: list[Path] = []
    artifacts: dict[str, ModelArtifact] = {}
    try:
        for name in names:
            target = targets[name]
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
            staging_directories.append(staging)
            _copy_snapshot(snapshot=snapshots[name], target=staging, cache_dir=cache_dir)
            _write_provenance(name, staging, sources.sources[name])
            artifact = _verify_component_artifact(name, sources.sources[name], staging)
            _normalize_artifact_permissions(staging)
            if _exists_or_symlink(target):
                raise ProvisioningError(f"{name} artifact target already exists")
            try:
                staging.rename(target)
            except OSError as error:
                raise ProvisioningError(f"{name} artifact cannot be atomically published") from error
            published_targets.append(target)
            artifacts[name] = artifact
    except BaseException:
        _remove_staging_directories(staging_directories)
        _remove_current_invocation_targets(tuple(published_targets))
        raise
    return artifacts, tuple(published_targets)


def _remove_staging_directories(staging_directories: Sequence[Path]) -> None:
    for staging in reversed(staging_directories):
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)


def _remove_current_invocation_targets(targets: Sequence[Path]) -> None:
    for target in reversed(targets):
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)


def _normalize_artifact_permissions(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProvisioningError("artifact staging directory is unavailable")
    directories: list[Path] = []
    files: list[Path] = []
    for current, child_directories, child_files in os.walk(root, followlinks=False):
        directory = Path(current)
        if directory.is_symlink():
            raise ProvisioningError("artifact staging directory contains an unsafe symlink")
        directories.append(directory)
        child_directories.sort()
        child_files.sort()
        for child in child_directories:
            if (directory / child).is_symlink():
                raise ProvisioningError("artifact staging directory contains an unsafe symlink")
        for child in child_files:
            item = directory / child
            if item.is_symlink() or not item.is_file():
                raise ProvisioningError("artifact staging directory contains an unsafe file")
            files.append(item)
    for directory in directories:
        directory.chmod(0o755)
    for item in files:
        item.chmod(0o644)


def _preflight_destinations(
    artifact_root: Path,
    output_manifest: Path,
    cache_dir: Path,
    sources: ModelSources,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    if artifact_root.is_symlink():
        raise ProvisioningError("artifact_root must not be a symlink")
    if cache_dir.is_symlink():
        raise ProvisioningError("cache_dir must not be a symlink")
    root = artifact_root.resolve(strict=False)
    output = output_manifest.resolve(strict=False)
    resolved_cache_dir = cache_dir.resolve(strict=False)
    targets = {name: root / sources.sources[name].relative_path for name in _COMPONENTS}
    for target in targets.values():
        if _paths_overlap(output, target):
            raise ProvisioningError("model manifest path must not overlap an artifact target")
    if any(_paths_overlap(resolved_cache_dir, path) for path in (root, output, *targets.values())):
        raise ProvisioningError("cache_dir must not overlap artifact_root, artifact targets, or model manifest")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.chmod(0o755)
    output_manifest.parent.chmod(0o755)
    return root, output, resolved_cache_dir, targets


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_token(name: str, environment: Mapping[str, str | None]) -> str | None:
    if not _TOKEN_ENVIRONMENT_NAME.fullmatch(name):
        raise ProvisioningError("token environment variable name is invalid")
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _require_reviewed_artifact_digest(name: str, source: ModelSource) -> str:
    if source.expected_artifact_content_sha256 is None:
        raise ProvisioningError(f"{name} source has no reviewed pinned artifact content SHA-256")
    return source.expected_artifact_content_sha256


def _verified_snapshot(
    *,
    name: str,
    source: ModelSource,
    api: HubApi,
    snapshot_download: Callable[..., str | Path],
    cache_dir: Path,
    token: str | None,
) -> Path:
    try:
        info = api.model_info(repo_id=source.repo_id, revision=source.revision, token=token)
    except Exception as error:
        raise ProvisioningError(f"{name} model metadata is unavailable") from error
    if getattr(info, "sha", None) != source.revision:
        raise ProvisioningError(f"{name} model revision does not match the configured commit")
    resolved_repo_id = getattr(info, "id", None)
    if (
        source.expected_resolved_repo_id is not None
        and isinstance(resolved_repo_id, str)
        and resolved_repo_id
        and resolved_repo_id != source.expected_resolved_repo_id
    ):
        raise ProvisioningError(f"{name} resolved repository does not match configured canonical ID")
    try:
        snapshot = snapshot_download(
            repo_id=source.repo_id,
            revision=source.revision,
            cache_dir=str(cache_dir),
            token=token,
        )
    except Exception as error:
        raise ProvisioningError(f"{name} model snapshot is unavailable") from error
    path = Path(snapshot)
    if path.is_symlink() or not path.is_dir():
        raise ProvisioningError(f"{name} model snapshot is not a directory")
    return path


def _copy_snapshot(*, snapshot: Path, target: Path, cache_dir: Path) -> None:
    copied_files = 0
    cache_root = cache_dir.resolve()
    for current, directories, files in os.walk(snapshot, followlinks=False):
        source_directory = Path(current)
        if source_directory.is_symlink():
            raise ProvisioningError("downloaded model snapshot contains an unsafe directory symlink")
        directories.sort()
        files.sort()
        for directory in directories:
            if (source_directory / directory).is_symlink():
                raise ProvisioningError("downloaded model snapshot contains an unsafe directory symlink")
        for filename in files:
            source_file = source_directory / filename
            if source_file.is_symlink():
                source_file = _cached_symlink_target(source_file, cache_root)
            if not source_file.is_file():
                raise ProvisioningError("downloaded model snapshot contains an unsafe file")
            destination = target / source_directory.relative_to(snapshot) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            copied_files += 1
    if copied_files == 0:
        raise ProvisioningError("downloaded model snapshot is empty")


def _cached_symlink_target(path: Path, cache_root: Path) -> Path:
    try:
        target = path.resolve(strict=True)
    except OSError as error:
        raise ProvisioningError("downloaded model snapshot contains a broken symlink") from error
    if cache_root not in target.parents or not target.is_file() or target.is_symlink():
        raise ProvisioningError("downloaded model snapshot symlink exits the Hub cache")
    return target


def _write_provenance(name: str, target: Path, source: ModelSource) -> None:
    actual_content_sha256 = _artifact_content_sha256(target)
    expected_content_sha256 = _require_reviewed_artifact_digest(name, source)
    if actual_content_sha256 != expected_content_sha256:
        raise ProvisioningError(f"{name} artifact content does not match reviewed pinned SHA-256")
    provenance = _expected_provenance(source, actual_content_sha256)
    marker = target / _PROVENANCE_FILENAME
    try:
        with marker.open("x", encoding="utf-8") as marker_file:
            marker_file.write(json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True))
            marker_file.write("\n")
    except FileExistsError as error:
        raise ProvisioningError("downloaded model snapshot contains a reserved provenance marker") from error


def refresh_existing_provenance(
    *,
    sources_path: Path,
    artifact_root: Path,
    components: Sequence[str] | None = None,
) -> None:
    """Atomically upgrades only verified script-owned provenance markers without downloads."""

    sources = load_model_sources(sources_path)
    selected = _selected_components(components)
    root = _resolve_existing_artifact_root(artifact_root)
    for name in selected:
        _refresh_component_provenance(name, root / sources.sources[name].relative_path, sources.sources[name])


def _resolve_existing_artifact_root(artifact_root: Path) -> Path:
    if artifact_root.is_symlink():
        raise ProvisioningError("artifact_root must not be a symlink")
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as error:
        raise ProvisioningError("artifact_root is unavailable") from error
    if not root.is_dir():
        raise ProvisioningError("artifact_root must be a directory")
    return root


def _refresh_component_provenance(name: str, target: Path, source: ModelSource) -> None:
    if target.is_symlink() or not target.is_dir():
        raise ProvisioningError(f"{name} artifact target is not a regular directory")
    marker = target / _PROVENANCE_FILENAME
    if marker.is_symlink() or not marker.is_file():
        raise ProvisioningError(f"{name} artifact has no script-owned provenance marker")
    actual_content_sha256 = _artifact_content_sha256(target)
    expected_content_sha256 = _require_reviewed_artifact_digest(name, source)
    if actual_content_sha256 != expected_content_sha256:
        raise ProvisioningError(f"{name} artifact content does not match reviewed pinned SHA-256")
    expected = _expected_provenance(source, expected_content_sha256)
    payload = _read_provenance_payload(name, marker)
    try:
        current = ArtifactProvenance.model_validate(payload)
    except ValidationError:
        try:
            legacy = LegacyArtifactProvenance.model_validate(payload)
        except ValidationError as error:
            raise ProvisioningError(f"{name} artifact provenance is invalid") from error
        if not _provenance_identity_matches(legacy, expected):
            raise ProvisioningError(f"{name} artifact provenance does not match configured source")
        _atomically_replace_provenance(marker, expected)
        return
    if current != expected:
        raise ProvisioningError(f"{name} artifact provenance or content does not match configured source")
    marker.chmod(0o644)


def _read_provenance_payload(name: str, marker: Path) -> dict[str, object]:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"{name} artifact provenance is invalid") from error
    if not isinstance(payload, dict):
        raise ProvisioningError(f"{name} artifact provenance is invalid")
    return cast(dict[str, object], payload)


def _provenance_identity_matches(
    provenance: LegacyArtifactProvenance,
    expected: ArtifactProvenance,
) -> bool:
    return (
        provenance.repo_id == expected.repo_id
        and provenance.expected_resolved_repo_id == expected.expected_resolved_repo_id
        and provenance.model_id == expected.model_id
        and provenance.model_revision == expected.model_revision
        and provenance.package == expected.package
        and provenance.package_version == expected.package_version
        and provenance.license == expected.license
    )


def _expected_provenance(source: ModelSource, artifact_content_sha256: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        repo_id=source.repo_id,
        expected_resolved_repo_id=source.expected_resolved_repo_id,
        model_id=source.model_id,
        model_revision=source.revision,
        package=source.package.name,
        package_version=source.package.version,
        license=source.license,
        artifact_content_sha256=artifact_content_sha256,
    )


def _atomically_replace_provenance(marker: Path, provenance: ArtifactProvenance) -> None:
    staging: Path | None = None
    try:
        descriptor, staging_name = tempfile.mkstemp(prefix=f".{marker.name}.staging-", dir=marker.parent)
        staging = Path(staging_name)
        with os.fdopen(descriptor, "wb") as marker_file:
            os.fchmod(marker_file.fileno(), 0o644)
            marker_file.write(_serialize_json(provenance))
            marker_file.flush()
            os.fsync(marker_file.fileno())
        if marker.is_symlink() or not marker.is_file():
            raise ProvisioningError("artifact provenance marker is no longer a regular file")
        os.replace(staging, marker)
        staging = None
        _fsync_directory(marker.parent)
    except OSError as error:
        raise ProvisioningError("artifact provenance marker cannot be atomically updated") from error
    finally:
        if staging is not None and _exists_or_symlink(staging):
            staging.unlink()


def _serialize_json(value: StrictFrozenModel) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verify_component_artifact(name: str, source: ModelSource, target: Path) -> ModelArtifact:
    expected_content_sha256 = _require_reviewed_artifact_digest(name, source)
    if target.is_symlink() or not target.is_dir():
        raise ProvisioningError(f"{name} artifact target is not a regular directory")
    marker = target / _PROVENANCE_FILENAME
    if marker.is_symlink() or not marker.is_file():
        raise ProvisioningError(f"{name} artifact has no script-owned provenance marker")
    try:
        provenance = ArtifactProvenance.model_validate(_read_provenance_payload(name, marker))
    except ValidationError as error:
        raise ProvisioningError(f"{name} artifact provenance is invalid") from error
    actual_content_sha256 = _artifact_content_sha256(target)
    if actual_content_sha256 != expected_content_sha256:
        raise ProvisioningError(f"{name} artifact content does not match reviewed pinned SHA-256")
    expected = _expected_provenance(source, expected_content_sha256)
    if provenance != expected:
        raise ProvisioningError(f"{name} artifact provenance or content does not match configured source")
    try:
        tree_sha256 = artifact_tree_sha256(target)
    except Exception as error:
        raise ProvisioningError(f"{name} artifact tree verification failed") from error
    return ModelArtifact(
        package=source.package.name,
        package_version=source.package.version,
        model_id=source.model_id,
        model_revision=source.revision,
        relative_path=source.relative_path,
        artifact_sha256=tree_sha256,
    )


def _artifact_content_sha256(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise ProvisioningError("artifact directory is unavailable")
    files: list[Path] = []
    for current, directories, filenames in os.walk(path, followlinks=False):
        directory = Path(current)
        if directory.is_symlink():
            raise ProvisioningError("artifact contains an unsafe directory symlink")
        directories.sort()
        filenames.sort()
        for child in directories:
            if (directory / child).is_symlink():
                raise ProvisioningError("artifact contains an unsafe directory symlink")
        for filename in filenames:
            item = directory / filename
            if item.is_symlink() or not item.is_file():
                raise ProvisioningError("artifact contains an unsafe file")
            if item.relative_to(path).as_posix() != _PROVENANCE_FILENAME:
                files.append(item)
    if not files:
        raise ProvisioningError("artifact directory is empty")
    digest = hashlib.sha256()
    digest.update(_ARTIFACT_CONTENT_DIGEST_DOMAIN)
    digest.update(len(files).to_bytes(8, "big"))
    for item in sorted(files):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        content_sha256 = hashlib.sha256()
        content_length = 0
        with item.open("rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                content_length += len(chunk)
                content_sha256.update(chunk)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_length.to_bytes(8, "big"))
        digest.update(content_sha256.digest())
    return digest.hexdigest()


def _write_or_verify_manifest(
    output: Path,
    manifest: SpeechModelManifest,
    *,
    publication: _ManifestPublication,
) -> SpeechModelManifest:
    if _exists_or_symlink(output):
        if output.is_symlink() or not output.is_file():
            raise ProvisioningError("model manifest is not a regular file")
        try:
            persisted = SpeechModelManifest.model_validate(json.loads(output.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ProvisioningError("model manifest is invalid") from error
        if persisted != manifest:
            raise ProvisioningError("model manifest does not match verified artifacts")
        try:
            output.chmod(0o644)
        except OSError as error:
            raise ProvisioningError("model manifest permissions cannot be normalized") from error
        return persisted

    _write_manifest_atomically(output, manifest, publication)
    return manifest


def _write_manifest_atomically(
    output: Path,
    manifest: SpeechModelManifest,
    publication: _ManifestPublication,
) -> None:
    try:
        descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.staging-", dir=output.parent)
        staging = Path(staging_name)
        publication.staging_path = staging
        with os.fdopen(descriptor, "wb") as manifest_file:
            os.fchmod(manifest_file.fileno(), 0o644)
            manifest_file.write(_serialize_json(manifest))
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as error:
            raise ProvisioningError("model manifest already exists") from error
        publication.created = True
        _fsync_directory(output.parent)
    except OSError as error:
        raise ProvisioningError("model manifest cannot be atomically published") from error
    else:
        _discard_manifest_staging(publication)


def _cleanup_manifest_publication(output: Path, publication: _ManifestPublication) -> None:
    staging = publication.staging_path
    if not publication.created and staging is not None and _manifest_staging_owns_output(staging, output):
        publication.created = True
    if publication.created and _exists_or_symlink(output):
        if output.is_symlink() or not output.is_file():
            raise ProvisioningError("new model manifest is no longer a regular file")
        output.unlink()
    _discard_manifest_staging(publication)


def _manifest_staging_owns_output(staging: Path, output: Path) -> bool:
    if output.is_symlink() or not _exists_or_symlink(staging) or not _exists_or_symlink(output):
        return False
    try:
        return os.path.samestat(staging.stat(), output.stat())
    except OSError:
        return False


def _discard_manifest_staging(publication: _ManifestPublication) -> None:
    staging = publication.staging_path
    if staging is not None and _exists_or_symlink(staging):
        staging.unlink()
    publication.staging_path = None


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("services/speech/model-sources.json"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--component", action="append", choices=_COMPONENTS)
    parser.add_argument("--token-env-name", default="HF_TOKEN")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    try:
        provision(
            sources_path=parsed.sources,
            artifact_root=parsed.artifact_root,
            output_manifest=parsed.output_manifest,
            cache_dir=parsed.cache_dir,
            components=parsed.component,
            token_env_name=parsed.token_env_name,
        )
    except ProvisioningError as error:
        print(f"provisioning blocked: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
