#!/usr/bin/env python3
"""Write a checked local speech model manifest after separate artifact provisioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.speech.manifest import ModelArtifact, SpeechModelManifest, artifact_tree_sha256

_COMPONENTS = ("diarization",)


def _component(arguments: argparse.Namespace, name: str, root: Path) -> ModelArtifact:
    relative_path = getattr(arguments, f"{name}_relative_path")
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{name} artifact path exits artifact root")
    return ModelArtifact(
        package=getattr(arguments, f"{name}_package"),
        package_version=getattr(arguments, f"{name}_package_version"),
        model_id=getattr(arguments, f"{name}_model_id"),
        model_revision=getattr(arguments, f"{name}_model_revision"),
        relative_path=relative_path,
        artifact_sha256=artifact_tree_sha256(path),
    )


def generate(arguments: argparse.Namespace) -> SpeechModelManifest:
    root = arguments.artifact_root.resolve()
    return SpeechModelManifest(diarization=_component(arguments, "diarization", root))


def _add_component_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(f"--{name}-relative-path", required=True)
    parser.add_argument(f"--{name}-package", required=True)
    parser.add_argument(f"--{name}-package-version", required=True)
    parser.add_argument(f"--{name}-model-id", required=True)
    parser.add_argument(f"--{name}-model-revision", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for component in _COMPONENTS:
        _add_component_arguments(parser, component)
    arguments = parser.parse_args()
    manifest = generate(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
