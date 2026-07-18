#!/usr/bin/env python3
"""Validate checked test-data provenance and block score release without speech corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from mtbank_ai.speech.dataset import ManifestError, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("test_data/manifest.yaml"))
    parser.add_argument("--require-release-corpus", action="store_true")
    arguments = parser.parse_args()
    try:
        entries = validate_manifest(arguments.manifest, require_release_corpus=arguments.require_release_corpus)
    except ManifestError as error:
        print(f"manifest validation failed: {error}")
        return 1
    print(f"manifest validation passed: {len(entries)} fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
