#!/usr/bin/env python3
"""Optionally refresh public GitHub metadata without executing competitor content.

No request is made unless --refresh is supplied. The frozen manifest is never changed
unless --write-reconstruction is also supplied, and a unified diff is always emitted.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    from scripts.competitor_common import dump_json_yaml, load_json_yaml, validate_manifest
except ModuleNotFoundError:  # Direct invocation: python scripts/fetch_competitor_metadata.py
    from competitor_common import dump_json_yaml, load_json_yaml, validate_manifest

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "evals" / "competitors" / "manifest.yaml"
SEARCH_URL = "https://api.github.com/search/repositories?q=mtbank&sort=updated&order=desc&per_page=100&page=1"
EXCLUDED_REPOSITORY = "vbuyel/mtbank-ai-hiring"


def get_public_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "mtbank-static-benchmark"})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - explicit public GitHub REST endpoint
        return json.loads(response.read().decode("utf-8"))


def refreshed_manifest(manifest: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    """Update public search metadata while retaining the documented historical exclusion."""
    items = search.get("items")
    if not isinstance(items, list):
        raise ValueError("GitHub response has no repository items")
    by_name = {item.get("full_name"): item for item in items if isinstance(item, dict)}
    rebuilt = json.loads(json.dumps(manifest))
    rebuilt["current_total_count"] = search.get("total_count")
    rebuilt["checked_at_utc"] = None
    for record in rebuilt["records"]:
        item = by_name.get(record["repo"])
        if item is None:
            continue
        record["archived"] = bool(item.get("archived"))
        record["license_status"] = (item.get("license") or {}).get("spdx_id") or "NOASSERTION"
        record["language"] = item.get("language")
        record["stars"] = item.get("stargazers_count", 0)
        record["forks"] = item.get("forks_count", 0)
        record["branch"] = item.get("default_branch") or record["branch"]
    return rebuilt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Allow one public unauthenticated GitHub REST search request",
    )
    parser.add_argument(
        "--write-reconstruction",
        action="store_true",
        help="Write the reconstructed manifest after --refresh; requires human review of emitted diff",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.write_reconstruction and not arguments.refresh:
        print("--write-reconstruction requires --refresh", file=sys.stderr)
        return 2
    if not arguments.refresh:
        print("No network request: pass --refresh to request public GitHub metadata.")
        return 0

    try:
        current = load_json_yaml(arguments.manifest)
        validate_manifest(current)
        search = get_public_json(SEARCH_URL)
        refreshed = refreshed_manifest(current, search)
        validate_manifest(refreshed)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"metadata refresh failed: {error}", file=sys.stderr)
        return 1

    old_text = dump_json_yaml(current)
    new_text = dump_json_yaml(refreshed)
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(arguments.manifest),
            tofile=str(arguments.manifest),
        )
    )
    print(diff or "No metadata differences.", end="" if diff else "\n")
    if arguments.write_reconstruction:
        arguments.manifest.write_text(new_text, encoding="utf-8")
        print("Wrote reconstructed metadata after emitting diff.")
    else:
        print("Frozen manifest was not changed; pass --write-reconstruction only after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
