from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_wheel_contains_versioned_policy_packs_and_reviewed_prompts(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source_root / "pyproject.toml")
    shutil.copytree(
        ROOT / "src",
        source_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    output = tmp_path / "dist"
    result = subprocess.run(
        ("uv", "build", "--wheel", "--offline", "--out-dir", str(output)),
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    wheels = tuple(output.glob("mtbank_ai-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        contents = set(wheel.namelist())
        for policy_name in ("compliance", "quality", "roles", "taxonomy"):
            assert f"mtbank_ai/policies/{policy_name}/v1.yaml" in contents
        for agent_name in ("classifier", "compliance", "quality", "summarizer", "trends"):
            prompt_path = f"mtbank_ai/agents/{agent_name}/prompt.md"
            assert prompt_path in contents
            prompt = wheel.read(prompt_path).decode("utf-8").strip()
            assert prompt
            if agent_name == "trends":
                assert "submit_trend" in prompt
