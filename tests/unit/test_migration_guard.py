from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def _migration_tests() -> ModuleType:
    path = ROOT / "tests" / "integration" / "test_postgres_migrations.py"
    spec = importlib.util.spec_from_file_location("postgres_migration_contract", path)
    if spec is None or spec.loader is None:
        raise AssertionError("migration test module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("url", "opt_in", "message"),
    (
        ("postgresql+asyncpg://user:password@localhost:5432/mtbank_test_guard", None, "явный"),
        ("postgresql://user:password@localhost:5432/mtbank_test_guard", "1", r"postgresql\+asyncpg"),
        (
            "postgresql+asyncpg://user:password@localhost:5432/mtbank_test_guard?ssl=false",
            "1",
            "query parameters",
        ),
        ("postgresql+asyncpg://user:password@localhost:5432/production", "1", "mtbank_test_"),
        ("postgresql+asyncpg:///mtbank_test_guard", "1", "недопустимый"),
        ("postgresql+asyncpg://user@localhost:5432/mtbank_test_guard", "1", "explicit authority"),
        ("postgresql+asyncpg://user:password@localhost:5432/mtbank_test_guard#fragment", "1", "недопустимый"),
        ("postgresql+asyncpg://user:raw@password@localhost:5432/mtbank_test_guard", "1", "недопустимый"),
    ),
)
def test_destructive_migration_guard_rejects_unsafe_targets_without_connecting(
    url: str,
    opt_in: str | None,
    message: str,
) -> None:
    module = _migration_tests()

    with pytest.raises(module.DestructiveDatabaseGuardError, match=message) as error:
        module._validate_destructive_database_target(url, opt_in)

    assert "password" not in str(error.value)
    assert url not in str(error.value)


@pytest.mark.parametrize("database_name", ("mtbank_test_guard", "guard_test"))
def test_destructive_migration_guard_accepts_only_explicit_test_database_paths(
    database_name: str,
) -> None:
    module = _migration_tests()
    url, expected_database = module._validate_destructive_database_target(
        f"postgresql+asyncpg://user:password@localhost:5432/{database_name}",
        "1",
    )

    assert url.drivername == "postgresql+asyncpg"
    assert not url.query
    assert expected_database == database_name


def test_destructive_migration_guard_hides_malformed_nfkc_url_secret() -> None:
    module = _migration_tests()
    secret = "NFKC-secret-sentinel"
    malformed_url = f"postgresql+asyncpg://user:{secret}@host：5432/mtbank_test_guard"

    with pytest.raises(module.DestructiveDatabaseGuardError) as error:
        module._validate_destructive_database_target(malformed_url, "1")

    assert secret not in str(error.value)
    assert malformed_url not in str(error.value)


def test_destructive_migration_guard_rejects_partial_and_ambient_configuration() -> None:
    module = _migration_tests()

    with pytest.raises(pytest.fail.Exception, match="требует URL"):
        module._database_target({"MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS": "1"})
    with pytest.raises(pytest.fail.Exception, match="ambient PostgreSQL"):
        module._database_target(
            {
                "MTBANK_TEST_DATABASE_URL": "postgresql+asyncpg://user:password@localhost:5432/mtbank_test_guard",
                "MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS": "1",
                "PGHOST": "localhost",
            }
        )
