"""Завершает Compose preflight при небезопасных runtime-секретах."""

from __future__ import annotations

import os
import sys

from mtbank_ai.runtime_secrets import SecretConfigurationError, validate_runtime_secrets


def main() -> None:
    try:
        validate_runtime_secrets(os.environ)
    except SecretConfigurationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print("Проверка runtime-секретов завершена успешно.")


if __name__ == "__main__":
    main()
