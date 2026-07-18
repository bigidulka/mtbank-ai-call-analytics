from __future__ import annotations

import pytest

from mtbank_ai.runtime_secrets import RUNTIME_SECRET_NAMES, SecretConfigurationError, validate_runtime_secrets

_SAFE_SECRETS = {
    "WEBUI_ADMIN_PASSWORD": "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F",
    "WEBUI_SECRET_KEY": "P4!mZ8@rC2#vL6$kQ9%tH3^dW7&nF1*B",
    "PIPELINES_API_KEY": "G6!xR1@pK8#sV3$mD9%qL2^hT7&cN4*J",
    "MTBANK_ATTACHMENT_SIGNING_KEY": "B9!wH4@zM7#rQ2$kF8%vL1^pC6&xT3*D",
    "MTBANK_API_KEY": "J3!sD8@vN1#kR6$mC9%wF2^pL7&xH4*Q",
    "POSTGRES_PASSWORD": "T8!pK2@wF7#nC4$mR1%vH9^sL6&xD3*G",
    "GROQ_API_KEY": "R5!gQ8@zL2#vM7$kC4%pH9^wD1&xF6*B",
}


@pytest.mark.parametrize("name", RUNTIME_SECRET_NAMES)
@pytest.mark.parametrize(
    "unsafe_value",
    (
        "0p3n-w3bu!",
        "example-secret-value-that-is-long-enough",
        "change-me-to-a-real-secret-value-now",
        "abc" * 12,
        "too-short",
    ),
)
def test_preflight_rejects_unsafe_value_for_every_secret(name: str, unsafe_value: str) -> None:
    environment = {**_SAFE_SECRETS, name: unsafe_value}

    with pytest.raises(SecretConfigurationError) as error:
        validate_runtime_secrets(environment)

    assert name in str(error.value)
    assert unsafe_value not in str(error.value)


def test_preflight_accepts_distinct_long_runtime_secrets() -> None:
    validate_runtime_secrets(_SAFE_SECRETS)
