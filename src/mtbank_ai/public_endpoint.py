"""DNS-aware fail-closed validation before sending a bearer credential."""

from __future__ import annotations

import socket
from ipaddress import ip_address


class PublicEndpointError(ValueError):
    """A credential-bearing endpoint is not a resolvable public DNS authority."""


def require_public_dns_host(host: str, port: int) -> None:
    """Rejects numeric aliases and every DNS answer that is not globally routable."""

    normalized = host.casefold().removesuffix(".")
    if not normalized.isascii() or normalized.endswith(".internal") or normalized in {"localhost", "local"}:
        raise PublicEndpointError("credential-bearing endpoint должен использовать public DNS host")
    try:
        ip_address(normalized)
    except ValueError:
        pass
    else:
        raise PublicEndpointError("credential-bearing endpoint не должен использовать numeric host")
    try:
        addresses = socket.getaddrinfo(normalized, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise PublicEndpointError("credential-bearing endpoint DNS resolution failed") from error
    if not addresses:
        raise PublicEndpointError("credential-bearing endpoint DNS resolution returned no addresses")
    for _family, _type, _protocol, _canonical_name, address in addresses:
        try:
            resolved = ip_address(address[0])
        except ValueError as error:
            raise PublicEndpointError("credential-bearing endpoint DNS returned invalid address") from error
        if not resolved.is_global:
            raise PublicEndpointError("credential-bearing endpoint DNS returned non-public address")
