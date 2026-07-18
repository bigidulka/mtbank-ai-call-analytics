#!/usr/bin/env python3
"""Fail-closed prepare/publish provisioner для remotely managed Cloudflare Tunnel."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlencode
from uuid import UUID

import httpx

ROOT = Path(__file__).parents[1]
TMP_ROOT = ROOT / "tmp"
STATE_FILE = TMP_ROOT / "cloudflare-tunnel-state.json"
API_BASE_URL = "https://api.cloudflare.com/client/v4"
CONNECTOR_UID_GID = 65532
TERMINAL_INGRESS = {"service": "http_status:404"}
DNS_COMMENT = "managed-by:mtbank-ai-cloudflare-provisioner"
_HEX_ID = re.compile(r"^[a-f0-9]{32}$")


class ProvisionError(RuntimeError):
    """Безопасная ошибка provisioner без HTTP body и credentials."""


@dataclass(frozen=True, slots=True)
class Settings:
    email: str
    global_key: str
    zone: str
    hostname: str
    tunnel_name: str
    token_file: Path

    def ingress_config(self) -> dict[str, object]:
        return {
            "config": {
                "ingress": [
                    {
                        "hostname": self.hostname,
                        "service": "http://gateway:8080",
                        "originRequest": {},
                    },
                    TERMINAL_INGRESS,
                ]
            }
        }


@dataclass(frozen=True, slots=True)
class OwnershipState:
    account_id: str
    zone_id: str
    tunnel_id: str
    zone: str
    hostname: str
    tunnel_name: str


class CloudflareClient(Protocol):
    def request(self, method: str, path: str, payload: object | None = None) -> object: ...


class HttpCloudflareClient:
    def __init__(self, settings: Settings) -> None:
        self._headers = {
            "Accept": "application/json",
            "X-Auth-Email": settings.email,
            "X-Auth-Key": settings.global_key,
        }
        self._client = httpx.Client(
            timeout=httpx.Timeout(10.0),
            trust_env=False,
            follow_redirects=False,
            headers=self._headers,
        )

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        try:
            with self._client.stream(
                method,
                f"{API_BASE_URL}{path}",
                json=payload,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise ProvisionError("Cloudflare API request failed")
                content = _read_bounded(response, 128 * 1024)
        except httpx.HTTPError as error:
            raise ProvisionError("Cloudflare API request failed") from error
        try:
            rendered = json.loads(content)
        except json.JSONDecodeError as error:
            raise ProvisionError("Cloudflare API returned invalid JSON") from error
        if not isinstance(rendered, dict) or rendered.get("success") is not True or "result" not in rendered:
            raise ProvisionError("Cloudflare API rejected provisioning request")
        return rendered["result"]


def _read_bounded(response: httpx.Response, maximum_bytes: int) -> bytes:
    content = bytearray()
    for chunk in response.iter_raw():
        if len(chunk) > maximum_bytes - len(content):
            raise ProvisionError("Cloudflare API response exceeds configured bound")
        content.extend(chunk)
    return bytes(content)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ProvisionError(f"{name} is required")
    return value


def _validate_hostname(value: str) -> str:
    normalized = value.casefold().removesuffix(".")
    if not normalized or len(normalized) > 253 or any(part == "" or len(part) > 63 for part in normalized.split(".")):
        raise ProvisionError("CLOUDFLARE_HOSTNAME is invalid")
    if any(not (character.isascii() and (character.isalnum() or character in {"-", "."})) for character in normalized):
        raise ProvisionError("CLOUDFLARE_HOSTNAME is invalid")
    return normalized


def _validate_name(name: str, value: str) -> str:
    if value != value.strip() or not value or len(value) > 100:
        raise ProvisionError(f"{name} is invalid")
    return value


def load_settings() -> Settings:
    return Settings(
        email=_validate_name("CF_EMAIL", _required_environment("CF_EMAIL")),
        global_key=_validate_name("CF_GLOBAL_API_KEY", _required_environment("CF_GLOBAL_API_KEY")),
        zone=_validate_hostname(_required_environment("CLOUDFLARE_ZONE")),
        hostname=_validate_hostname(_required_environment("CLOUDFLARE_HOSTNAME")),
        tunnel_name=_validate_name("CLOUDFLARE_TUNNEL_NAME", _required_environment("CLOUDFLARE_TUNNEL_NAME")),
        token_file=Path(_required_environment("CLOUDFLARE_TUNNEL_TOKEN_FILE")),
    )


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProvisionError(f"Cloudflare {description} response is invalid")
    return value


def _mapping_list(value: object, description: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ProvisionError(f"Cloudflare {description} response is invalid")
    return value


def _zone_and_account(client: CloudflareClient, settings: Settings) -> tuple[str, str]:
    zones = _mapping_list(
        client.request("GET", f"/zones?{urlencode({'name': settings.zone, 'status': 'active'})}"),
        "zones",
    )
    if len(zones) != 1:
        raise ProvisionError("Cloudflare zone must be uniquely active")
    zone = zones[0]
    if zone.get("name") != settings.zone or zone.get("status") != "active":
        raise ProvisionError("Cloudflare zone is not exact active ownership")
    account = _mapping(zone.get("account"), "zone account")
    zone_id = zone.get("id")
    account_id = account.get("id")
    if not isinstance(zone_id, str) or not isinstance(account_id, str):
        raise ProvisionError("Cloudflare zone ownership is invalid")
    return zone_id, account_id


def _state_matches(state: OwnershipState, settings: Settings, zone_id: str, account_id: str, tunnel_id: str) -> bool:
    return state == OwnershipState(
        account_id=account_id,
        zone_id=zone_id,
        tunnel_id=tunnel_id,
        zone=settings.zone,
        hostname=settings.hostname,
        tunnel_name=settings.tunnel_name,
    )


def _read_state() -> OwnershipState:
    try:
        metadata = STATE_FILE.lstat()
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisionError("local Cloudflare ownership state is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or not isinstance(payload, dict)
        or set(payload) != {"account_id", "zone_id", "tunnel_id", "zone", "hostname", "tunnel_name"}
    ):
        raise ProvisionError("local Cloudflare ownership state is invalid")
    typed_payload = cast(dict[str, object], payload)
    account_id = typed_payload.get("account_id")
    zone_id = typed_payload.get("zone_id")
    tunnel_id = typed_payload.get("tunnel_id")
    zone = typed_payload.get("zone")
    hostname = typed_payload.get("hostname")
    tunnel_name = typed_payload.get("tunnel_name")
    if not all(isinstance(value, str) for value in (account_id, zone_id, tunnel_id, zone, hostname, tunnel_name)):
        raise ProvisionError("local Cloudflare ownership state is invalid")
    account_id = cast(str, account_id)
    zone_id = cast(str, zone_id)
    tunnel_id = cast(str, tunnel_id)
    zone = cast(str, zone)
    hostname = cast(str, hostname)
    tunnel_name = cast(str, tunnel_name)
    if not _HEX_ID.fullmatch(account_id) or not _HEX_ID.fullmatch(zone_id):
        raise ProvisionError("local Cloudflare ownership state is invalid")
    try:
        if str(UUID(tunnel_id)) != tunnel_id:
            raise ValueError
        _validate_hostname(zone)
        _validate_hostname(hostname)
        _validate_name("tunnel_name", tunnel_name)
    except (ValueError, ProvisionError) as error:
        raise ProvisionError("local Cloudflare ownership state is invalid") from error
    return OwnershipState(account_id, zone_id, tunnel_id, zone, hostname, tunnel_name)


def _atomic_write(path: Path, content: bytes, *, mode: int, uid_gid: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        os.write(descriptor, content)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        if uid_gid is not None:
            os.fchown(descriptor, uid_gid, uid_gid)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    except OSError as error:
        raise ProvisionError("protected Cloudflare file cannot be written with required ownership") from error
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_state(state: OwnershipState) -> None:
    _atomic_write(STATE_FILE, (json.dumps(asdict(state), sort_keys=True) + "\n").encode("utf-8"), mode=0o600)


def _write_connector_token(token: object, path: Path) -> None:
    if not isinstance(token, str) or not token:
        raise ProvisionError("Cloudflare tunnel token response is invalid")
    _atomic_write(path, (token + "\n").encode("utf-8"), mode=0o400, uid_gid=CONNECTOR_UID_GID)


def _require_protected_connector_token(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ProvisionError("protected connector token is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_uid != CONNECTOR_UID_GID
        or metadata.st_gid != CONNECTOR_UID_GID
        or metadata.st_size == 0
    ):
        raise ProvisionError("connector token file protection is not exact")


def _expected_dns_record(settings: Settings, tunnel_id: str) -> dict[str, object]:
    return {
        "type": "CNAME",
        "name": settings.hostname,
        "content": f"{tunnel_id}.cfargotunnel.com",
        "proxied": True,
        "ttl": 1,
        "comment": DNS_COMMENT,
    }


def _exact_dns_record(record: dict[str, object], expected: dict[str, object]) -> bool:
    return all(record.get(key) == value for key, value in expected.items())


def _tunnel_connected(tunnel: dict[str, object]) -> bool:
    connections = tunnel.get("connections")
    return isinstance(connections, list) and any(
        isinstance(connection, dict) and connection.get("is_pending_reconnect") is False for connection in connections
    )


def _require_exact_tunnel(tunnel: dict[str, object], state: OwnershipState, settings: Settings) -> None:
    if (
        tunnel.get("id") != state.tunnel_id
        or tunnel.get("name") != settings.tunnel_name
        or tunnel.get("config_src") != "cloudflare"
    ):
        raise ProvisionError("Cloudflare tunnel identity or remote management drifted")


def _exact_tunnel_configuration(
    configuration: dict[str, object],
    expected: dict[str, object],
    tunnel_id: str,
) -> bool:
    if configuration == expected:
        return True
    if set(configuration) != {"config", "created_at", "source", "tunnel_id", "version"}:
        return False
    remote_config = configuration.get("config")
    expected_config = expected.get("config")
    config_matches = remote_config == expected_config
    if isinstance(remote_config, dict) and isinstance(expected_config, dict):
        config_matches = config_matches or (
            set(remote_config) == {"ingress", "warp-routing"}
            and remote_config.get("ingress") == expected_config.get("ingress")
            and remote_config.get("warp-routing") == {"enabled": False}
        )
    return (
        config_matches
        and configuration.get("source") == "cloudflare"
        and configuration.get("tunnel_id") == tunnel_id
        and isinstance(configuration.get("created_at"), str)
        and bool(configuration.get("created_at"))
        and isinstance(configuration.get("version"), int)
        and cast(int, configuration.get("version")) >= 1
    )


def prepare(settings: Settings, client: CloudflareClient | None, *, apply: bool) -> OwnershipState | None:
    if not apply:
        return None
    if client is None:
        raise ProvisionError("prepare --apply requires a Cloudflare client")
    if os.geteuid() != 0:
        raise ProvisionError("prepare --apply requires permission to set connector UID/GID")

    zone_id, account_id = _zone_and_account(client, settings)
    tunnel_query = urlencode(
        {"name": settings.tunnel_name, "is_deleted": "false", "per_page": 100}
    )
    tunnels = _mapping_list(
        client.request("GET", f"/accounts/{account_id}/cfd_tunnel?{tunnel_query}"),
        "tunnels",
    )
    matches = [tunnel for tunnel in tunnels if tunnel.get("name") == settings.tunnel_name]
    if len(matches) > 1:
        raise ProvisionError("Cloudflare tunnel ownership is ambiguous")

    created = not matches
    if created:
        create_payload = {"name": settings.tunnel_name, "config_src": "cloudflare"}
        tunnel = _mapping(
            client.request("POST", f"/accounts/{account_id}/cfd_tunnel", create_payload),
            "created tunnel",
        )
    else:
        tunnel = matches[0]
    tunnel_id = tunnel.get("id")
    if not isinstance(tunnel_id, str):
        raise ProvisionError("Cloudflare tunnel ID is invalid")
    if tunnel.get("name") != settings.tunnel_name or tunnel.get("config_src") != "cloudflare":
        raise ProvisionError("Cloudflare tunnel is not exact remotely managed configuration")

    state = OwnershipState(account_id, zone_id, tunnel_id, settings.zone, settings.hostname, settings.tunnel_name)
    if not created:
        existing_state = _read_state()
        if not _state_matches(existing_state, settings, zone_id, account_id, tunnel_id):
            raise ProvisionError("reused Cloudflare tunnel is not locally owned")

    configuration = _mapping(
        client.request("GET", f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations"),
        "tunnel configuration",
    )
    expected_configuration = settings.ingress_config()
    if created:
        client.request("PUT", f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations", expected_configuration)
    elif not _exact_tunnel_configuration(configuration, expected_configuration, tunnel_id):
        raise ProvisionError("reused Cloudflare tunnel configuration drifted")

    token = client.request("GET", f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token")
    _write_connector_token(token, settings.token_file)
    _write_state(state)
    return state


def publish(
    settings: Settings,
    client: CloudflareClient | None,
    *,
    apply: bool,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    if not apply:
        return False
    if client is None:
        raise ProvisionError("publish --apply requires a Cloudflare client")
    _require_protected_connector_token(settings.token_file)
    state = _read_state()
    if state.zone != settings.zone or state.hostname != settings.hostname or state.tunnel_name != settings.tunnel_name:
        raise ProvisionError("local Cloudflare ownership state does not match requested publish")
    tunnel = _mapping(
        client.request("GET", f"/accounts/{state.account_id}/cfd_tunnel/{state.tunnel_id}"),
        "tunnel",
    )
    _require_exact_tunnel(tunnel, state, settings)
    configuration = _mapping(
        client.request("GET", f"/accounts/{state.account_id}/cfd_tunnel/{state.tunnel_id}/configurations"),
        "tunnel configuration",
    )
    if not _exact_tunnel_configuration(configuration, settings.ingress_config(), state.tunnel_id):
        raise ProvisionError("Cloudflare tunnel configuration drifted")

    run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.cloudflare.yml",
            "up",
            "-d",
            "cloudflared",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = monotonic() + 30.0
    while True:
        tunnel = _mapping(
            client.request("GET", f"/accounts/{state.account_id}/cfd_tunnel/{state.tunnel_id}"),
            "tunnel",
        )
        if _tunnel_connected(tunnel):
            break
        if monotonic() >= deadline:
            raise ProvisionError("Cloudflare connector did not become healthy before deadline")
        sleep(1.0)

    dns_query = urlencode({"name.exact": settings.hostname, "type": "CNAME", "per_page": 100})
    records = _mapping_list(
        client.request("GET", f"/zones/{state.zone_id}/dns_records?{dns_query}"),
        "DNS records",
    )
    if len(records) > 1:
        raise ProvisionError("Cloudflare DNS ownership is ambiguous")
    expected_record = _expected_dns_record(settings, state.tunnel_id)
    if not records:
        client.request("POST", f"/zones/{state.zone_id}/dns_records", expected_record)
        return True
    if not _exact_dns_record(records[0], expected_record):
        raise ProvisionError("Cloudflare DNS record drifted or is not owned")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (commands.add_parser("prepare"), commands.add_parser("publish")):
        command.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    settings = load_settings()
    if not arguments.apply:
        print(f"Cloudflare {arguments.command} dry-run: no mutation")
        return 0

    client = HttpCloudflareClient(settings)
    try:
        if arguments.command == "prepare":
            prepare(settings, client, apply=True)
            print("Cloudflare tunnel prepared")
        else:
            publish(settings, client, apply=True)
            print("Cloudflare tunnel published")
    except subprocess.CalledProcessError as error:
        raise ProvisionError("cloudflared overlay start failed") from error
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvisionError as error:
        print(f"Cloudflare provisioning blocked: {error}")
        raise SystemExit(1) from None
