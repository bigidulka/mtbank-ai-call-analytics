from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
ZONE_ID = "a" * 32
ACCOUNT_ID = "b" * 32
TUNNEL_ID = "123e4567-e89b-12d3-a456-426614174000"


def _module() -> Any:
    path = ROOT / "scripts" / "provision_cloudflare_tunnel.py"
    spec = importlib.util.spec_from_file_location("provision_cloudflare_tunnel", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Cloudflare provisioner unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _settings(module: Any, tmp_path: Path) -> Any:
    return module.Settings(
        email="admin@example.test",
        global_key="global-key-secret",
        zone="cloud-tunnel-mega-obx1.space",
        hostname="cloud.cloud-tunnel-mega-obx1.space",
        tunnel_name="mtbank-ai-gateway",
        token_file=tmp_path / "cloudflared-token",
    )


def _state(module: Any, settings: Any) -> Any:
    return module.OwnershipState(
        account_id=ACCOUNT_ID,
        zone_id=ZONE_ID,
        tunnel_id=TUNNEL_ID,
        zone=settings.zone,
        hostname=settings.hostname,
        tunnel_name=settings.tunnel_name,
    )


class _Client:
    def __init__(self, module: Any, settings: Any, *, tunnels: list[dict[str, object]], configuration: object) -> None:
        self._module = module
        self._settings = settings
        self.tunnels = tunnels
        self.configuration = configuration
        self.tunnel = {"id": TUNNEL_ID, "name": settings.tunnel_name, "config_src": "cloudflare"}
        self.tunnel_connections: list[object] = []
        self.records: list[dict[str, object]] = []
        self.calls: list[tuple[str, str, object | None]] = []

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        self.calls.append((method, path, payload))
        if path.startswith("/zones?"):
            return [
                {
                    "id": ZONE_ID,
                    "name": self._settings.zone,
                    "status": "active",
                    "account": {"id": ACCOUNT_ID},
                }
            ]
        if path.startswith(f"/accounts/{ACCOUNT_ID}/cfd_tunnel?"):
            return self.tunnels
        if path == f"/accounts/{ACCOUNT_ID}/cfd_tunnel" and method == "POST":
            tunnel = {"id": TUNNEL_ID, "name": self._settings.tunnel_name, "config_src": "cloudflare"}
            self.tunnels.append(tunnel)
            return tunnel
        if path == f"/accounts/{ACCOUNT_ID}/cfd_tunnel/{TUNNEL_ID}/configurations":
            if method == "GET":
                return self.configuration
            self.configuration = payload
            return payload
        if path == f"/accounts/{ACCOUNT_ID}/cfd_tunnel/{TUNNEL_ID}/token":
            return "connector-token-secret"
        if path == f"/accounts/{ACCOUNT_ID}/cfd_tunnel/{TUNNEL_ID}":
            connections = self.tunnel_connections.pop(0) if self.tunnel_connections else []
            return {**self.tunnel, "connections": connections}
        if path.startswith(f"/zones/{ZONE_ID}/dns_records?"):
            return self.records
        if path == f"/zones/{ZONE_ID}/dns_records" and method == "POST":
            assert isinstance(payload, dict)
            self.records.append(payload)
            return payload
        raise AssertionError(f"unexpected Cloudflare request {method} {path}")


def test_dry_runs_are_mutation_free(tmp_path: Path) -> None:
    module = _module()
    settings = _settings(module, tmp_path)

    assert module.prepare(settings, None, apply=False) is None
    assert not module.publish(settings, None, apply=False)
    assert not settings.token_file.exists()


def test_prepare_creates_exact_remote_configuration_and_protected_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    chown_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(module.os, "fchown", lambda descriptor, uid, gid: chown_calls.append((uid, gid)))
    client = _Client(module, settings, tunnels=[], configuration={"config": {"ingress": []}})

    state = module.prepare(settings, client, apply=True)

    assert state == _state(module, settings)
    assert (
        "GET",
        f"/accounts/{ACCOUNT_ID}/cfd_tunnel?name={settings.tunnel_name}&is_deleted=false&per_page=100",
        None,
    ) in client.calls
    assert client.configuration == settings.ingress_config()
    assert '"originRequest": {}' in json.dumps(client.configuration)
    assert chown_calls == [(65532, 65532)]
    assert settings.token_file.read_text(encoding="utf-8") == "connector-token-secret\n"
    assert settings.token_file.stat().st_mode & 0o777 == 0o400
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["tunnel_id"] == TUNNEL_ID


def test_exact_tunnel_configuration_accepts_cloudflare_response_envelope(tmp_path: Path) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    expected = settings.ingress_config()
    wrapped = {
        "config": {**expected["config"], "warp-routing": {"enabled": False}},
        "created_at": "2026-07-18T22:24:56Z",
        "source": "cloudflare",
        "tunnel_id": TUNNEL_ID,
        "version": 1,
    }

    assert module._exact_tunnel_configuration(wrapped, expected, TUNNEL_ID) is True
    assert module._exact_tunnel_configuration({**wrapped, "tunnel_id": "other"}, expected, TUNNEL_ID) is False


def test_prepare_rejects_duplicate_or_reused_configuration_drift_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    tunnel = {"id": TUNNEL_ID, "name": settings.tunnel_name, "config_src": "cloudflare"}
    duplicate = _Client(module, settings, tunnels=[tunnel, tunnel.copy()], configuration=settings.ingress_config())

    with pytest.raises(module.ProvisionError, match="ambiguous"):
        module.prepare(settings, duplicate, apply=True)
    assert all(method != "PUT" for method, _, _ in duplicate.calls)

    state = _state(module, settings)
    module._write_state(state)
    monkeypatch.setattr(module, "_read_state", lambda: state)
    drifted = _Client(module, settings, tunnels=[tunnel], configuration={"config": {"ingress": []}})
    with pytest.raises(module.ProvisionError, match="drifted"):
        module.prepare(settings, drifted, apply=True)
    assert all(method != "PUT" for method, _, _ in drifted.calls)


def test_token_write_fails_closed_when_required_ownership_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    token_file = tmp_path / "cloudflared-token"

    def deny_chown(descriptor: int, uid: int, gid: int) -> None:
        del descriptor, uid, gid
        raise PermissionError("denied")

    monkeypatch.setattr(module.os, "fchown", deny_chown)
    with pytest.raises(module.ProvisionError, match="required ownership"):
        module._write_connector_token("connector-token-secret", token_file)
    assert not token_file.exists()


def test_read_state_rejects_unsafe_file_or_invalid_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(module, "STATE_FILE", state_file)
    module._write_state(_state(module, settings))

    state_file.chmod(0o644)
    with pytest.raises(module.ProvisionError, match="state is invalid"):
        module._read_state()

    state_file.chmod(0o600)
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["account_id"] = "not-a-cloudflare-id"
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.ProvisionError, match="state is invalid"):
        module._read_state()

    state_file.unlink()
    target = tmp_path / "state-target.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    target.chmod(0o600)
    state_file.symlink_to(target)
    with pytest.raises(module.ProvisionError, match="state is invalid"):
        module._read_state()


def test_publish_waits_for_connector_before_creating_exact_dns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "state.json")
    module._write_state(_state(module, settings))
    monkeypatch.setattr(module, "_require_protected_connector_token", lambda path: None)
    client = _Client(module, settings, tunnels=[], configuration=settings.ingress_config())
    client.tunnel_connections = [[], [{"is_pending_reconnect": False}]]
    command: list[object] = []
    clock = iter((0.0, 0.0, 1.0, 1.0))

    def run(argv: list[str], **kwargs: object) -> Any:
        command.extend(argv)
        assert kwargs["check"] is True
        return None

    assert module.publish(settings, client, apply=True, run=run, sleep=lambda _: None, monotonic=lambda: next(clock))

    dns_post = next(index for index, call in enumerate(client.calls) if call[0] == "POST" and "/dns_records" in call[1])
    connected_probe = max(index for index, call in enumerate(client.calls[:dns_post]) if call[1].endswith(TUNNEL_ID))
    assert dns_post > connected_probe
    assert command == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.cloudflare.yml",
        "up",
        "-d",
        "cloudflared",
    ]
    assert client.records == [module._expected_dns_record(settings, TUNNEL_ID)]
    assert (
        "GET",
        f"/zones/{ZONE_ID}/dns_records?name.exact={settings.hostname}&type=CNAME&per_page=100",
        None,
    ) in client.calls


def test_publish_rejects_remote_identity_or_config_drift_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "state.json")
    module._write_state(_state(module, settings))
    monkeypatch.setattr(module, "_require_protected_connector_token", lambda path: None)
    subprocess_calls: list[object] = []

    def run(*args: object, **kwargs: object) -> None:
        subprocess_calls.append((args, kwargs))

    identity_drift = _Client(module, settings, tunnels=[], configuration=settings.ingress_config())
    identity_drift.tunnel["name"] = "other-tunnel"
    with pytest.raises(module.ProvisionError, match="identity"):
        module.publish(settings, identity_drift, apply=True, run=run)
    assert not subprocess_calls
    assert all("/dns_records" not in path for _, path, _ in identity_drift.calls)

    configuration_drift = _Client(module, settings, tunnels=[], configuration={"config": {"ingress": []}})
    with pytest.raises(module.ProvisionError, match="configuration drifted"):
        module.publish(settings, configuration_drift, apply=True, run=run)
    assert not subprocess_calls
    assert all("/dns_records" not in path for _, path, _ in configuration_drift.calls)


def test_publish_rejects_dns_multiplicity_or_drift_without_takeover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    monkeypatch.setattr(module, "STATE_FILE", tmp_path / "state.json")
    module._write_state(_state(module, settings))
    monkeypatch.setattr(module, "_require_protected_connector_token", lambda path: None)
    client = _Client(module, settings, tunnels=[], configuration=settings.ingress_config())
    client.tunnel_connections = [[], [{"is_pending_reconnect": False}]]
    client.records = [{"id": "one"}, {"id": "two"}]

    with pytest.raises(module.ProvisionError, match="ambiguous"):
        module.publish(settings, client, apply=True, run=lambda *args, **kwargs: None, sleep=lambda _: None)
    assert all(method != "POST" for method, _, _ in client.calls)

    drifted = _Client(module, settings, tunnels=[], configuration=settings.ingress_config())
    drifted.tunnel_connections = [[], [{"is_pending_reconnect": False}]]
    drifted.records = [{"type": "CNAME", "name": settings.hostname, "content": "other.example.test"}]
    with pytest.raises(module.ProvisionError, match="drifted"):
        module.publish(settings, drifted, apply=True, run=lambda *args, **kwargs: None, sleep=lambda _: None)
    assert all(method != "POST" for method, _, _ in drifted.calls)


def test_http_client_uses_bounded_direct_global_key_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    settings = _settings(module, tmp_path)
    captured: dict[str, object] = {}

    class CapturingHttpClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(module.httpx, "Client", CapturingHttpClient)
    client = module.HttpCloudflareClient(settings)
    client.close()

    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["X-Auth-Email"] == settings.email
    assert headers["X-Auth-Key"] == settings.global_key
    source = (ROOT / "scripts" / "provision_cloudflare_tunnel.py").read_text(encoding="utf-8")
    assert "response.text" not in source
    assert "response.content" not in source
    assert '"CF_EMAIL"' in source
    assert '"CF_GLOBAL_API_KEY"' in source
    assert "CLOUDFLARE_EMAIL" not in source
    publish_source = source[source.index("def publish(") : source.index("def main()")]
    assert '"PUT"' not in publish_source and '"PATCH"' not in publish_source
    assert "global-key-secret" not in source


def test_cloudflared_overlay_isolated_and_documents_controlled_rollback() -> None:
    compose = (ROOT / "docker-compose.cloudflare.yml").read_text(encoding="utf-8")
    operations = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")

    assert (
        "cloudflare/cloudflared:2026.7.1@"
        "sha256:b4d7b15b9e9256ee3c9d243a49cec440aa6d7acbf6b2e2c1086ebfff939ec48d" in compose
    )
    assert "platform: linux/amd64" in compose
    assert "gateway:\n        condition: service_healthy" in compose
    assert "- tunnel\n      - run\n      - --token-file" in compose
    assert "/run/secrets/cloudflare_tunnel_token" in compose
    assert "CLOUDFLARE_TUNNEL_TOKEN=" not in compose
    assert "environment:" not in compose
    assert "ports:" not in compose and "expose:" not in compose and "healthcheck:" not in compose
    assert compose.count("- gateway-ingress") == 1
    assert "restart: unless-stopped" in compose
    assert "remove the owned DNS record first, stop the `cloudflared`" in operations
    assert "optionally delete the dedicated tunnel last" in operations
    assert "Actual external side effects were not run" in operations
    assert "CF_EMAIL" in operations and "CF_GLOBAL_API_KEY" in operations
    assert "CLOUDFLARE_EMAIL" not in operations
