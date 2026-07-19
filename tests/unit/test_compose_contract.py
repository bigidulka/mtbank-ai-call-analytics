from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
OPENWEBUI_DIGEST = "sha256:9fcea9c6e32ab60b0498f3986c6cdf651ddbe61db48d2213a3d28048ddd673d4"
PIPELINES_DIGEST = "sha256:b48e9bc338ce2be0acfbeff01810db72408a12f07739f9e3879c1f2b00952d6e"
NGINX_DIGEST = "sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10"
POSTGRES_DIGEST = "sha256:16bc17c64a573ef34162af9298258d1aec548232985b33ed7b1eac33ba35c229"
PYTHON_DIGEST = "sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca"
UV_DIGEST = "sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d"


def _service(compose: str, name: str) -> str:
    match = re.search(rf"^  {re.escape(name)}:\n", compose, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"service {name} отсутствует")
    next_service = re.search(r"^  [a-z][a-z0-9-]*:\n", compose[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_service.start() if next_service is not None else len(compose)
    return compose[match.end() : end]


def _service_networks(service: str) -> tuple[str, ...]:
    match = re.search(r"^    networks:\n((?:      - [a-z][a-z0-9-]*\n?)+)", service, flags=re.MULTILINE)
    if match is None:
        raise AssertionError("service networks отсутствуют")
    return tuple(line.removeprefix("      - ") for line in match.group(1).splitlines())


def test_compose_pins_artifacts_and_keeps_backend_services_private() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    preflight_service = _service(compose, "secrets-preflight")
    pipelines_service = _service(compose, "pipelines")
    openwebui_service = _service(compose, "openwebui")
    gateway_service = _service(compose, "gateway")
    bootstrap_service = _service(compose, "model-bootstrap")
    postgres_service = _service(compose, "postgres")
    migrate_service = _service(compose, "migrate")
    api_service = _service(compose, "api")
    speech_service = _service(compose, "speech")

    assert f"ghcr.io/open-webui/open-webui@{OPENWEBUI_DIGEST}" in compose
    assert f"ghcr.io/open-webui/pipelines@{PIPELINES_DIGEST}" in compose
    assert f"nginx@{NGINX_DIGEST}" in gateway_service
    assert '\n    expose:\n      - "9099"' in pipelines_service
    assert compose.count("\n    ports:\n") == 1
    assert '\n    ports:\n      - "127.0.0.1:${OPENWEBUI_PORT:-3000}:8080"\n    networks:' in gateway_service
    for private_service in (
        preflight_service,
        pipelines_service,
        openwebui_service,
        bootstrap_service,
        postgres_service,
        migrate_service,
        api_service,
        speech_service,
    ):
        assert "\n    ports:" not in private_service
    assert _service_networks(pipelines_service) == ("pipeline-internal", "application-internal")
    assert _service_networks(openwebui_service) == ("frontend", "pipeline-internal")
    assert _service_networks(gateway_service) == (
        "frontend",
        "gateway-ingress",
        "application-internal",
        "monitoring-internal",
    )
    assert _service_networks(bootstrap_service) == ("frontend",)
    assert _service_networks(postgres_service) == ("application-internal",)
    assert _service_networks(migrate_service) == ("application-internal",)
    assert _service_networks(speech_service) == ("application-internal", "speech-egress")
    assert _service_networks(api_service) == ("application-internal", "model-egress")
    assert compose.count("      - gateway-ingress") == 1
    assert compose.count("      - model-egress") == 1
    network_section = compose[compose.index("\nnetworks:\n") + 1 :]
    assert set(re.findall(r"^  ([a-z][a-z0-9-]*):$", network_section, flags=re.MULTILINE)) == {
        "application-internal",
        "frontend",
        "gateway-ingress",
        "model-egress",
        "monitoring-internal",
        "speech-egress",
        "pipeline-internal",
    }
    assert "  frontend:\n    internal: true" in network_section
    assert "  gateway-ingress:\n    internal: false" in network_section
    assert "  pipeline-internal:\n    internal: true" in network_section
    assert "  application-internal:\n    internal: true" in network_section
    assert "  model-egress:\n    internal: false" in network_section
    assert "  speech-egress:\n    internal: false" in network_section
    assert "mtbank_attachment_bridge.py" not in compose


def test_compose_uses_fail_closed_secret_preflight() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    preflight = _service(compose, "secrets-preflight")
    pipelines_service = _service(compose, "pipelines")
    openwebui_service = _service(compose, "openwebui")

    assert f"image: ghcr.io/open-webui/open-webui@{OPENWEBUI_DIGEST}" in preflight
    assert "validate_runtime_secrets.py" in preflight
    assert "network_mode: none" in preflight
    assert "condition: service_completed_successfully" in pipelines_service
    assert "condition: service_completed_successfully" in openwebui_service
    for name in (
        "WEBUI_ADMIN_PASSWORD",
        "WEBUI_SECRET_KEY",
        "PIPELINES_API_KEY",
        "MTBANK_ATTACHMENT_SIGNING_KEY",
        "MTBANK_API_KEY",
        "POSTGRES_PASSWORD",
        "GROQ_API_KEY",
    ):
        assert f"{name}: ${{{name}:?set {name} in .env}}" in preflight


def test_compose_authenticates_every_pipelines_route() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    pipelines_service = _service(compose, "pipelines")

    assert "auth_wrapper:app" in pipelines_service
    assert "--log-level warning --no-access-log" in pipelines_service
    assert 'GLOBAL_LOG_LEVEL: "WARNING"' in pipelines_service
    assert "os.environ['PIPELINES_API_KEY']" in pipelines_service
    assert "http://localhost:9099/v1/models" in pipelines_service


def test_compose_activates_pipeline_to_api_without_model_egress() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    pipelines_service = _service(compose, "pipelines")
    api_service = _service(compose, "api")
    speech_service = _service(compose, "speech")
    postgres_service = _service(compose, "postgres")

    assert "MTBANK_API_KEY: ${MTBANK_API_KEY:?set MTBANK_API_KEY in .env}" in pipelines_service
    assert 'MTBANK_PIPELINE_PROBE_MODE: "false"' in pipelines_service
    assert "NO_PROXY: localhost,127.0.0.1,::1,api,openwebui,pipelines,gateway" in pipelines_service
    assert _service_networks(pipelines_service) == ("pipeline-internal", "application-internal")
    assert "model-egress" not in pipelines_service
    assert _service_networks(api_service) == ("application-internal", "model-egress")
    assert _service_networks(speech_service) == ("application-internal", "speech-egress")
    assert "model-egress" not in speech_service
    assert "speech-egress" not in api_service
    assert "model-egress" not in postgres_service
    assert "speech-egress" not in postgres_service


def test_compose_disables_unneeded_openwebui_paths_and_sets_upload_limit() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    openwebui_service = _service(compose, "openwebui")

    assert 'ENABLE_TITLE_GENERATION: "False"' in openwebui_service
    assert 'ENABLE_FOLLOW_UP_GENERATION: "False"' in openwebui_service
    assert 'ENABLE_TAGS_GENERATION: "False"' in openwebui_service
    assert 'ENABLE_OLLAMA_API: "False"' in openwebui_service
    assert 'BYPASS_EMBEDDING_AND_RETRIEVAL: "True"' in openwebui_service
    assert 'RAG_EMBEDDING_ENGINE: "openai"' in openwebui_service
    assert 'ENABLE_PERSISTENT_CONFIG: "False"' in openwebui_service
    assert 'RAG_FILE_MAX_SIZE: "25"' in openwebui_service
    assert 'USER_PERMISSIONS_CHAT_WEB_UPLOAD: "False"' in openwebui_service
    assert 'USER_PERMISSIONS_CHAT_STT: "False"' in openwebui_service
    assert 'USER_PERMISSIONS_CHAT_TTS: "False"' in openwebui_service
    assert 'USER_PERMISSIONS_CHAT_CALL: "False"' in openwebui_service
    assert 'USER_PERMISSIONS_FEATURES_WEB_SEARCH: "False"' in openwebui_service
    assert 'ENABLE_WEB_SEARCH: "False"' in openwebui_service
    assert "openwebui_wrapper:app" in openwebui_service
    assert "--log-level\n      - warning\n      - --no-access-log" in openwebui_service
    assert 'GLOBAL_LOG_LEVEL: "WARNING"' in openwebui_service
    assert "./openwebui_wrapper.py:/app/backend/openwebui_wrapper.py:ro" in openwebui_service
    assert "PYTHONPATH: /app/src" in openwebui_service
    assert 'HTTP_PROXY: ""' in openwebui_service
    assert "NO_PROXY: localhost,127.0.0.1,::1,openwebui,pipelines,gateway" in openwebui_service
    assert "RAG_EMBEDDING_MODEL" not in openwebui_service
    assert "mem_limit:" in openwebui_service
    assert "pids_limit:" in openwebui_service
    assert "model-bootstrap:" in compose
    assert "model_capability_bootstrap.py" in compose
    assert "OPENWEBUI_BOOTSTRAP_URL: http://openwebui:8080" in compose
    assert "./src/mtbank_ai:/bootstrap/src/mtbank_ai:ro" in compose
    assert "PYTHONPATH: /bootstrap/src" in compose


def test_openwebui_wrapper_installs_fetch_adjacent_image_guard() -> None:
    wrapper = (ROOT / "openwebui_wrapper.py").read_text(encoding="utf-8")

    assert "from open_webui.utils import middleware as openwebui_middleware" in wrapper
    assert "install_remote_image_fetch_guard(openwebui_middleware)" in wrapper
    assert wrapper.index("install_remote_image_fetch_guard(openwebui_middleware)") < wrapper.index(
        "OpenWebUIPreBodyGuard(upstream_app)"
    )


def test_gateway_waits_for_one_shot_model_bootstrap() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    gateway_service = _service(compose, "gateway")
    bootstrap_service = _service(compose, "model-bootstrap")

    assert "model-bootstrap:\n        condition: service_completed_successfully" in gateway_service
    assert "entrypoint:\n      - python\n      - /bootstrap/model_capability_bootstrap.py" in bootstrap_service
    assert "command: []" in bootstrap_service
    assert "tail -f /dev/null" not in bootstrap_service
    assert "/tmp/model-bootstrap-ready" not in bootstrap_service
    assert "healthcheck:" not in bootstrap_service


def test_foundation_services_are_pinned_internal_and_migration_gated() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "app.Dockerfile").read_text(encoding="utf-8")
    postgres_service = _service(compose, "postgres")
    migrate_service = _service(compose, "migrate")
    api_service = _service(compose, "api")
    gateway_service = _service(compose, "gateway")

    assert compose.startswith("name: mtbank-ai-call-analytics\n")
    assert f"image: postgres@{POSTGRES_DIGEST}" in postgres_service
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}" in postgres_service
    assert "secrets-preflight:\n        condition: service_completed_successfully" in postgres_service
    assert "secrets-preflight:\n        condition: service_completed_successfully" in migrate_service
    assert "postgres:\n        condition: service_healthy" in migrate_service
    assert "command:\n      - alembic\n      - upgrade\n      - head" in migrate_service
    assert "secrets-preflight:\n        condition: service_completed_successfully" in api_service
    assert "postgres:\n        condition: service_healthy" in api_service
    assert "migrate:\n        condition: service_completed_successfully" in api_service
    assert "speech:\n        condition: service_healthy" in api_service
    assert "api:\n        condition: service_healthy" in _service(compose, "pipelines")

    database_environment = (
        "MTBANK_DATABASE__HOST: postgres",
        'MTBANK_DATABASE__PORT: "5432"',
        "MTBANK_DATABASE__NAME: ${POSTGRES_DB:-mtbank_ai}",
        "MTBANK_DATABASE__USER: ${POSTGRES_USER:-mtbank_ai}",
        "MTBANK_DATABASE__PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}",
        'MTBANK_DATABASE__POOL_SIZE: "5"',
        'MTBANK_DATABASE__MAX_OVERFLOW: "10"',
        'MTBANK_DATABASE__POOL_TIMEOUT_SECONDS: "2"',
        'MTBANK_DATABASE__CONNECT_TIMEOUT_SECONDS: "2"',
        'MTBANK_DATABASE__COMMAND_TIMEOUT_SECONDS: "2"',
        'MTBANK_DATABASE__READINESS_TIMEOUT_SECONDS: "3"',
    )
    for item in database_environment:
        assert item in migrate_service
        assert item in api_service

    assert "MTBANK_ENVIRONMENT" not in migrate_service
    assert "MTBANK_API__" not in migrate_service
    api_environment = (
        "MTBANK_ENVIRONMENT: ${MTBANK_ENVIRONMENT:-production}",
        "MTBANK_API__HOST: 0.0.0.0",
        'MTBANK_API__PORT: "8000"',
        "MTBANK_API__API_KEY: ${MTBANK_API_KEY:?set MTBANK_API_KEY in .env}",
        'MTBANK_API__MAX_JSON_BYTES: "1048576"',
        'MTBANK_API__MAX_UPLOAD_BYTES: "26214400"',
        'MTBANK_API__MULTIPART_RESERVE_BYTES: "65536"',
        'MTBANK_API__ALLOWED_MEDIA_TYPES: \'["audio/wav","audio/x-wav","audio/mpeg","audio/ogg"]\'',
        'MTBANK_API__ALLOWED_URL_SCHEMES: \'["http","https"]\'',
    )
    for item in api_environment:
        assert item in api_service

    for service in (migrate_service, api_service):
        assert "\n    read_only: true" in service
        assert "\n    tmpfs:" in service
        assert "mem_limit:" in service
        assert "pids_limit:" in service
    assert '\n    expose:\n      - "8000"' in api_service
    assert "http://127.0.0.1:8000/health/ready" in api_service
    assert _service_networks(gateway_service) == (
        "frontend",
        "gateway-ingress",
        "application-internal",
        "monitoring-internal",
    )
    assert f"FROM python@{PYTHON_DIGEST}" in dockerfile
    assert f"ghcr.io/astral-sh/uv:0.11.16@{UV_DIGEST}" in dockerfile
    assert "^uv 0\\.11\\.16" in dockerfile
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "uv export --frozen --no-dev --no-emit-project" in dockerfile
    assert "--no-hashes" not in dockerfile
    assert "uv pip install" in dockerfile and "--require-hashes" in dockerfile
    assert '"--ws-max-size", "98304", "--ws-max-queue", "1"' in dockerfile
    assert "postgres-data:" in compose


def test_api_projects_complete_fail_closed_analysis_runtime_configuration() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api_service = _service(compose, "api")

    def required_environment(name: str) -> str:
        return f"{name}: ${{{name}:?set {name} in .env}}"

    required_configuration = (
        required_environment("MTBANK_AGENT_RUNTIME__GATEWAY__BASE_URL"),
        required_environment("MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY"),
        required_environment("MTBANK_AGENT_RUNTIME__GATEWAY__MODELS__DEFAULT_MODEL"),
        "MTBANK_SPEECH__MODE: internal_http",
        "MTBANK_SPEECH__BASE_URL: http://speech:8010",
        "MTBANK_SPEECH__TRANSCRIPTION_PATH: /v1/transcribe",
        'MTBANK_SPEECH__TIMEOUT_SECONDS: "180"',
        required_environment("MTBANK_WORKFLOW__CODE_SHA"),
        'MTBANK_WORKFLOW__DEADLINE_SECONDS: "60"',
        'MTBANK_WORKFLOW__URL_TIMEOUT_SECONDS: "15"',
        'MTBANK_WORKFLOW__URL_MAX_REDIRECTS: "3"',
        "MTBANK_WORKFLOW__PRIVACY_MODE: redacted-cloud",
    )
    for configuration in required_configuration:
        assert configuration in api_service


def test_starlette_is_an_exact_direct_runtime_dependency_in_project_and_lock() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    runtime_dependencies = project["project"]["dependencies"]

    assert "starlette==1.3.1" in runtime_dependencies
    assert "uvicorn>=0.51,<1" in runtime_dependencies
    assert "python-multipart>=0.0.32,<1" in runtime_dependencies
    assert "starlette==1.3.1" not in project["project"]["optional-dependencies"]["test"]
    assert "starlette==1.3.1" not in project["dependency-groups"]["dev"]
    assert project["build-system"]["build-backend"] == "setuptools.build_meta"

    locked_project = next(package for package in lock["package"] if package["name"] == "mtbank-ai")
    assert "starlette" in {dependency["name"] for dependency in locked_project["dependencies"]}
    assert "starlette" not in {dependency["name"] for dependency in locked_project["optional-dependencies"]["test"]}
    assert "starlette" not in {dependency["name"] for dependency in locked_project["dev-dependencies"]["dev"]}
    starlette_metadata = next(
        dependency for dependency in locked_project["metadata"]["requires-dist"] if dependency["name"] == "starlette"
    )
    assert starlette_metadata["specifier"] == "==1.3.1"


def test_environment_example_requires_all_secrets_to_be_supplied() -> None:
    environment_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = {
        line.partition("=")[0]: line.partition("=")[2]
        for line in environment_example.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }

    assert "PIPELINES_PORT" not in values
    for name in (
        "WEBUI_ADMIN_PASSWORD",
        "WEBUI_SECRET_KEY",
        "PIPELINES_API_KEY",
        "MTBANK_ATTACHMENT_SIGNING_KEY",
        "MTBANK_API_KEY",
        "POSTGRES_PASSWORD",
    ):
        assert values[name] == ""
    assert values["MTBANK_ENVIRONMENT"] == "production"
    assert values["POSTGRES_DB"] == "mtbank_ai"
    assert values["POSTGRES_USER"] == "mtbank_ai"


def test_gitignore_ignores_local_environment_variants_but_keeps_example() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines
    assert lines.index(".env.*") < lines.index("!.env.example")
    assert "models/artifacts/" in lines
    assert "models/*" in lines
    assert "!models/manifest.json" in lines
    assert "!models/.gitkeep" in lines
    assert "release-evidence/" in lines


def test_compact_readme_links_to_the_api_error_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    assert "docs/api.md" in readme
    assert '{"error":' in api
    assert "404/405" in api
    assert "Allow" in api


def test_gateway_config_limits_body_and_keeps_streaming_proxy_settings() -> None:
    config = (ROOT / "gateway" / "nginx.conf").read_text(encoding="utf-8")
    server_start = config.index("server {")
    location_start = config.index("    location / {", server_start)
    server_directives = config[server_start:location_start]

    assert "    access_log off;" in server_directives
    assert "    error_log /dev/stderr crit;" in server_directives
    assert config.count("access_log off;") == 1
    assert config.count("error_log /dev/stderr crit;") == 1
    assert "error_log off;" not in config
    assert "client_max_body_size 26279936;" in config
    assert "proxy_request_buffering on;" in config
    assert "proxy_buffering off;" in config
    assert "proxy_http_version 1.1;" in config
    assert "proxy_read_timeout 300s;" in config
    assert "Upgrade $http_upgrade;" in config
    assert "Connection $connection_upgrade;" in config
    assert "X-Forwarded-Host $host;" in config
    assert "X-Forwarded-Port $server_port;" in config
    assert "map $http_x_forwarded_proto $forwarded_proto {" in config
    assert "    default $scheme;" in config
    assert "    ~^https$ https;" in config
    assert "~*^https$" not in config
    assert config.count("$http_x_forwarded_proto") == 1
    assert "X-Forwarded-Proto $scheme;" not in config
    assert config.count("X-Forwarded-Proto $forwarded_proto;") == 5
    assert "location = /analyze {" in config
    assert "proxy_pass http://api:8000/analyze;" in config
    assert "location = /trends {" in config
    assert "proxy_pass http://api:8000/trends;" in config
    assert "location = /ws/transcribe {" in config
    assert "proxy_pass http://api:8000/ws/transcribe;" in config
    assert "location ^~ /grafana/ {" in config
    assert "proxy_pass http://grafana:3000;" in config
    assert "X-Forwarded-Prefix /grafana;" in config
    assert "proxy_pass http://openwebui:8080;" in config


def test_verify_skill_checks_port_bindings_without_compose_port() -> None:
    skill = (ROOT / ".claude" / "skills" / "verify" / "SKILL.md").read_text(encoding="utf-8")

    assert "\n   docker compose port" not in skill
    assert ".HostConfig.PortBindings" in skill
    assert "openwebui_guard_e2e.py" in skill
    assert "openwebui_image_sink_contract.py" in skill
    assert "service_completed_successfully" in skill
    assert "assert_runtime_logs_clean.py" in skill
    assert "docker compose up -d --build --force-recreate --wait --wait-timeout 120" in skill
    assert "docker compose exec -T gateway nginx -T 2>&1" in skill
    assert (
        "OPENWEBUI_E2E_URL=http://127.0.0.1:3000 uv run --offline --no-sync python "
        "scripts/openwebui_attachment_e2e.py" in skill
    )
    assert "location = /analyze {" in skill
    assert "location = /ws/transcribe {" in skill
    assert "location ^~ /grafana/ {" in skill
    assert "grep -Fxc '    access_log off;'" in skill
    assert "grep -Fxc '    error_log /dev/stderr crit;'" in skill
    assert "docker compose logs --no-color --since 10m openwebui pipelines gateway" in skill
    assert "mtbank-ai-call-analytics_gateway-ingress" in skill
    assert "mtbank-ai-call-analytics_application-internal" in skill
    assert "mtbank-ai-call-analytics_monitoring-internal" in skill
    assert "docker network inspect -f '{{.Internal}}'" in skill
    assert ".NetworkSettings.Networks" in skill
    assert ".Containers" in skill
    assert "docker compose ps -q --all model-bootstrap" in skill
    assert "model-bootstrap` завершает работу" in skill
    assert "остаётся healthy" not in skill
    assert "test \"$(docker inspect -f '{{len .HostConfig.PortBindings}}'" in skill
    assert "controlled" in skill
    assert "native 404/405" in skill
    assert "Allow" in skill
    assert "migration" in skill
    assert "roundtrip" in skill
    assert "MTBANK_ALLOW_DESTRUCTIVE_DB_TESTS=1" in skill
    assert "не печат" in skill.lower()


def test_runtime_e2e_covers_privacy_and_processing_negatives() -> None:
    attachment_e2e = (ROOT / "scripts" / "openwebui_attachment_e2e.py").read_text(encoding="utf-8")
    guard_e2e = (ROOT / "scripts" / "openwebui_guard_e2e.py").read_text(encoding="utf-8")
    image_sink_contract = (ROOT / "scripts" / "openwebui_image_sink_contract.py").read_text(encoding="utf-8")

    for route in (
        "/api/v1/retrieval/process/file",
        "/api/v1/retrieval/process/files/batch",
        "/api/v1/retrieval/process/text",
        "/api/v1/files/attachment-id/data/content/update",
        "/api/v1/knowledge/knowledge-id/file/add",
        "/api/v1/knowledge/knowledge-id/file/update",
        "/api/v1/knowledge/knowledge-id/files/batch/add",
    ):
        assert route in guard_e2e
    assert "/api/v1/files/?process=true" in attachment_e2e
    assert '"/api/chat/completions", "/api/v1/chat/completions"' in attachment_e2e
    assert "user_message" in attachment_e2e
    assert "/api/v1/chats/new" in attachment_e2e
    assert "_CANONICAL_FIXTURE_ID = \"synthetic-card-complaint-telephone\"" in attachment_e2e
    assert "AnalyzeResponse.model_validate(payload)" in attachment_e2e
    assert "<pre>(.*?)</pre>" in attachment_e2e
    assert "production Pipeline output не должен echo authoritative filename" in attachment_e2e
    assert 'importlib.import_module("openwebui_wrapper")' in image_sink_contract
    assert "is_remote_image_fetch_guard_installed" in image_sink_contract
    assert "remote_form.messages_reads != 1" in image_sink_contract


def test_single_pipeline_adr_records_revisions_and_security_boundaries() -> None:
    adr = (ROOT / "docs" / "adr" / "0001-openwebui-attachment-pipeline.md").read_text(encoding="utf-8")

    assert "ecd48e2f718220a6400ecf49eafd4867a38feb10" in adr
    assert "039f9c54f8e9f9bcbabde02c2c853e80d25c79e4" in adr
    assert NGINX_DIGEST in adr
    assert "nginx/1.27.5" in adr
    assert "public read" in adr
    assert "file_context=false" in adr
    assert "GLOBAL_LOG_LEVEL=WARNING" in adr
    assert "Вложение недоступно" in adr
    assert "remote-image" in adr
    assert "convert_url_images_to_base64" in adr
    assert "до `validate_url`, DNS" in adr
    assert "INFO payload/auth context" in adr
    assert "mtbank_attachment_bridge" not in adr
    assert not (ROOT / "pipelines" / "mtbank_attachment_bridge.py").exists()


def test_monitoring_stack_is_pinned_private_and_dashboard_queries_emitted_metrics() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    prometheus = _service(compose, "prometheus")
    grafana = _service(compose, "grafana")
    collector = _service(compose, "otel-collector")
    tempo = _service(compose, "tempo")
    dashboard = (ROOT / "monitoring" / "grafana" / "dashboards" / "mtbank-overview.json").read_text(
        encoding="utf-8"
    )

    for service in (prometheus, grafana, collector, tempo):
        assert "@sha256:" in service
        assert "\n    ports:" not in service
    assert _service_networks(prometheus) == ("application-internal", "monitoring-internal")
    assert _service_networks(grafana) == ("monitoring-internal",)
    assert 'GF_SERVER_ROOT_URL: "%(protocol)s://%(domain)s/grafana/"' in grafana
    assert "%(http_port)s" not in grafana
    assert 'GF_SERVER_SERVE_FROM_SUB_PATH: "true"' in grafana
    assert "/var/lib/grafana/dashboards:ro" in grafana
    assert "healthcheck:" in grafana
    provider = (ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboard.yml").read_text(
        encoding="utf-8"
    )
    assert "path: /var/lib/grafana/dashboards" in provider
    assert "monitoring-internal:\n    internal: true" in compose
    for metric in (
        "mtbank_api_calls_total",
        "mtbank_quality_total",
        "mtbank_topic_calls_total",
        "mtbank_stage_latency_seconds",
        "mtbank_agent_tokens_total",
    ):
        assert metric in dashboard
