FROM python@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca AS python-runtime

FROM nvidia/cuda@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9

COPY --from=python-runtime /usr/local /usr/local

ENV PATH="/app/services/speech/.venv/bin:/usr/local/bin:${PATH}" \
    PYTHONPATH="/app/src:/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /usr/local/bin/

RUN python --version | grep -Eq '^Python 3\.11\.' \
    && uv --version | grep -Eq '^uv 0\.11\.16( |$)' \
    && if [ -f /etc/apt/sources.list ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; fi \
    && if [ -f /etc/apt/sources.list.d/debian.sources ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app/services/speech

COPY services/speech/pyproject.toml services/speech/uv.lock ./
COPY docker/wheelhouse /opt/wheelhouse
ARG USE_WHEELHOUSE=0
RUN if [ "$USE_WHEELHOUSE" = "1" ]; then \
        uv export --frozen --no-dev --no-emit-project --output-file /tmp/speech-requirements.txt \
        && uv venv .venv \
        && uv pip install --python .venv/bin/python --no-index --find-links /opt/wheelhouse --require-hashes \
            --requirements /tmp/speech-requirements.txt; \
    elif [ "$USE_WHEELHOUSE" = "0" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    else \
        echo "USE_WHEELHOUSE must be 0 or 1" >&2 \
        && exit 2; \
    fi \
    && rm -rf /opt/wheelhouse

COPY src /app/src
COPY services /app/services

USER 10001:10001

EXPOSE 8010

CMD ["uvicorn", "services.speech.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8010", "--workers", "1", "--ws-max-size", "65540", "--ws-max-queue", "1", "--log-level", "warning", "--no-access-log"]
