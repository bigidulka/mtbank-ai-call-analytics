FROM python@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /usr/local/bin/

RUN uv --version | grep -Eq '^uv 0\.11\.16( |$)' \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY docker/wheelhouse /opt/wheelhouse
ARG USE_WHEELHOUSE=0
RUN if [ "$USE_WHEELHOUSE" = "1" ]; then \
        uv export --frozen --no-dev --no-emit-project --output-file /tmp/runtime-requirements.txt \
        && uv venv .venv \
        && uv pip install --python .venv/bin/python --no-index --find-links /opt/wheelhouse --require-hashes \
            --requirements /tmp/runtime-requirements.txt; \
    elif [ "$USE_WHEELHOUSE" = "0" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    else \
        echo "USE_WHEELHOUSE must be 0 or 1" >&2 \
        && exit 2; \
    fi \
    && rm -rf /opt/wheelhouse

COPY alembic.ini ./
COPY src ./src

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "mtbank_ai.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "98304", "--ws-max-queue", "1", "--log-level", "warning", "--no-access-log"]
