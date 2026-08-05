# Experimental profile only: unofficial ChatGPT Web transcription bridge.
# Never publish this daemon's port to a public interface; it must stay
# reachable only from sibling containers on an internal Docker network, or
# from an operator's own machine through an SSH tunnel for one-time pairing.
FROM rust@sha256:9a7159329166b45f453351a077367f501aa3e98378f7e327530e7966a139d05f AS builder

WORKDIR /build
RUN if [ -f /etc/apt/sources.list ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; fi \
    && if [ -f /etc/apt/sources.list.d/debian.sources ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install --no-install-recommends --yes pkg-config libdbus-1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY services/chatgpt-bridge/Cargo.toml services/chatgpt-bridge/Cargo.lock ./
COPY services/chatgpt-bridge/src ./src
RUN cargo build --release --locked

FROM python@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca

RUN if [ -f /etc/apt/sources.list ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; fi \
    && if [ -f /etc/apt/sources.list.d/debian.sources ]; then sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install --no-install-recommends --yes ffmpeg ca-certificates libdbus-1-3 libsystemd0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

COPY --from=builder /build/target/release/chatgpt-transcribe-connect /usr/local/bin/chatgpt-transcribe-connect

ENV HOME=/home/app \
    XDG_CONFIG_HOME=/home/app/.config

USER 10001:10001
WORKDIR /home/app
EXPOSE 37182
ENTRYPOINT ["chatgpt-transcribe-connect"]
CMD ["--listen", "0.0.0.0:37182", "serve"]
