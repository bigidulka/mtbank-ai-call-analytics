#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release --locked
python3 -m json.tool extension/manifest.json >/dev/null
node --check extension/background.js
node --check extension/popup.js

if rg -n --hidden -g '!target/**' -g '!release-verify*/**' -g '!Cargo.lock' -g '!scripts/verify.sh' \
  '(eyJhbGci|__Secure-[^" ]+=[^" ]+|Bearer ey|gho_[A-Za-z0-9]|cfk_)' .; then
  echo 'Potential credential found in tracked project files' >&2
  exit 1
fi

test -z "$(git status --porcelain --untracked-files=no)"

verify_dir=target/release-verification
rm -rf "$verify_dir"
mkdir -p "$verify_dir"
gh release download v0.1.0 -R bigidulka/chatgpt-transcribe-connect -D "$verify_dir" --clobber
(
  cd "$verify_dir"
  sha256sum -c SHA256SUMS
  mkdir linux macos-x64 macos-arm64 windows extension
  tar -xzf linux-x86_64.tar.gz -C linux
  tar -xzf macos-x86_64.tar.gz -C macos-x64
  tar -xzf macos-aarch64.tar.gz -C macos-arm64
  unzip -q windows-x86_64.zip -d windows
  unzip -q chatgpt-transcribe-connector-extension.zip -d extension
  ./linux/chatgpt-transcribe-connect --version | grep -F '0.1.0'
  file macos-x64/chatgpt-transcribe-connect | grep -F 'Mach-O 64-bit x86_64'
  file macos-arm64/chatgpt-transcribe-connect | grep -F 'Mach-O 64-bit arm64'
  file windows/chatgpt-transcribe-connect.exe | grep -F 'PE32+ executable'
  python3 -m json.tool extension/manifest.json >/dev/null
  node --check extension/background.js
  node --check extension/popup.js
)

echo 'VERIFICATION_OK'
