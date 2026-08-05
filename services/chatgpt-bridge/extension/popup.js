const daemon = document.querySelector("#daemon");
const dot = document.querySelector("#dot");
const code = document.querySelector("#code");
const button = document.querySelector("#connect");
const status = document.querySelector("#status");

function message(type, text) {
  status.className = `status ${type}`;
  status.textContent = text;
}

async function send(payload) {
  return chrome.runtime.sendMessage(payload);
}

async function refresh() {
  const response = await send({ type: "status" });
  if (!response?.ok) {
    daemon.textContent = "Daemon unavailable on port 37182";
    dot.className = "dot bad";
    message("error", "Start daemon, then reopen this popup.");
    return;
  }
  daemon.textContent = response.value.connected ? "Daemon online · ChatGPT connected" : "Daemon online · not connected";
  dot.className = "dot ok";
}

button.addEventListener("click", async () => {
  if (!code.value.trim()) return message("error", "Enter pairing code.");
  button.disabled = true;
  message("", "Connecting…");
  const response = await send({ type: "connect", pairingCode: code.value });
  button.disabled = false;
  if (!response?.ok) return message("error", response?.error ?? "Connection failed.");
  code.value = "";
  message("success", "Connected. Credentials stored by local daemon.");
  await refresh();
});

refresh().catch((error) => message("error", error.message));
