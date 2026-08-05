const DAEMON = "http://127.0.0.1:37182";

const SESSION_COOKIE_PREFIXES = [
  "__Secure-next-auth.session-token",
  "__Secure-authjs.session-token",
  "next-auth.session-token",
  "authjs.session-token"
];

function isSessionCookie(name) {
  return SESSION_COOKIE_PREFIXES.some((prefix) => name === prefix || name.startsWith(`${prefix}.`));
}

async function chatGptCookies() {
  // Brave may expose host-only and partitioned cookies differently. Query both
  // concrete origins and domain, then deduplicate without logging values.
  const groups = await Promise.all([
    chrome.cookies.getAll({ url: "https://chatgpt.com/" }),
    chrome.cookies.getAll({ url: "https://www.chatgpt.com/" }),
    chrome.cookies.getAll({ domain: "chatgpt.com" })
  ]);
  const cookies = groups.flat();
  const values = Object.fromEntries(cookies.map(({ name, value }) => [name, value]));
  if (!Object.keys(values).some(isSessionCookie)) {
    const visibleNames = [...new Set(cookies.map(({ name }) => name))].sort();
    throw new Error(
      `ChatGPT session cookie not visible to extension. Visible ChatGPT cookies: ${visibleNames.join(", ") || "none"}. Reload chatgpt.com and check Brave Shields.`
    );
  }
  return values;
}

async function chatGptSession() {
  const response = await fetch("https://chatgpt.com/api/auth/session", {
    credentials: "include",
    cache: "no-store"
  });
  if (!response.ok) throw new Error(`ChatGPT session request failed: HTTP ${response.status}`);
  const session = await response.json();
  if (!session.accessToken) throw new Error("ChatGPT access token missing. Sign in again.");
  return session;
}

async function connect(pairingCode) {
  const [cookies, session] = await Promise.all([chatGptCookies(), chatGptSession()]);
  const response = await fetch(`${DAEMON}/internal/pair`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pairing_code: pairingCode.trim(),
      access_token: session.accessToken,
      cookies,
      expires_at: session.expires ?? null
    })
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result?.error?.message ?? `Daemon rejected pairing: HTTP ${response.status}`);
  return result;
}

async function daemonStatus() {
  const response = await fetch(`${DAEMON}/v1/status`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Daemon status failed: HTTP ${response.status}`);
  return response.json();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const operation = message.type === "connect" ? connect(message.pairingCode) : daemonStatus();
  operation.then((value) => sendResponse({ ok: true, value }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
