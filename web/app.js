let currentChatId = null;

const chatListEl = document.getElementById("chatList");
const threadEl = document.getElementById("thread");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const statusEl = document.getElementById("status");
const newChatBtn = document.getElementById("newChatBtn");
const sendBtn = document.getElementById("sendBtn");

function setStatus(text) {
  statusEl.textContent = text || "";
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "msg-role";
  label.textContent = role === "user" ? "You" : "Answer";

  const body = document.createElement("div");
  body.className = "msg-content";
  body.innerHTML = escapeHtml(content);

  msg.appendChild(label);
  msg.appendChild(body);
  threadEl.appendChild(msg);
}

function clearThread() {
  threadEl.innerHTML = "";
}

function scrollThreadToBottom() {
  threadEl.scrollTop = threadEl.scrollHeight;
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString();
}

async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `Request failed (${resp.status})`;
    throw new Error(detail);
  }
  return data;
}

async function loadChatList() {
  const data = await fetchJson("/api/chats");
  const chats = data.chats || [];

  chatListEl.innerHTML = "";
  if (chats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chat-meta";
    empty.textContent = "No chats yet";
    chatListEl.appendChild(empty);
    return;
  }

  for (const c of chats) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chat-item";
    btn.setAttribute("role", "listitem");
    btn.dataset.chatId = c.id;
    btn.setAttribute("aria-current", c.id === currentChatId ? "true" : "false");

    const title = document.createElement("div");
    title.className = "chat-title";
    title.textContent = c.title || "(untitled)";

    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = formatDate(c.created_at);

    btn.appendChild(title);
    btn.appendChild(meta);
    btn.addEventListener("click", () => openChat(c.id));
    chatListEl.appendChild(btn);
  }
}

async function openChat(chatId) {
  currentChatId = chatId;
  setStatus("");
  await loadChatList();

  const data = await fetchJson(`/api/chats/${encodeURIComponent(chatId)}`);
  const chat = data.chat;

  clearThread();
  const messages = (chat && chat.messages) || [];
  for (const m of messages) {
    if (!m || !m.role) continue;
    renderMessage(m.role, m.content || "");
  }
  scrollThreadToBottom();
}

function newChat() {
  currentChatId = null;
  clearThread();
  setStatus("");
  loadChatList();
  questionInput.focus();
}

async function sendQuestion(question) {
  setStatus("Thinking…");
  sendBtn.disabled = true;

  renderMessage("user", question);
  scrollThreadToBottom();

  try {
    const payload = {
      question,
      chat_id: currentChatId,
      top_k: 10,
    };
    const data = await fetchJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    currentChatId = data.chat_id;
    renderMessage("assistant", data.answer || "");
    scrollThreadToBottom();
    await loadChatList();
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e));
  } finally {
    sendBtn.disabled = false;
  }
}

newChatBtn.addEventListener("click", newChat);

askForm.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const q = (questionInput.value || "").trim();
  if (!q) return;
  questionInput.value = "";
  await sendQuestion(q);
});

(async function boot() {
  try {
    await loadChatList();
    newChat();
  } catch (e) {
    setStatus(e.message || String(e));
  }
})();
