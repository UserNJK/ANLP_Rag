/* ─── State ─── */
let currentChatId = null;

/* ─── DOM refs ─── */
const chatListEl     = document.getElementById("chatList");
const threadEl       = document.getElementById("thread");
const askForm        = document.getElementById("askForm");
const questionInput  = document.getElementById("questionInput");
const statusEl       = document.getElementById("status");
const newChatBtn     = document.getElementById("newChatBtn");
const sendBtn        = document.getElementById("sendBtn");
const welcomeEl      = document.getElementById("welcome");
const typingEl       = document.getElementById("typing");
const sidebarEl      = document.getElementById("sidebar");
const sidebarToggle  = document.getElementById("sidebarToggle");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");

/* ─── Helpers ─── */
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

function formatMetric(value) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return "—";
  return n.toFixed(3);
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
  });
}

/* ─── Welcome / Thread visibility ─── */
function showWelcome() {
  welcomeEl.classList.remove("hidden");
  threadEl.classList.remove("active");
}

function showThread() {
  welcomeEl.classList.add("hidden");
  threadEl.classList.add("active");
}

/* ─── Typing indicator ─── */
function showTyping() {
  typingEl.hidden = false;
  scrollThreadToBottom();
}

function hideTyping() {
  typingEl.hidden = true;
}

/* ─── Mobile sidebar ─── */
function openSidebar() {
  sidebarEl.classList.add("open");
  sidebarBackdrop.hidden = false;
}

function closeSidebar() {
  sidebarEl.classList.remove("open");
  sidebarBackdrop.hidden = true;
}

if (sidebarToggle)  sidebarToggle.addEventListener("click", openSidebar);
if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebar);

/* ─── Message rendering (ChatGPT-style full-width rows) ─── */
function renderMessage(role, content, metrics) {
  showThread();

  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const inner = document.createElement("div");
  inner.className = "msg-inner";

  // Avatar
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "You" : "Bot";

  // Body column
  const body = document.createElement("div");
  body.className = "msg-body";

  const roleLabel = document.createElement("div");
  roleLabel.className = "msg-role";
  roleLabel.textContent = role === "user" ? "You" : "Bot";

  body.appendChild(roleLabel);

  if (role === "user") {
    const contentEl = document.createElement("div");
    contentEl.className = "msg-content";
    contentEl.innerHTML = escapeHtml(typeof content === "string" ? content : "");
    body.appendChild(contentEl);
  }

  inner.appendChild(avatar);
  inner.appendChild(body);
  msg.appendChild(inner);
  threadEl.appendChild(msg);
}

/* ─── Render dual-section assistant message ─── */
function renderDualMessage(ragAnswer, nonRagAnswer, ragMetrics) {
  showThread();

  const msg = document.createElement("div");
  msg.className = "msg assistant";

  const inner = document.createElement("div");
  inner.className = "msg-inner";

  // Avatar
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "Bot";

  // Body column
  const body = document.createElement("div");
  body.className = "msg-body";

  const roleLabel = document.createElement("div");
  roleLabel.className = "msg-role";
  roleLabel.textContent = "Bot";
  body.appendChild(roleLabel);

  // --- RAG section ---
  const ragSection = document.createElement("div");
  ragSection.className = "answer-section rag-section";

  const ragHeader = document.createElement("div");
  ragHeader.className = "section-header rag-header";
  ragHeader.textContent = "RAG";
  ragSection.appendChild(ragHeader);

  const ragContent = document.createElement("div");
  ragContent.className = "msg-content";
  ragContent.innerHTML = escapeHtml(ragAnswer || "");
  ragSection.appendChild(ragContent);

  // RAG metrics
  if (ragMetrics) {
    const metricsEl = document.createElement("div");
    metricsEl.className = "msg-metrics";

    const pairs = [
      { key: "rouge_l",            label: "ROUGE-L" },
      { key: "semantic_score",     label: "Semantic" },
      { key: "hallucination_ratio", label: "Hallucination" },
    ];

    for (const p of pairs) {
      const badge = document.createElement("span");
      badge.className = "metric-badge";
      badge.innerHTML = `<span class="metric-label">${p.label}</span> ${formatMetric(ragMetrics[p.key])}`;
      metricsEl.appendChild(badge);
    }

    ragSection.appendChild(metricsEl);
  }

  body.appendChild(ragSection);

  // --- Non-RAG section ---
  const nonRagSection = document.createElement("div");
  nonRagSection.className = "answer-section non-rag-section";

  const nonRagHeader = document.createElement("div");
  nonRagHeader.className = "section-header non-rag-header";
  nonRagHeader.textContent = "Non RAG";
  nonRagSection.appendChild(nonRagHeader);

  const nonRagContent = document.createElement("div");
  nonRagContent.className = "msg-content";
  nonRagContent.innerHTML = escapeHtml(nonRagAnswer || "");
  nonRagSection.appendChild(nonRagContent);

  body.appendChild(nonRagSection);

  inner.appendChild(avatar);
  inner.appendChild(body);
  msg.appendChild(inner);
  threadEl.appendChild(msg);
}

function clearThread() {
  threadEl.innerHTML = "";
}

function scrollThreadToBottom() {
  requestAnimationFrame(() => {
    threadEl.scrollTop = threadEl.scrollHeight;
  });
}

/* ─── API ─── */
async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `Request failed (${resp.status})`;
    throw new Error(detail);
  }
  return data;
}

/* ─── Chat list ─── */
async function loadChatList() {
  const data = await fetchJson("/api/chats");
  const chats = data.chats || [];

  chatListEl.innerHTML = "";

  if (chats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chat-list-empty";
    empty.textContent = "No conversations yet";
    chatListEl.appendChild(empty);
    return;
  }

  for (const c of chats) {
    const row = document.createElement("div");
    row.className = "chat-row";

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
    btn.addEventListener("click", () => {
      closeSidebar();
      openChat(c.id);
    });

    const del = document.createElement("button");
    del.type = "button";
    del.className = "chat-delete";
    del.setAttribute("aria-label", "Delete chat");
    del.addEventListener("click", async (evt) => {
      evt.stopPropagation();
      await deleteChat(c.id);
    });

    row.appendChild(btn);
    row.appendChild(del);
    chatListEl.appendChild(row);
  }
}

/* ─── Chat operations ─── */
async function openChat(chatId) {
  currentChatId = chatId;
  setStatus("");
  await loadChatList();

  const data = await fetchJson(`/api/chats/${encodeURIComponent(chatId)}`);
  const chat = data.chat;

  clearThread();
  const messages = (chat && chat.messages) || [];

  if (messages.length === 0) {
    showWelcome();
  } else {
    showThread();
    for (const m of messages) {
      if (!m || !m.role) continue;
      if (m.role === "assistant" && (m.rag_answer || m.non_rag_answer)) {
        renderDualMessage(m.rag_answer || m.content || "", m.non_rag_answer || "", m.rag_metrics || m.metrics);
      } else {
        renderMessage(m.role, m.content || "", m.metrics);
      }
    }
  }

  scrollThreadToBottom();
}

async function deleteChat(chatId) {
  const ok = window.confirm("Delete this chat?");
  if (!ok) return;

  try {
    await fetchJson(`/api/chats/${encodeURIComponent(chatId)}`, { method: "DELETE" });

    if (currentChatId === chatId) {
      currentChatId = null;
      clearThread();
      showWelcome();
    }

    await loadChatList();
    setStatus("");
  } catch (e) {
    setStatus(e.message || String(e));
  }
}

function newChat() {
  currentChatId = null;
  clearThread();
  showWelcome();
  setStatus("");
  loadChatList();
  questionInput.focus();
}

/* ─── Send question ─── */
async function sendQuestion(question) {
  setStatus("");
  sendBtn.disabled = true;

  renderMessage("user", question);
  scrollThreadToBottom();
  showTyping();

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

    hideTyping();
    currentChatId = data.chat_id;
    renderDualMessage(data.rag_answer || "", data.non_rag_answer || "", data.rag_metrics);
    scrollThreadToBottom();
    await loadChatList();
  } catch (e) {
    hideTyping();
    setStatus(e.message || String(e));
  } finally {
    sendBtn.disabled = false;
  }
}

/* ─── Suggestion chips ─── */
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.querySelector(".chip-text");
    const q = (text ? text.textContent : chip.textContent).trim();
    if (!q) return;
    questionInput.value = "";
    sendQuestion(q);
  });
});

/* ─── Event listeners ─── */
newChatBtn.addEventListener("click", newChat);

askForm.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const q = (questionInput.value || "").trim();
  if (!q) return;
  questionInput.value = "";
  await sendQuestion(q);
});

/* ─── Boot ─── */
(async function boot() {
  try {
    await loadChatList();
    newChat();
  } catch (e) {
    setStatus(e.message || String(e));
  }
})();
