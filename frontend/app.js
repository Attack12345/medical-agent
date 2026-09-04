/* 智慧问诊Agent系统前端逻辑（DEV_DOC §9，M6.1 结构化卡片版）
 * - fetch 封装（JSON + Bearer token）；token 存内存变量（不做 localStorage）
 * - POST /chat/stream 用 fetch 流式读取（EventSource 不支持 POST）
 * - SSE 事件：intent / evidence（仅缓存）/ token（打字机）/ risk / done
 * - done 事件带 sections/tags → 结构化卡片渲染（对标黑马界面）
 * - esc 防注入：所有动态内容用 textContent 渲染
 */
"use strict";

const API = "/api/v1";

let token = null;
let userId = null;
let currentConvId = null;
let evidenceCache = [];   // 当前轮证据（默认不渲染，仅"查看依据"展开用）
let sectionsCache = [];
let tagsCache = {};
let sendStart = 0;

// ---------- 工具 ----------

function esc(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

async function api(path, method = "GET", body = null) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = "Bearer " + token;
  const resp = await fetch(API + path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.message || data.detail?.message || ("HTTP " + resp.status));
  return data;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  const flow = document.getElementById("message-flow");
  flow.scrollTop = flow.scrollHeight;
}

// ---------- 系统状态（/ready 探针） ----------

async function refreshStatus() {
  try {
    const data = await api("/ready");
    setStatus("st-neo4j", data.neo4j);
    setStatus("st-qdrant", data.qdrant);
    setStatus("st-agent", data.mysql && data.neo4j && data.qdrant);
  } catch (e) {
    setStatus("st-neo4j", false);
    setStatus("st-qdrant", false);
    setStatus("st-agent", false);
  }
}

function setStatus(id, ok) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = ok ? "正常" : "异常";
  node.className = "status-badge " + (ok ? "ok" : "fail");
}

// ---------- 登录 ----------

async function doLogin() {
  const u = document.getElementById("login-username").value.trim();
  const p = document.getElementById("login-password").value;
  const msg = document.getElementById("login-msg");
  msg.textContent = "";
  try {
    const data = await api("/auth/login", "POST", { username: u, password: p });
    token = data.token;
    userId = data.user_id;
    enterApp();
  } catch (e) {
    msg.textContent = e.message;
  }
}

async function doRegister() {
  const u = document.getElementById("login-username").value.trim();
  const p = document.getElementById("login-password").value;
  const msg = document.getElementById("login-msg");
  msg.textContent = "";
  try {
    await api("/auth/register", "POST", { username: u, password: p });
    msg.style.color = "#15803d";
    msg.textContent = "注册成功，请登录";
  } catch (e) {
    msg.textContent = e.message;
  }
}

function enterApp() {
  document.getElementById("login-panel").classList.add("hidden");
  document.getElementById("chat-panel").classList.remove("hidden");
  document.getElementById("user-info").textContent = "用户 #" + userId;
  refreshStatus();
  setInterval(refreshStatus, 30000);
  loadConversations();
  newConversation();
}

// ---------- 会话 ----------

async function loadConversations() {
  try {
    const data = await api("/conversations");
    const list = document.getElementById("conv-list");
    list.innerHTML = "";
    for (const c of data.conversations || []) {
      const li = el("li", c.id === currentConvId ? "active" : "", c.title || ("会话 #" + c.id));
      li.title = c.title || ("会话 #" + c.id);
      li.addEventListener("click", () => openConversation(c.id));
      list.appendChild(li);
    }
  } catch (e) { console.error(e); }
}

async function newConversation() {
  const data = await api("/conversations", "POST", { title: "新会话" });
  currentConvId = data.conversation_id;
  document.getElementById("message-flow").innerHTML = "";
  hideRiskBanner();
  loadConversations();
}

async function openConversation(convId) {
  currentConvId = convId;
  const flow = document.getElementById("message-flow");
  flow.innerHTML = "";
  hideRiskBanner();
  const data = await api("/conversations/" + convId + "/messages");
  for (const m of data.messages || []) {
    if (m.role === "USER") appendUser(m.content);
    else appendAssistantCard(m.content, m.evidence_json || {});
  }
  loadConversations();
}

// ---------- 消息渲染 ----------

function appendUser(text) {
  const flow = document.getElementById("message-flow");
  const row = el("div", "msg-row user");
  row.appendChild(el("div", "user-bubble", text));
  flow.appendChild(row);
  scrollToBottom();
}

function appendAssistantCard(content, evidence) {
  const flow = document.getElementById("message-flow");
  const row = el("div", "msg-row assistant");
  const card = el("div", "assistant-card");
  const title = el("div", "card-title", "智慧问诊AGENT系统 · 就医建议");
  card.appendChild(title);

  const answer = el("div", "card-answer", content || "");
  card.appendChild(answer);

  // 结构化小节（sections：title + points）
  const sections = (evidence && evidence.sections) || [];
  for (const sec of sections) {
    if (!sec || !sec.points || !sec.points.length) continue;
    const wrap = el("div", "card-section");
    wrap.appendChild(el("h4", null, sec.title || ""));
    const ul = el("ul");
    for (const pt of sec.points) ul.appendChild(el("li", null, pt));
    wrap.appendChild(ul);
    card.appendChild(wrap);
  }

  // 标签（tags：symptoms/departments）
  const tags = (evidence && evidence.tags) || {};
  const tagWrap = el("div", "card-tags");
  if (tags.symptoms && tags.symptoms.length) {
    const g = el("div", "tag-group", "症状:");
    for (const s of tags.symptoms) g.appendChild(el("span", "tag", s));
    tagWrap.appendChild(g);
  }
  if (tags.departments && tags.departments.length) {
    const g = el("div", "tag-group", "推荐科室:");
    for (const d of tags.departments) g.appendChild(el("span", "tag", d));
    tagWrap.appendChild(g);
  }
  if (tagWrap.children.length) card.appendChild(tagWrap);

  // 免责声明
  if (evidence && evidence.disclaimer_flag) {
    card.appendChild(el("div", "card-disclaimer", "重要声明: 本建议仅供参考，不能替代专业医疗诊断。请及时就医，遵医嘱用药!"));
  }

  // 元信息：耗时 + 复制
  const meta = el("div", "card-meta");
  meta.appendChild(el("span", null, (evidence && evidence.elapsed_ms) ? evidence.elapsed_ms + " ms" : ""));
  const copyBtn = el("button", "copy-btn", "复制");
  copyBtn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(content || ""); copyBtn.textContent = "已复制"; setTimeout(() => { copyBtn.textContent = "复制"; }, 1500); } catch (e) { /* 剪贴板不可用时静默 */ }
  });
  meta.appendChild(copyBtn);
  card.appendChild(meta);

  // 查看依据（可读来源折叠区）
  const pool = (evidence && evidence.evidence_pool) || [];
  if (pool.length) card.appendChild(evidenceToggle(pool));

  if (evidence && evidence.risk_level === "HIGH") showRiskBanner("⚠ 您描述的情况可能属于急症，请立即就医，必要时拨打120。");
  row.appendChild(card);
  flow.appendChild(row);
  scrollToBottom();
}

function evidenceToggle(pool) {
  const wrap = el("div");
  const toggle = el("div", "evidence-toggle", "查看依据 ▾");
  const box = el("div", "evidence-box");
  box.classList.add("hidden");
  for (const p of pool || []) {
    let readable;
    if (p.type === "GRAPH_NODE") {
      // 技术符号不外露：剥掉关系前缀（VISITS:/TREATS:），箭头改为顿号连接
      const parts = String(p.ref || "").replace(/^[A-Z_]+:/, "").split("→").map(s => s.trim()).filter(Boolean);
      readable = "涉及：" + parts.join("、") + "（知识图谱）";
    } else {
      readable = String(p.quote || "相关资料").slice(0, 20) + "（资料检索）";
    }
    box.appendChild(el("div", null, readable));
  }
  toggle.addEventListener("click", () => box.classList.toggle("hidden"));
  wrap.appendChild(toggle);
  wrap.appendChild(box);
  return wrap;
}

function showRiskBanner(text) {
  const banner = document.getElementById("risk-banner");
  banner.textContent = text;
  banner.classList.remove("hidden");
}

function hideRiskBanner() {
  document.getElementById("risk-banner").classList.add("hidden");
}

// ---------- SSE 流式问答（§7.2） ----------

async function sendQuestion() {
  const input = document.getElementById("question-input");
  const question = input.value.trim();
  if (!question || !currentConvId) return;
  input.value = "";
  appendUser(question);
  hideRiskBanner();

  // 助手占位（打字机）
  const flow = document.getElementById("message-flow");
  const row = el("div", "msg-row assistant");
  const card = el("div", "assistant-card");
  const title = el("div", "card-title", "智慧问诊AGENT系统 · 就医建议");
  const answer = el("div", "card-answer typing");
  card.appendChild(title);
  card.appendChild(answer);
  row.appendChild(card);
  flow.appendChild(row);
  scrollToBottom();

  evidenceCache = [];
  sectionsCache = [];
  tagsCache = {};
  sendStart = Date.now();
  const sendBtn = document.getElementById("btn-send");
  sendBtn.disabled = true;

  try {
    const resp = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
      body: JSON.stringify({ conversation_id: currentConvId, question }),
    });
    if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        let event = null;
        let data = null;
        for (const line of part.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7).trim();
          else if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
        }
        if (!event || !data) continue;
        handleSseEvent(event, data, answer, card);
      }
    }
  } catch (e) {
    answer.textContent = "请求失败：" + e.message;
  } finally {
    answer.classList.remove("typing");
    sendBtn.disabled = false;
    renderCardExtras(card, answer.textContent);
    loadConversations();  // 首问后侧栏标题同步更新（M8.7）
  }
}

function handleSseEvent(event, data, answer, card) {
  switch (event) {
    case "intent":
      console.log("[intent]", data.intent);
      break;
    case "evidence":
      evidenceCache = data.pool || [];
      break;
    case "token":
      answer.textContent += data.text;
      scrollToBottom();
      break;
    case "risk":
      if (data.risk_level === "HIGH") showRiskBanner("⚠ 您描述的情况可能属于急症，请立即就医，必要时拨打120。");
      break;
    case "done":
      sectionsCache = data.sections || [];
      tagsCache = data.tags || {};
      evidenceCache = (data.evidence && data.evidence.evidence_pool) || [];
      break;
    case "error":
      answer.textContent = "服务错误：" + (data.message || "未知错误");
      break;
  }
}

function renderCardExtras(card, answerText) {
  // 结构化小节
  for (const sec of sectionsCache) {
    if (!sec || !sec.points || !sec.points.length) continue;
    const wrap = el("div", "card-section");
    wrap.appendChild(el("h4", null, sec.title || ""));
    const ul = el("ul");
    for (const pt of sec.points) ul.appendChild(el("li", null, pt));
    wrap.appendChild(ul);
    card.appendChild(wrap);
  }
  // 标签
  const tagWrap = el("div", "card-tags");
  if (tagsCache.symptoms && tagsCache.symptoms.length) {
    const g = el("div", "tag-group", "症状:");
    for (const s of tagsCache.symptoms) g.appendChild(el("span", "tag", s));
    tagWrap.appendChild(g);
  }
  if (tagsCache.departments && tagsCache.departments.length) {
    const g = el("div", "tag-group", "推荐科室:");
    for (const d of tagsCache.departments) g.appendChild(el("span", "tag", d));
    tagWrap.appendChild(g);
  }
  if (tagWrap.children.length) card.appendChild(tagWrap);

  // 免责声明
  card.appendChild(el("div", "card-disclaimer", "重要声明: 本建议仅供参考，不能替代专业医疗诊断。请及时就医，遵医嘱用药!"));

  // 元信息：耗时 + 复制
  const meta = el("div", "card-meta");
  meta.appendChild(el("span", null, (Date.now() - sendStart) + " ms"));
  const copyBtn = el("button", "copy-btn", "复制");
  copyBtn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(answerText || ""); copyBtn.textContent = "已复制"; setTimeout(() => { copyBtn.textContent = "复制"; }, 1500); } catch (e) { /* 静默 */ }
  });
  meta.appendChild(copyBtn);
  card.appendChild(meta);

  // 查看依据
  if (evidenceCache.length) card.appendChild(evidenceToggle(evidenceCache));
}

// ---------- 事件绑定 ----------

document.getElementById("btn-login").addEventListener("click", doLogin);
document.getElementById("btn-register").addEventListener("click", doRegister);
document.getElementById("btn-new-conv").addEventListener("click", newConversation);
document.getElementById("btn-send").addEventListener("click", sendQuestion);
document.getElementById("question-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});
// "我能帮您"四卡片为功能展示，不绑定点击（M8.7）

// 页面加载即显示登录
document.getElementById("login-panel").classList.remove("hidden");
refreshStatus();
