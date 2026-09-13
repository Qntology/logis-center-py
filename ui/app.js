let currentSchema = null;
let lastResults = null;
let baseModelsReady = false;
let modelsLoaded = false;

function waitForApi(callback) {
  if (window.pywebview && window.pywebview.api) {
    callback();
  } else {
    setTimeout(function () { waitForApi(callback); }, 300);
  }
}

function onJobState(info) {
  const btn = document.querySelector('#btn-run');
  if (!btn) return;
  if (info.phase === 'wait') {
    btn.disabled = true;
    btn.dataset.label = btn.dataset.label || btn.textContent;
    btn.textContent = '대기 중… (' + info.waiting + '건)';
  } else if (info.phase === 'start') {
    btn.disabled = true;
    btn.dataset.label = btn.dataset.label || btn.textContent;
    btn.textContent = info.title + ' 진행 중…';
  } else {
    btn.disabled = false;
    if (btn.dataset.label) btn.textContent = btn.dataset.label;
  }
}

let modelBusy = false;
let lastStatus = null;

function fmtBytes(n) {
  const v = Number(n || 0);
  if (v >= 1024 * 1024 * 1024) return (v / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  if (v >= 1024 * 1024) return (v / (1024 * 1024)).toFixed(1) + " MB";
  if (v > 0) return (v / 1024).toFixed(0) + " KB";
  return "-";
}

function esc(s) {
  return String(s == null ? "" : s).replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function modelRow(info) {
  const ready = !!info.ready;
  const manual = !!info.manual_only;
  const active = lastStatus && lastStatus.current === info.id ? " active" : "";

  let stateCls, stateTxt;
  if (ready) {
    stateCls = "ready";
    stateTxt = "✅ 준비됨";
  } else if (manual) {
    stateCls = "manual";
    stateTxt = "➖ 선택";
  } else {
    stateCls = "missing";
    stateTxt = "❌ 누락";
  }

  let sub = info.repo || info.dir || "-";
  if (manual && info.note) {
    sub = info.note;
  } else if (!ready && info.missing && info.missing.length) {
    sub += " · 누락: " + info.missing.slice(0, 3).join(", ") +
      (info.missing.length > 3 ? " 외 " + (info.missing.length - 3) : "");
  }

  const canGet = !!info.repo && !manual && !modelBusy && !ready;
  const canDel = !modelBusy && info.bytes > 0;

  return (
    '<div class="model-row' + active + (manual ? " manual" : "") +
    '" data-id="' + esc(info.id) + '" title="' + esc(info.dir || "") + '">' +
    '<div class="m-main">' +
    '<span class="m-name">' + esc(info.label) + "</span>" +
    '<span class="m-sub">' + esc(sub) + "</span>" +
    "</div>" +
    '<span class="m-size">' + fmtBytes(info.bytes) + "</span>" +
    '<span class="m-state ' + stateCls + '">' + stateTxt + "</span>" +
    '<button class="get" onclick="downloadModel(\'' + info.id + '\')"' +
    (canGet ? "" : " disabled") + ">" + (manual ? "수동" : "받기") + "</button>" +
    '<button class="del" onclick="deleteModel(\'' + info.id + '\')"' +
    (canDel ? "" : " disabled") + ">삭제</button>" +
    "</div>"
  );
}

async function refreshModelStatus(openIfMissing) {
  if (!window.pywebview || !window.pywebview.api) return null;
  try {
    const status = await window.pywebview.api.check_models();
    lastStatus = status;
    modelBusy = !!status.busy;

    if (status.models_ready) markModelsLoaded();

    const code = status.lang_code || "";
    updateLangBadge(
      code, status.lang_name,
      status.language_resolved, status.bootstrap_languages
    );
    updateCrossover(status.crossover);

    const scopeLabel = document.getElementById("lang-scope-label");
    if (scopeLabel) {
      if (status.language_resolved) {
        scopeLabel.textContent = "(" + code + " / " + (status.lang_name || code) + ")";
      } else {
        const boot = (status.bootstrap_languages || []).join(", ");
        scopeLabel.textContent = "(미확정 → 기본 " + boot + ")";
      }
    }

    const bootEl = document.getElementById("model-list-bootstrap");
    const bootSec = document.getElementById("bootstrap-section");
    const boots = status.bootstrap || [];
    if (bootEl && bootSec) {
      if (boots.length) {
        bootSec.style.display = "";
        bootEl.style.display = "";
        bootEl.innerHTML = boots.map(modelRow).join("");
      } else {
        bootSec.style.display = "none";
        bootEl.style.display = "none";
      }
    }

    const total = document.getElementById("settings-total");
    if (total) total.textContent = "총 " + fmtBytes(status.total_bytes);

    const baseEl = document.getElementById("model-list-base");
    if (baseEl) baseEl.innerHTML = (status.base || []).map(modelRow).join("");

    const langEl = document.getElementById("model-list-lang");
    if (langEl) langEl.innerHTML = (status.lang || []).map(modelRow).join("");

    const stEl = document.getElementById("model-list-stanza");
    if (stEl) stEl.innerHTML = status.stanza ? modelRow(status.stanza) : "";

    const autoEl = document.getElementById("auto-fetch-toggle");
    if (autoEl) autoEl.checked = !!status.auto_fetch;

    const btnCancel = document.getElementById("btn-cancel-dl");
    if (btnCancel) btnCancel.style.display = modelBusy ? "inline-block" : "none";
    const btnAll = document.getElementById("btn-fetch-all");
    if (btnAll) btnAll.disabled = modelBusy;
    const btnDel = document.getElementById("btn-delete-all");
    if (btnDel) btnDel.disabled = modelBusy;

    const missing = (status.missing_core || []).length;
    baseModelsReady = missing === 0;

    if (openIfMissing && missing > 0) {
      showNotice(
        "필수 모델 " + missing + "개가 없습니다.\n" +
        "[받기] 또는 [누락 모델 전체 받기] 로 내려받거나,\n" +
        "models/ 하위에 직접 배치하세요.",
        "warn"
      );
      openSettings();
    }

    return status;
  } catch (e) {
    appendLog("❌ 모델 상태 확인 실패: " + e);
    return null;
  }
}

function openSettings() {
  document.getElementById("settings-overlay").style.display = "flex";
  refreshModelStatus(false);
}

function closeSettings() {
  document.getElementById("settings-overlay").style.display = "none";
}

function showNotice(msg, kind) {
  const el = document.getElementById("settings-notice");
  if (!el) return;
  el.className = kind || "";
  el.textContent = msg;
  el.style.display = "block";
}

function showToast(msg, kind, ms) {
  const wrap = document.getElementById("toast-wrap");
  if (!wrap) return;
  const el = document.createElement("div");
  el.className = "toast " + (kind || "");
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(function () {
    el.style.opacity = "0";
    setTimeout(function () { el.remove(); }, 300);
  }, ms || 6000);
}

async function downloadModel(id) {
  if (modelBusy) {
    showToast("이미 다운로드가 진행 중입니다.", "warn");
    return;
  }
  try {
    const res = await window.pywebview.api.download_model(id);
    if (res.ok) {
      modelBusy = true;
      appendLog("📥 다운로드 시작: " + id);
      refreshModelStatus(false);
    } else {
      showToast(res.error || "다운로드를 시작하지 못했습니다.", "err");
    }
  } catch (e) {
    appendLog("❌ 다운로드 예외: " + e);
  }
}

async function downloadBootstrap() {
  if (modelBusy) {
    showToast("이미 다운로드가 진행 중입니다.", "warn");
    return;
  }
  try {
    const res = await window.pywebview.api.download_bootstrap();
    if (res.ok) {
      modelBusy = true;
      const codes = (res.codes || ["eng", "kor"]).join(", ");
      showNotice("기본 언어(" + codes + ") 임베딩 모델을 내려받습니다.", "");
      appendLog("📥 기본 언어 부트스트랩 다운로드 시작: " + codes);
      refreshModelStatus(false);
    } else {
      showToast(res.error || "시작하지 못했습니다.", "err");
    }
  } catch (e) {
    appendLog("❌ 부트스트랩 다운로드 예외: " + e);
  }
}

async function downloadAllMissing() {
  if (modelBusy) {
    showToast("이미 다운로드가 진행 중입니다.", "warn");
    return;
  }
  try {
    const res = await window.pywebview.api.download_all_missing();
    if (res.ok) {
      modelBusy = true;
      showNotice("누락 모델을 순차적으로 내려받습니다.", "");
      appendLog("📥 누락 모델 전체 다운로드 시작");
      refreshModelStatus(false);
    } else {
      showToast(res.error || "시작하지 못했습니다.", "err");
    }
  } catch (e) {
    appendLog("❌ 전체 다운로드 예외: " + e);
  }
}

async function cancelDownload() {
  try {
    await window.pywebview.api.cancel_download();
    appendLog("⏹ 다운로드 취소를 요청했습니다.");
  } catch (e) {
    appendLog("❌ 취소 실패: " + e);
  }
}

async function deleteModel(id) {
  if (modelBusy) {
    showToast("다운로드 중에는 삭제할 수 없습니다.", "warn");
    return;
  }
  if (!confirm("이 모델의 가중치를 삭제할까요?\n" + id)) return;
  try {
    const res = await window.pywebview.api.delete_model(id);
    if (res.ok) {
      appendLog("🗑 삭제 완료: " + id + " (" + fmtBytes(res.freed) + " 확보)");
    } else {
      showToast(res.error || "삭제 실패", "err");
    }
    refreshModelStatus(false);
  } catch (e) {
    appendLog("❌ 삭제 예외: " + e);
  }
}

async function deleteAllModels() {
  if (modelBusy) {
    showToast("다운로드 중에는 삭제할 수 없습니다.", "warn");
    return;
  }
  if (!confirm("모든 모델 가중치를 삭제할까요?\n다시 사용하려면 재다운로드가 필요합니다.")) return;
  try {
    const res = await window.pywebview.api.delete_all_models();
    if (res.ok) {
      appendLog("🗑 전체 삭제 완료 (" + fmtBytes(res.freed) + " 확보)");
      showNotice("전체 모델을 삭제했습니다.", "");
    }
    refreshModelStatus(false);
  } catch (e) {
    appendLog("❌ 전체 삭제 예외: " + e);
  }
}

async function onAutoFetchChange() {
  const el = document.getElementById("auto-fetch-toggle");
  if (!el) return;
  try {
    await window.pywebview.api.set_auto_fetch(el.checked);
    appendLog("⚙ 모델 자동 취득: " + (el.checked ? "켬" : "끔"));
  } catch (e) {
    appendLog("❌ 설정 변경 실패: " + e);
  }
}

function onModelRequired(payload) {
  const action = payload.action || "";
  const msg = payload.message || "";

  if (action === "open_settings") {
    showNotice(msg, payload.auto ? "" : "warn");
    if (!payload.auto) openSettings();
    else if ((payload.models || []).length) openSettings();
    showToast(msg, payload.auto ? "warn" : "err", 9000);
    modelBusy = !!payload.auto;
    refreshModelStatus(false);
    return;
  }

  if (action === "fetch_done") {
    showNotice(msg, "done");
    showToast(msg, "ok");
    modelBusy = false;
    refreshModelStatus(false);
    return;
  }

  if (action === "fetch_failed") {
    showNotice(msg, "fail");
    showToast(msg, "err", 12000);
    modelBusy = false;
    openSettings();
    refreshModelStatus(false);
  }
}

function updateLangBadge(code, name, resolved, bootstrap) {
  const el = document.getElementById("lang-badge");
  if (!el) return;

  if (resolved && code) {
    el.textContent = "🌐 " + code + (name ? " (" + name + ")" : "");
    el.className = "meta-badge on";
    el.title = "판별 확정된 언어입니다.";
    return;
  }

  const boot = (bootstrap && bootstrap.length) ? bootstrap.join("+") : "eng+kor";
  el.textContent = "🌐 미확정 · " + boot;
  el.className = "meta-badge";
  el.title = "언어를 확정하지 못해 기본 언어 모델로 진행합니다: " + boot;
}

function updateCrossover(cx) {
  if (!cx) return;
  const phase = document.getElementById("phase-badge");
  const vram = document.getElementById("vram-badge");
  if (phase) {
    phase.textContent = "🔀 " + (cx.phase || "idle");
    phase.className = cx.phase === "idle" ? "meta-badge" : "meta-badge on";
  }
  if (vram && typeof cx.vram_free_gb === "number") {
    vram.textContent = "📊 VRAM " + cx.vram_free_gb.toFixed(2) + " GB";
  }
}

function onModelEvent(info) {
  const wrap = document.getElementById("dl-progress-wrap");
  const fill = document.getElementById("dl-progress-fill");
  const label = document.getElementById("dl-progress-label");
  const status = info.status || "";

  if (status === "start" || status === "progress" || status === "retry") {
    modelBusy = true;
    if (wrap) wrap.style.display = "block";
    const pct = info.percent || 0;
    if (fill) fill.style.width = pct + "%";
    if (label) {
      if (info.total > 0) {
        const mbDown = (info.downloaded / (1024 * 1024)).toFixed(1);
        const mbTotal = (info.total / (1024 * 1024)).toFixed(1);
        label.textContent = info.label + " — " + mbDown + " / " + mbTotal + " MB (" + pct + "%)";
      } else {
        label.textContent = info.label;
      }
    }
    const st = document.getElementById("status-text");
    if (st) st.textContent = "다운로드 " + pct + "% — " + (info.label || "");
    return;
  }

  if (status === "repo_start" || status === "lang_start") {
    modelBusy = true;
    if (wrap) wrap.style.display = "block";
    appendLog("📥 " + info.label);
    refreshModelStatus(false);
    return;
  }

  if (status === "repo_done" || status === "lang_done" || status === "all_done") {
    if (status === "all_done") modelBusy = false;
    if (wrap && status === "all_done") wrap.style.display = "none";
    if (fill) fill.style.width = "0%";
    appendLog("✅ " + info.label);
    refreshModelStatus(false);
    return;
  }

  if (status === "repo_error" || status === "lang_error" || status === "error") {
    modelBusy = false;
    if (wrap) wrap.style.display = "none";
    appendLog("❌ " + info.label);
    showToast(info.label || "다운로드 실패", "err", 9000);
    refreshModelStatus(false);
    return;
  }

  if (status === "cancelled" || status === "cancel_requested") {
    modelBusy = false;
    if (wrap) wrap.style.display = "none";
    appendLog("⏹ " + (info.label || "취소됨"));
    refreshModelStatus(false);
    return;
  }

  if (status === "deleted") {
    modelBusy = false;
    refreshModelStatus(false);
    return;
  }

  if (status === "exists" || status === "ready" || status === "skip" || status === "file_done") {
    if (status !== "file_done") appendLog("  · " + info.label);
  }
}

function markModelsLoaded() {
  modelsLoaded = true;
  const btn = document.getElementById("btn-load-models");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "로드 완료";
  }
}

function markModelsUnloaded() {
  modelsLoaded = false;
  const btn = document.getElementById("btn-load-models");
  if (btn) {
    btn.disabled = false;
    btn.textContent = "모델 로드";
  }
}

function markModelsLoading(label) {
  const btn = document.getElementById("btn-load-models");
  if (btn) {
    btn.disabled = true;
    btn.textContent = label || "로드 중...";
  }
}

async function loadModels() {
  markModelsLoading("로드 중...");
  try {
    const res = await window.pywebview.api.load_models();
    if (res.ok) {
      markModelsLoaded();
      appendLog("✅ 모델 준비 완료 (" + (res.providers || []).join(", ") + ")");
      showToast("모델 준비 완료", "ok", 4000);
      await refreshModelStatus(false);
      return true;
    }
    markModelsUnloaded();
    appendLog("❌ 모델 로드 실패: " + res.error);
    showNotice(res.error, "fail");
    openSettings();
    return false;
  } catch (e) {
    appendLog("❌ 모델 로드 예외: " + e);
    markModelsUnloaded();
    return false;
  }
}

async function unloadModels() {
  try {
    const res = await window.pywebview.api.unload_models();
    markModelsUnloaded();
    updateCrossover(res.crossover);
    appendLog("♻️ 모델을 반납했습니다.");
  } catch (e) {
    appendLog("❌ 반납 실패: " + e);
  }
}

let evalHistory = [];
let evalIndex = -1;

async function toggleDevTools() {
  try {
    const res = await window.pywebview.api.toggle_devtools();
    if (res && res.ok) {
      if (res.mode === "remote") {
        appendLog("🛠 원격 DevTools 를 브라우저에서 열었습니다: " + res.url);
        showToast("크롬에서 DevTools 를 열었습니다.\n" + res.url, "ok", 6000);
      } else {
        appendLog("🛠 내장 DevTools 토글");
      }
      return;
    }

    const msg = (res && res.error) || "DevTools 를 열지 못했습니다.";
    const hint = (res && res.hint) || "";
    appendLog("⚠️ " + msg, "js-warn");
    if (hint) appendLog("   " + hint, "js-info");
    openDebug(res && res.status);
  } catch (e) {
    appendLog("❌ DevTools 호출 실패: " + e, "js-err");
    openDebug(null);
  }
}

function openDebug(status) {
  const ov = document.getElementById("debug-overlay");
  if (!ov) return;
  ov.style.display = "flex";

  const badge = document.getElementById("debug-mode-badge");
  if (badge) {
    if (status && status.listening) {
      badge.textContent = "원격 :" + status.port + " 사용 가능";
    } else if (status && status.webview && status.webview.toggle_devtools) {
      badge.textContent = "내장 DevTools 사용 가능";
    } else {
      badge.textContent = "인앱 전용 (원격/내장 불가)";
    }
  }

  switchDebugTab("eval");
  setTimeout(function () {
    const inp = document.getElementById("dbg-eval-line");
    if (inp) inp.focus();
  }, 50);

  dbgPrint(
    "인앱 디버그 콘솔입니다. JS 표현식을 입력하면 이 창에서 평가됩니다.\n" +
    "예: document.querySelectorAll('.logis-result').length\n" +
    "    await window.pywebview.api.check_models()\n",
    ""
  );
}

function closeDebug() {
  const ov = document.getElementById("debug-overlay");
  if (ov) ov.style.display = "none";
}

function switchDebugTab(name) {
  const tabs = document.querySelectorAll(".dbg-tab");
  for (const t of tabs) {
    t.classList.toggle("active", t.dataset.tab === name);
  }
  const bodies = { eval: "dbg-eval", env: "dbg-env", dom: "dbg-dom", api: "dbg-api" };
  for (const [k, id] of Object.entries(bodies)) {
    const el = document.getElementById(id);
    if (el) el.style.display = k === name ? "flex" : "none";
  }
  if (name === "env") loadDebugEnv();
  if (name === "api") loadDebugApi();
}

function dbgPrint(text, cls) {
  const pre = document.getElementById("dbg-eval-out");
  if (!pre) return;
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = text + "\n";
  pre.appendChild(span);
  pre.scrollTop = pre.scrollHeight;
}

function fmtResult(v) {
  if (v === undefined) return "undefined";
  if (v === null) return "null";
  if (typeof v === "function") return v.toString().slice(0, 400);
  if (v instanceof Element) {
    return "<" + v.tagName.toLowerCase() +
      (v.id ? "#" + v.id : "") +
      (v.className ? "." + String(v.className).split(" ").join(".") : "") + ">";
  }
  if (v instanceof NodeList || Array.isArray(v)) {
    const arr = Array.from(v).slice(0, 20).map(fmtResult);
    return "[" + arr.join(", ") + (v.length > 20 ? ", … " + v.length + "개" : "") + "]";
  }
  if (typeof v === "object") {
    try {
      return JSON.stringify(v, null, 2);
    } catch (e) {
      return String(v);
    }
  }
  return String(v);
}

async function runEval(code) {
  dbgPrint("> " + code, "dbg-cmd");
  try {
    const fn = new Function("return (async () => { return (" + code + "); })();");
    const result = await fn();
    dbgPrint(fmtResult(result), "dbg-ret");
  } catch (e1) {
    try {
      const fn2 = new Function("return (async () => { " + code + " })();");
      const result2 = await fn2();
      dbgPrint(fmtResult(result2), "dbg-ret");
    } catch (e2) {
      dbgPrint((e2 && e2.stack) ? e2.stack : String(e2), "dbg-exc");
    }
  }
}

function onEvalKey(ev) {
  const inp = ev.target;
  if (ev.key === "Enter") {
    const code = inp.value.trim();
    if (!code) return;
    evalHistory.push(code);
    evalIndex = evalHistory.length;
    inp.value = "";
    runEval(code);
    return;
  }
  if (ev.key === "ArrowUp") {
    ev.preventDefault();
    if (evalIndex > 0) {
      evalIndex -= 1;
      inp.value = evalHistory[evalIndex] || "";
    }
    return;
  }
  if (ev.key === "ArrowDown") {
    ev.preventDefault();
    if (evalIndex < evalHistory.length - 1) {
      evalIndex += 1;
      inp.value = evalHistory[evalIndex] || "";
    } else {
      evalIndex = evalHistory.length;
      inp.value = "";
    }
  }
}

async function loadDebugEnv() {
  const pre = document.getElementById("dbg-env-out");
  if (!pre) return;
  pre.textContent = "조회 중...";
  try {
    const snap = await window.pywebview.api.debug_snapshot();
    snap.ui = {
      userAgent: navigator.userAgent,
      viewport: window.innerWidth + "x" + window.innerHeight,
      dpr: window.devicePixelRatio,
      url: location.href,
    };
    pre.textContent = JSON.stringify(snap, null, 2);
  } catch (e) {
    pre.textContent = "스냅샷 조회 실패: " + e;
  }
}

function queryDom() {
  const sel = document.getElementById("dbg-dom-sel");
  const out = document.getElementById("dbg-dom-out");
  if (!sel || !out) return;
  const q = sel.value.trim();
  if (!q) return;
  try {
    const nodes = document.querySelectorAll(q);
    let txt = "matched: " + nodes.length + "\n\n";
    Array.from(nodes).slice(0, 30).forEach(function (n, i) {
      txt += "[" + i + "] " + fmtResult(n) + "\n";
      const attrs = [];
      for (const a of n.attributes || []) {
        attrs.push(a.name + '="' + a.value + '"');
      }
      if (attrs.length) txt += "     " + attrs.join(" ") + "\n";
      const t = (n.textContent || "").trim().slice(0, 120);
      if (t) txt += "     text: " + t + "\n";
      txt += "\n";
    });
    out.textContent = txt;
  } catch (e) {
    out.textContent = "선택자 오류: " + e;
  }
}

function loadDebugApi() {
  const pre = document.getElementById("dbg-api-out");
  if (!pre) return;
  if (!window.pywebview || !window.pywebview.api) {
    pre.textContent = "pywebview API 미준비";
    return;
  }
  const names = Object.keys(window.pywebview.api).sort();
  let txt = "노출된 Python API " + names.length + "개\n\n";
  for (const n of names) {
    txt += "  await window.pywebview.api." + n + "()\n";
  }
  txt += "\nConsole 탭에서 그대로 호출해 결과를 확인할 수 있습니다.";
  pre.textContent = txt;
}

async function refreshDevtoolsBadge() {
  const badge = document.getElementById("devtools-badge");
  if (!badge) return;
  try {
    const st = await window.pywebview.api.devtools_status();
    badge.style.display = "inline-block";
    badge.style.cursor = "pointer";
    badge.onclick = toggleDevTools;

    if (st && st.listening) {
      badge.className = "meta-badge on";
      badge.textContent = "🛠 :" + st.port + " (" + st.targets + ")";
      badge.title = "크롬에서 " + st.url + " 로 접속하세요";
    } else if (st && st.webview && st.webview.toggle_devtools) {
      badge.className = "meta-badge on";
      badge.textContent = "🛠 내장";
      badge.title = "F12 로 내장 DevTools 를 엽니다";
    } else {
      badge.className = "meta-badge";
      badge.textContent = "🛠 인앱";
      badge.title = (st && st.hint) || "Ctrl+Shift+D 로 인앱 디버그 콘솔을 엽니다";
    }
  } catch (e) {
    badge.style.display = "none";
  }
}

async function openImage() {
  try {
    const res = await window.pywebview.api.load_input();
    if (!res.ok) {
      if (res.error) appendLog("⚠️ " + res.error);
      return;
    }
    if (res.mode === "text") {
      appendLog("📂 텍스트 로드 완료: " + res.chars + "자");
    } else {
      appendLog("📂 이미지 로드 완료: " + res.width + "x" + res.height);
    }
    if (!modelsLoaded) {
      appendLog("ℹ️ 모델이 아직 로드되지 않았습니다. [실행] 을 누르면 자동으로 로드한 뒤 언어 판별과 분류까지 이어서 진행합니다.");
      return;
    }
    await detectLanguage();
    if (res.mode !== "text") await autoSelectSchema();
  } catch (e) {
    appendLog("❌ 입력 로드 예외: " + e);
  }
}

async function detectLanguage() {
  try {
    const res = await window.pywebview.api.detect_language();
    if (!res.ok) return;
    const lang = res.language || {};
    const st = await refreshModelStatus(false);
    const resolved = st ? !!st.language_resolved : false;

    renderMeta({
      "언어": (lang.code || "-") + " / " + (lang.name || "-"),
      "확정여부": resolved ? "확정" : "미확정 (기본 언어 사용)",
      "문자체계": lang.script || "-",
      "마진": (lang.margin || 0).toFixed(4),
      "판정단계": lang.stage || "-",
      "사용 언어": st && st.active_codes ? st.active_codes.join(", ") : "-",
    });

    if (resolved) {
      appendLog("🌐 언어 확정: " + lang.code + " (" + lang.name + ")");
    } else {
      const boot = (st && st.bootstrap_languages || ["eng", "kor"]).join(", ");
      appendLog("⚠️ 언어 미확정 — 기본 언어 " + boot + " 로 진행합니다.", "js-warn");
      showToast(
        "언어를 확정하지 못했습니다.\n기본 언어(" + boot + ") 모델로 진행합니다.",
        "warn", 7000
      );
    }
  } catch (e) {
    appendLog("❌ 언어 판별 예외: " + e);
  }
}

async function autoSelectSchema() {
  try {
    const cls = await window.pywebview.api.classify_document();
    if (!cls.ok) {
      appendLog("⚠️ 문서 분류 실패: " + cls.error);
      return;
    }
    const v = cls.verdict || {};
    appendLog(
      "🔍 문서 유형: " + v.group + "/" + v.code +
      " (margin " + (v.code_margin || 0).toFixed(4) + ")"
    );
    if (cls.low_confidence) {
      appendLog(
        "⚠️ 문서 유형 판정 신뢰도가 낮습니다. 상단 드롭다운에서 스키마를 직접 고르세요.",
        "js-warn"
      );
      showToast(
        "문서 유형 자동 판정이 불확실합니다.\n스키마를 직접 선택하면 정확해집니다.",
        "warn", 9000
      );
    }

    const sel = document.getElementById("schema-select");
    let matched = false;
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === cls.schema) {
        sel.selectedIndex = i;
        matched = true;
        break;
      }
    }
    if (matched) {
      await onSchemaChange();
      appendLog("📋 스키마 자동 선택: " + cls.schema);
    } else {
      appendLog("⚠️ 분류된 스키마를 목록에서 찾지 못했습니다: " + cls.schema);
    }
  } catch (e) {
    appendLog("❌ 자동 분류 예외: " + e);
  }
}

function renderMeta(obj) {
  const el = document.getElementById("meta-panel");
  if (!el) return;
  let html = "";
  for (const [k, v] of Object.entries(obj)) {
    html +=
      '<div class="meta-row"><span class="meta-key">' + k + "</span>" +
      '<span class="meta-val">' + v + "</span></div>";
  }
  el.innerHTML = html;
}

async function initSchemas() {
  if (!window.pywebview || !window.pywebview.api) return;
  try {
    const res = await window.pywebview.api.list_schemas();
    if (!res.ok) return;
    const sel = document.getElementById("schema-select");
    sel.innerHTML = '<option value="">스키마 선택</option>';
    for (const s of res.schemas) {
      const opt = document.createElement("option");
      opt.value = s.filename;
      opt.textContent = s.domain + " / " + s.doc_type + " (" + s.field_count + " 필드)";
      sel.appendChild(opt);
    }
  } catch (e) {
    appendLog("❌ 스키마 목록 예외: " + e);
  }
}

async function onSchemaChange() {
  const sel = document.getElementById("schema-select");
  const filename = sel.value;
  if (!filename) {
    currentSchema = null;
    return;
  }
  try {
    const res = await window.pywebview.api.load_schema(filename);
    if (res.ok) {
      currentSchema = res.schema;
      appendLog("📋 스키마 로드: " + res.domain + " / " + res.doc_type + " (" + res.fields.length + " 필드)");
      renderFieldList(res.fields);
    }
  } catch (e) {
    appendLog("❌ 스키마 로드 예외: " + e);
  }
}

function renderFieldList(fields) {
  const container = document.getElementById("field-results");
  container.innerHTML = "";
  for (const f of fields) {
    const div = document.createElement("div");
    div.className = "field-item";
    div.id = "field-" + f;
    div.innerHTML =
      '<span class="field-name">' + f + "</span>" +
      '<span class="field-score">-</span>';
    container.appendChild(div);
  }
}

async function runPipeline() {
  const iouTh = parseFloat(document.getElementById("iou-threshold").value);
  const marginTh = parseFloat(document.getElementById("margin-threshold").value);
  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  document.getElementById("nms-log").innerHTML = "";
  document.getElementById("crop-previews").innerHTML = "";

  const autoLoad = !modelsLoaded;
  if (autoLoad) {
    markModelsLoading("자동 로드 중...");
    appendLog("🧩 모델이 아직 로드되지 않아 실행과 함께 자동으로 로드합니다.");
    updateProgress(3, "모델 자동 로드 중");
  }

  try {
    const schemaArg = currentSchema ? JSON.stringify(currentSchema) : "";
    const res = await window.pywebview.api.run_pipeline(
      schemaArg, iouTh, marginTh, ""
    );

    if (res.models_ready) markModelsLoaded();
    if (res.crossover) updateCrossover(res.crossover);

    if (!res.ok) {
      if (autoLoad && !res.models_ready) markModelsUnloaded();
      appendLog("❌ 파이프라인 실패: " + res.error);
      showToast(res.error || "파이프라인 실패", "err", 9000);
      refreshModelStatus(false);
      return;
    }

    lastResults = res;
    const items = res.results || res.assignments || [];
    appendLog("✅ 파이프라인 완료 (" + (res.mode || "-") + "): " + items.length + "개 결과");

    const lang = res.language || {};
    if (lang.code) {
      renderMeta({
        "언어": (lang.code || "-") + " / " + (lang.name || "-"),
        "문자체계": lang.script || "-",
        "마진": (lang.margin || 0).toFixed(4),
        "판정단계": lang.stage || "-",
      });
    }

    for (const name of (res.absent_fields || [])) {
      const el = document.getElementById("field-" + name);
      if (el) {
        el.className = "field-item suppressed";
        el.querySelector(".field-score").textContent = "부재";
      }
    }

    for (const r of items) {
      const name = r.field_name || r.category;
      const el = document.getElementById("field-" + name);
      if (el) {
        el.className = "field-item confirmed";
        const sc = typeof r.score === "number" ? r.score.toFixed(4) : "-";
        el.querySelector(".field-score").textContent = sc;
      }
    }

    const arena = res.arena || {};
    if (arena.rounds) {
      appendLog(
        "🥊 NMS ARENA — 라운드 " + arena.rounds +
        " | 존재 " + (arena.present || []).length +
        " | 부재 " + (arena.absent || []).length +
        " | 무주공산 " + (arena.unclaimed || 0) + "패치"
      );
    }

    renderNmsLog(res.log || []);
    if (res.mode === "vision") renderCropPreviews(items);
    else renderRecord(res.record || {});

    document.getElementById("btn-save").disabled = false;

    if (autoLoad) await refreshModelStatus(false);
  } catch (e) {
    appendLog("❌ 파이프라인 예외: " + e);
    if (autoLoad) markModelsUnloaded();
  } finally {
    btn.disabled = false;
  }
}

function renderNmsLog(lines) {
  const el = document.getElementById("nms-log");
  if (!el) return;
  const filtered = lines.filter(function (l) {
    return /🚫|✨|🔒|⚠️|🌀|🔗|🧬|🔀|♻️|✂️/.test(l);
  });
  el.innerHTML = filtered.map(function (l) {
    return "<div>" + l.replace(/</g, "&lt;") + "</div>";
  }).join("");
  el.scrollTop = el.scrollHeight;
}

function renderRecord(record) {
  const container = document.getElementById("crop-previews");
  if (!container) return;
  let html = '<pre style="font-size:11px;line-height:1.6;white-space:pre-wrap;">';
  html += JSON.stringify(record, null, 2).replace(/</g, "&lt;");
  html += "</pre>";
  container.innerHTML = html;
}

function renderCropPreviews(results) {
  const container = document.getElementById("crop-previews");
  container.innerHTML = "";
  for (const r of results) {
    const card = document.createElement("div");
    card.className = "crop-card";

    const img = document.createElement("img");
    if (r.crop_base64) {
      img.src = "data:image/png;base64," + r.crop_base64;
    }

    const label = document.createElement("span");
    label.className = "crop-label";
    label.textContent = r.field_name || r.category;

    const score = document.createElement("span");
    score.className = "crop-score";
    const sc = typeof r.score === "number" ? r.score.toFixed(4) : "-";
    const mg = typeof r.margin === "number" ? r.margin.toFixed(4) : "-";
    score.textContent = "score " + sc + " / margin " + mg;

    const value = document.createElement("span");
    value.className = "crop-score";
    value.textContent = (r.value || r.ocr_text || "-").slice(0, 60);

    card.appendChild(img);
    card.appendChild(label);
    card.appendChild(score);
    card.appendChild(value);

    if (r.upscale && r.upscale > 1.01) {
      const up = document.createElement("span");
      up.className = "crop-score";
      up.textContent = "upscale " + r.upscale.toFixed(2) + "x / tiles " + (r.tiles || 1);
      card.appendChild(up);
    }

    container.appendChild(card);
  }
}

async function saveResults() {
  if (!lastResults) {
    appendLog("⚠️ 저장할 결과가 없습니다.");
    return;
  }
  try {
    const res = await window.pywebview.api.save_results("");
    if (res.ok) {
      appendLog("💾 결과 저장 완료: " + res.output_dir);
    } else {
      appendLog("❌ 저장 실패: " + res.error);
    }
  } catch (e) {
    appendLog("❌ 저장 예외: " + e);
  }
}

let consoleHookOn = true;
let consoleLineCount = 0;
const CONSOLE_MAX_LINES = 4000;

function appendLog(msg, cls) {
  const pre = document.getElementById("console-output");
  if (!pre) return;

  if (cls) {
    const span = document.createElement("span");
    span.className = cls;
    span.textContent = msg + "\n";
    pre.appendChild(span);
  } else {
    pre.appendChild(document.createTextNode(msg + "\n"));
  }

  consoleLineCount += 1;
  if (consoleLineCount > CONSOLE_MAX_LINES) {
    while (pre.childNodes.length > CONSOLE_MAX_LINES * 0.8) {
      pre.removeChild(pre.firstChild);
    }
    consoleLineCount = pre.childNodes.length;
  }

  pre.scrollTop = pre.scrollHeight;
}

function clearConsole() {
  const pre = document.getElementById("console-output");
  if (pre) pre.textContent = "";
  consoleLineCount = 0;
}

function copyConsole() {
  const pre = document.getElementById("console-output");
  if (!pre) return;
  const text = pre.textContent || "";
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () {
      showToast("콘솔 로그를 클립보드에 복사했습니다.", "ok", 3000);
    });
  } else {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    showToast("콘솔 로그를 복사했습니다.", "ok", 3000);
  }
}

function toggleConsoleHook() {
  const el = document.getElementById("js-console-hook");
  consoleHookOn = el ? !!el.checked : true;
  appendLog(consoleHookOn ? "🛠 JS 콘솔 캡처 켬" : "🛠 JS 콘솔 캡처 끔");
}

function fmtArgs(args) {
  const out = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a instanceof Error) {
      out.push(a.stack || (a.name + ": " + a.message));
    } else if (typeof a === "object" && a !== null) {
      try {
        out.push(JSON.stringify(a));
      } catch (e) {
        out.push(String(a));
      }
    } else {
      out.push(String(a));
    }
  }
  return out.join(" ");
}

function installConsoleHook() {
  const levels = [
    ["log", "", "⬜"],
    ["info", "js-info", "🟦"],
    ["warn", "js-warn", "🟨"],
    ["error", "js-err", "🟥"],
  ];

  for (const [name, cls, icon] of levels) {
    const orig = console[name] ? console[name].bind(console) : function () {};
    console[name] = function () {
      try {
        orig.apply(null, arguments);
      } catch (e) {}
      if (!consoleHookOn) return;
      const text = fmtArgs(arguments);
      appendLog(icon + " " + text, cls);
      if ((name === "error" || name === "warn") &&
          window.pywebview && window.pywebview.api &&
          window.pywebview.api.js_console) {
        try {
          window.pywebview.api.js_console(name, text);
        } catch (e) {}
      }
    };
  }

  window.addEventListener("error", function (ev) {
    const msg =
      (ev.message || "Unknown error") +
      " @ " + (ev.filename || "?") + ":" + (ev.lineno || 0) + ":" + (ev.colno || 0);
    appendLog("🟥 " + msg, "js-err");
    if (ev.error && ev.error.stack) appendLog(ev.error.stack, "js-err");
    if (window.pywebview && window.pywebview.api && window.pywebview.api.js_console) {
      try {
        window.pywebview.api.js_console("error", msg);
      } catch (e) {}
    }
  });

  window.addEventListener("unhandledrejection", function (ev) {
    let msg = "Unhandled promise rejection";
    try {
      msg += ": " + (ev.reason && ev.reason.stack ? ev.reason.stack : String(ev.reason));
    } catch (e) {}
    appendLog("🟥 " + msg, "js-err");
    if (window.pywebview && window.pywebview.api && window.pywebview.api.js_console) {
      try {
        window.pywebview.api.js_console("error", msg);
      } catch (e) {}
    }
  });
}

function installDevtoolsKeys() {
  document.addEventListener("keydown", function (e) {
    const isF12 = e.key === "F12";
    const isInspect = (e.ctrlKey || e.metaKey) && e.shiftKey &&
      (e.key === "I" || e.key === "i");
    const isConsole = (e.ctrlKey || e.metaKey) && e.shiftKey &&
      (e.key === "J" || e.key === "j");
    const isInapp = (e.ctrlKey || e.metaKey) && e.shiftKey &&
      (e.key === "D" || e.key === "d");

    if (isInapp) {
      e.preventDefault();
      const ov = document.getElementById("debug-overlay");
      if (ov && ov.style.display === "flex") closeDebug();
      else openDebug(null);
      return;
    }

    if (isF12 || isInspect || isConsole) {
      e.preventDefault();
      toggleDevTools();
      return;
    }

    if (e.key === "Escape") {
      const dbg = document.getElementById("debug-overlay");
      if (dbg && dbg.style.display === "flex") {
        closeDebug();
        return;
      }
      const st = document.getElementById("settings-overlay");
      if (st && st.style.display === "flex") closeSettings();
      return;
    }

    if ((e.ctrlKey || e.metaKey) && (e.key === "L" || e.key === "l")) {
      e.preventDefault();
      clearConsole();
    }
  });
}

function updateProgress(pct, label) {
  const fill = document.getElementById("progress-bar-fill");
  fill.style.width = pct + "%";
  const labelEl = document.getElementById("progress-label");
  labelEl.textContent = label || pct + "%";
  const statusEl = document.getElementById("status-text");
  statusEl.textContent = label || pct + "%";
}

window.addEventListener("DOMContentLoaded", function () {
  installConsoleHook();
  installDevtoolsKeys();

  appendLog("🚀 NMS-OCR 시작");
  appendLog("ℹ️ F12 : DevTools | Ctrl+Shift+D : 인앱 콘솔 | Ctrl+L : 로그 지우기");
  appendLog("⏳ pywebview API 대기 중...");

  waitForApi(async function () {
    appendLog("✅ pywebview API 준비 완료");
    initSchemas();
    refreshDevtoolsBadge();
    refreshModelStatus(true);

    try {
      const gpuInfo = await window.pywebview.api.get_gpu_info();
      const badge = document.getElementById("gpu-badge");
      if (badge) {
        if (gpuInfo.vram && gpuInfo.vram.available) {
          const low = gpuInfo.low_vram ? " ⚠️" : "";
          badge.textContent = "🎮 " + gpuInfo.accel_label + " | " + gpuInfo.vram.total_gb + " GB" + low;
          badge.className = gpuInfo.low_vram ? "gpu-rocm" : "gpu-cuda";
          appendLog("🎮 GPU: " + gpuInfo.accel_label + " | VRAM: " + gpuInfo.vram.total_gb + " GB");
          if (gpuInfo.low_vram) {
            appendLog(
              "⚠️ VRAM " + gpuInfo.vram.total_gb + " GB — 저VRAM 모드로 동작합니다.",
              "js-warn"
            );
            appendLog(
              "   정제 LLM(Qwen3.5-4B)은 4bit 양자화/오프로드로 로드되며, " +
              "부족하면 OCR 원문으로 폴백합니다.",
              "js-info"
            );
            showToast(
              "VRAM " + gpuInfo.vram.total_gb + " GB 감지 — 저VRAM 모드\n" +
              "bitsandbytes 설치를 권장합니다: pip install bitsandbytes",
              "warn",
              10000
            );
          }
        } else {
          badge.textContent = "💻 " + gpuInfo.accel_label;
          badge.className = "gpu-cpu";
          appendLog("💻 GPU 미사용: " + gpuInfo.accel_label);
        }
      }
    } catch (e) {
      appendLog("❌ GPU 정보 조회 실패: " + e);
    }

    setInterval(async function () {
      try {
        const cx = await window.pywebview.api.get_crossover();
        updateCrossover(cx);
      } catch (e) {}
    }, 2000);
  });
});