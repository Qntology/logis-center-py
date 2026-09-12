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

function msItem(id, name, ready, detail) {
  const cls = ready ? "ready" : "missing";
  const label = ready ? "✅ 준비 완료" : "❌ 누락";
  const sub = detail ? ' <span style="opacity:0.6;font-size:11px;">' + detail + "</span>" : "";
  return (
    '<div class="ms-item" id="ms-' + id + '">' +
    '<span class="ms-name">' + name + sub + "</span>" +
    '<span class="ms-state ' + cls + '">' + label + "</span>" +
    "</div>"
  );
}

async function refreshModelStatus() {
  if (!window.pywebview || !window.pywebview.api) return;
  try {
    const status = await window.pywebview.api.check_models();
    const overlay = document.getElementById("model-status-overlay");
    const baseList = document.getElementById("ms-model-list");
    const langList = document.getElementById("ms-lang-list");
    const statusText = document.getElementById("ms-status-text");

    let baseHtml = "";
    let anyMissing = false;

    for (const [key, info] of Object.entries(status.base || {})) {
      const mb = (info.bytes / (1024 * 1024)).toFixed(1);
      if (!info.ready || !info.config) anyMissing = true;
      baseHtml += msItem(key, info.label, info.ready && info.config, mb + " MB");
    }
    baseList.innerHTML = baseHtml;

    let langHtml = "";
    const code = status.lang_code || "";
    if (code) {
      langHtml += '<div style="font-size:12px;color:#a6adc8;margin:4px 0;">언어 스코프: ' +
        code + " (" + (status.lang_name || code) + ")</div>";
      for (const [kind, info] of Object.entries(status.language || {})) {
        const mb = (info.bytes / (1024 * 1024)).toFixed(1);
        langHtml += msItem("lang-" + kind, info.label, info.ready, mb + " MB");
      }
      const st = status.stanza || {};
      if (st.label) {
        const mb = ((st.bytes || 0) / (1024 * 1024)).toFixed(1);
        langHtml += msItem("stanza", st.label, st.ready, mb + " MB");
      }
    }
    langList.innerHTML = langHtml;

    baseModelsReady = !anyMissing;
    updateCrossover(status.crossover);
    updateLangBadge(code, status.lang_name);

    if (anyMissing) {
      statusText.textContent =
        "필수 모델이 없습니다. models/ 하위에 직접 배치하세요.\n" +
        "  models/alphaedge-ai/  models/ax-ve/  models/hayai/\n" +
        "이 앱은 고정 모델을 자동 다운로드하지 않습니다.\n" +
        "언어 스코프 모델(SigLIP2 / Qwen3 / Stanza)은 판별 후 자동 취득합니다.";
      overlay.style.display = "flex";
    } else {
      statusText.textContent = "모든 필수 모델이 준비되어 있습니다.";
      overlay.style.display = "none";
      appendLog("✅ 필수 모델 3종 준비 확인");
    }
  } catch (e) {
    appendLog("❌ 모델 상태 확인 실패: " + e);
  }
}

function closeModelStatus() {
  document.getElementById("model-status-overlay").style.display = "none";
}

function updateLangBadge(code, name) {
  const el = document.getElementById("lang-badge");
  if (!el) return;
  if (code) {
    el.textContent = "🌐 " + code + (name ? " (" + name + ")" : "");
    el.className = "meta-badge on";
  } else {
    el.textContent = "🌐 언어 미판별";
    el.className = "meta-badge";
  }
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

function onModelDownload(info) {
  const wrap = document.getElementById("ms-progress-wrap");
  const fill = document.getElementById("ms-progress-fill");
  const label = document.getElementById("ms-progress-label");
  const statusText = document.getElementById("ms-status-text");

  if (info.status === "start" || info.status === "progress") {
    if (wrap) wrap.style.display = "block";
    const pct = info.percent || 0;
    if (fill) fill.style.width = pct + "%";
    if (label && info.total > 0) {
      const mbDown = (info.downloaded / (1024 * 1024)).toFixed(1);
      const mbTotal = (info.total / (1024 * 1024)).toFixed(1);
      label.textContent = info.label + " — " + mbDown + " / " + mbTotal + " MB (" + pct + "%)";
    } else if (label) {
      label.textContent = info.label;
    }
    if (statusText) statusText.textContent = info.label;
    return;
  }

  if (info.status === "lang_done" || info.status === "repo_done") {
    if (wrap) wrap.style.display = "none";
    appendLog("✅ " + info.label);
    refreshModelStatus();
    return;
  }

  if (info.status === "lang_error" || info.status === "repo_error" || info.status === "error") {
    if (wrap) wrap.style.display = "none";
    appendLog("❌ " + info.label);
    return;
  }

  if (info.status === "cancelled") {
    if (wrap) wrap.style.display = "none";
    appendLog("⏹ " + info.label);
  }
}

async function loadModels() {
  const btn = document.getElementById("btn-load-models");
  btn.disabled = true;
  btn.textContent = "로드 중...";
  try {
    const res = await window.pywebview.api.load_models();
    if (res.ok) {
      modelsLoaded = true;
      btn.textContent = "로드 완료";
      appendLog("✅ 모델 준비 완료 (" + (res.providers || []).join(", ") + ")");
      await refreshModelStatus();
    } else {
      btn.textContent = "모델 로드";
      btn.disabled = false;
      appendLog("❌ 모델 로드 실패: " + res.error);
      const st = document.getElementById("ms-status-text");
      if (st) st.textContent = res.error;
      document.getElementById("model-status-overlay").style.display = "flex";
    }
  } catch (e) {
    appendLog("❌ 모델 로드 예외: " + e);
    btn.disabled = false;
    btn.textContent = "모델 로드";
  }
}

async function unloadModels() {
  try {
    const res = await window.pywebview.api.unload_models();
    modelsLoaded = false;
    const btn = document.getElementById("btn-load-models");
    btn.disabled = false;
    btn.textContent = "모델 로드";
    updateCrossover(res.crossover);
    appendLog("♻️ 모델을 반납했습니다.");
  } catch (e) {
    appendLog("❌ 반납 실패: " + e);
  }
}

async function toggleDevTools() {
  try {
    await window.pywebview.api.toggle_devtools();
  } catch (e) {
    appendLog("❌ DevTools 열기 실패: " + e);
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
      appendLog("ℹ️ [모델 로드] 를 먼저 실행하면 언어 판별과 자동 분류가 동작합니다.");
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
    updateLangBadge(lang.code, lang.name);
    renderMeta({
      "언어": (lang.code || "-") + " / " + (lang.name || "-"),
      "문자체계": lang.script || "-",
      "마진": (lang.margin || 0).toFixed(4),
      "판정단계": lang.stage || "-",
    });
    appendLog("🌐 언어 확정: " + lang.code + " (" + lang.name + ")");
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

  try {
    const schemaArg = currentSchema ? JSON.stringify(currentSchema) : "";
    const res = await window.pywebview.api.run_pipeline(
      schemaArg, iouTh, marginTh, ""
    );

    if (res.crossover) updateCrossover(res.crossover);

    if (!res.ok) {
      appendLog("❌ 파이프라인 실패: " + res.error);
      return;
    }

    lastResults = res;
    const items = res.results || res.assignments || [];
    appendLog("✅ 파이프라인 완료 (" + (res.mode || "-") + "): " + items.length + "개 결과");

    for (const r of items) {
      const name = r.field_name || r.category;
      const el = document.getElementById("field-" + name);
      if (el) {
        el.className = "field-item confirmed";
        const sc = typeof r.score === "number" ? r.score.toFixed(4) : "-";
        el.querySelector(".field-score").textContent = sc;
      }
    }

    renderNmsLog(res.log || []);
    if (res.mode === "vision") renderCropPreviews(items);
    else renderRecord(res.record || {});

    document.getElementById("btn-save").disabled = false;
  } catch (e) {
    appendLog("❌ 파이프라인 예외: " + e);
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

function appendLog(msg) {
  const pre = document.getElementById("console-output");
  pre.textContent += msg + "\n";
  pre.scrollTop = pre.scrollHeight;
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
  appendLog("🚀 NMS-OCR 시작");
  appendLog("⏳ pywebview API 대기 중...");

  waitForApi(async function () {
    appendLog("✅ pywebview API 준비 완료");
    initSchemas();
    refreshModelStatus();

    try {
      const gpuInfo = await window.pywebview.api.get_gpu_info();
      const badge = document.getElementById("gpu-badge");
      if (badge) {
        if (gpuInfo.vram && gpuInfo.vram.available) {
          badge.textContent = "🎮 " + gpuInfo.accel_label + " | " + gpuInfo.vram.total_gb + " GB";
          badge.className = "gpu-cuda";
          appendLog("🎮 GPU: " + gpuInfo.accel_label + " | VRAM: " + gpuInfo.vram.total_gb + " GB");
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