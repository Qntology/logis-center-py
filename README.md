# Logis-center-py · Vision / Text NMS Extraction

> **Local-first Document Understanding Engine** — A fully offline pipeline that classifies a document type, competes field anchors against image patches, crops only the regions that win, and reads them line by line. No external API calls, no cloud inference.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Model File Paths](#model-file-paths)
- [AI Model Architecture](#ai-model-architecture)
- [Line-by-Line Reading](#line-by-line-reading)
- [Memory Management (VRAM / RAM Staging)](#memory-management--vram--ram-staging)
- [Build & Run](#build--run)
- [Runtime Flags](#runtime-flags)
- [License](#license)

---

## Overview

| Domain | Description |
|--------|-------------|
| **Trade Documents** | Assembles 55 trade form schemas (CI, PL, BL, LC, PO, COO, …) from `bias.json` `base + overlay`, classifies the incoming page, and extracts a structured record with a relay plan linking documents to each other. |
| **Comics / Webtoon** | Assembles 6 comic schemas (WEBTOON, MANGA, COMIC_STRIP, GRAPHIC_NOVEL, STORYBOARD, ILLUSTRATION) and extracts dialogue, narration, sound effects, signage, and cast attributes as arrays. |
| **Plain Text** | Runs the text-only pipeline — natural-language chunking, surprisal scoring, NMS battle, format gating, PLINKO, and exclusive assignment. |

Everything — language detection, document classification, OCR, refinement, grounding, and vector indexing — runs **on the local machine**.

---

## Key Features

### Language Determination (Evidence-Ordered)
- **Unicode block decisive** — A single Hangul / Kana / Thai / Greek / Hebrew / Tamil codepoint settles the language immediately; no cosine needed.
- **Latin function-word profile** — For Latin script, scores function-word banks for `eng / fra / deu / spa / ita / por / nld` and requires a hit margin before committing.
- **CTC confidence race** — Loads each candidate PP-OCRv5 recognizer, measures mean CTC confidence on the actual page, and picks the winner.
- **Served-scope guard** — Refuses to commit to a language whose models are not installed when the margin is weak, so a multi-gigabyte download is never triggered by noise.

### Vision Pipeline
- **Patch Grid** — SigLIP2 joint tower produces an aspect-preserving letterboxed grid; dense patch features are pooled down to a cap so one patch covers a label+value cell.
- **Field Affinity Map** — Category-level column cosine: patch-axis z-score → max-pool field-count debias (Gumbel) → category-axis recalibration → chrome anchor subtraction.
- **NMS Arena** — Patch-level exclusive competition across rounds; categories that fail to hold a minimum territory are declared **absent** and written as `null` rather than hallucinated.
- **Spatial Residual** — Decomposes global tone + row spread + column spread, keeping only the residual for categories whose activation is too diffuse.
- **Crop Planning** — Connected components → row-band expansion → oversized split → IoU suppression → coverage merge → **text-box snap** → coverage guarantee.
- **Value Grounding** — Extreme-value (Gumbel) normalized surprisal of the extracted string inside vs. outside its own crop; cross-crop claims are dropped.

### Text Pipeline
- **Chunker** — Dual-track (surface + canonical) sliding windows over natural language.
- **Surprisal** — Bias bank max-pool minus prejudice bank, z-scored across fields, penalized by bank size.
- **NMS Battle** — Overlapping chunks absorb losers; gap bridging assigns orphan spans to the stronger neighbour.
- **Format Gate** — Infers `DATE / TRACKING_CODE / IDENTIFIER / LINK / NUMERIC / ENUM / PHONE / ADDRESS / SYNTHESIS / TEXT` per field and rejects mismatches.
- **PLINKO** — Cliff-detection window growth that stops when the score collapses, then overrides weaker chunk assignments.
- **Exclusive Assign** — Confirmed label-value pairs first, then greedy 1:1 assignment with margin thresholds.

### Record Shaping & Indexing
- **Identity Resolution** — `doc_number → no → reference_number → OCR shape scan → task_{crc32}` fallback chain with homoglyph normalization (`O→0`, `I→1`, `S→5`).
- **Envelope Addresses** — SHA-1 derived `id / digest / cc / bcc / ref / to`, plus a CRC32 `index` for relay routing.
- **Relay Plan** — Emits reference edges (`reference_bl`, `reference_invoice`) to the other 55 trade forms, suppressed when identity was a fallback.
- **zvec Store** — Append-only binary vector log with blake2b keys, tombstones, and ratio-triggered compaction.

---

## Project Structure

```
nms-ocr/
├── app.py                          # Entry point — NMSOcrApp, CLI + pywebview JSApi
├── requirements.txt                # Python dependencies (PyPI only)
├── setup_and_run.bat               # Windows one-shot setup: venv → torch → deps → paddle → run
├── LICENSE                         # Apache License 2.0
├── readme.md
│
├── core/                           # Model lifecycle, language, memory, storage
│   ├── __init__.py                 # Public surface + lazy import map
│   ├── device.py                   # Accelerator detect, dtype select, VRAM probe
│   ├── memory.py                   # System RAM probe, staging admission, reclaim
│   ├── crossover.py                # Phase switching + VRAM/RAM-aware slot eviction
│   ├── jobqueue.py                 # Serialized pipeline slots with backpressure
│   ├── registry.py                 # Per-step model requirements, prereq expansion
│   ├── model_manager.py            # MODEL_SPECS, LANG_REPO_TEMPLATES, Paddle slugs
│   ├── model_fetcher.py            # HF resumable downloader (base / lang / stanza)
│   ├── paddle_bootstrap.py         # PaddlePaddle auto-install + oneDNN runtime flags
│   ├── llm.py                      # TextEmbedder, RefinerLLM, EmbeddingRouter
│   ├── siglip_joint.py             # SigLIP2 joint vision-text space + self-test
│   ├── embedding.py                # A.X-VE patch embedder + PatchGrid
│   ├── ppocr.py                    # PaddleOCRRec + PaddleTextDetector (shared)
│   ├── pdf_render.py               # PDFium render + text layer, multi-page, backends
│   ├── line_reader.py              # Line split, read windows, weighted cross-vote
│   ├── nlp.py                      # Stanza gate (UPOS / DEPREL / lemma)
│   ├── lang_codes.py               # ISO maps, script ranges, decisive blocks, priors
│   ├── language.py                 # LanguageDetector — block → profile → cosine
│   ├── text_prep.py                # Sanitize, label-value pairs, label echo gate
│   ├── nms.py                      # Candidate / NMSProcessor, Gumbel helpers
│   ├── scoring.py                  # FieldScorer — bias/prejudice cosine
│   ├── trade_schema.py             # 55 trade forms from bias.json base+overlay
│   ├── comics_schema.py            # 6 comic schemas from bias.json base+overlay
│   ├── bias_bridge.py              # Auxiliary multilingual bias dictionary bridge
│   ├── bias_dictionary.json        # Auxiliary bias/prejudice/semantic phrases
│   ├── record_shape.py             # Envelope record, digest, relay plan
│   ├── zvec.py                     # Local vector store (append log + compaction)
│   ├── phrase_cache.py             # Anchor embedding cache over zvec
│   ├── cropper.py                  # RegionCropper — bbox crop + JSON export
│   ├── fp8.py                      # E4M3 weight storage + KV cache adapter
│   ├── diagnostics.py              # Tensor/config/space self-diagnosis
│   └── devtools.py                 # Remote DevTools port, webview capabilities
│
├── vision_pipeline/                # Image → structured record
│   ├── __init__.py
│   ├── patch_grid.py               # VisionPatchGrid, gutters, legibility, table band
│   ├── doc_type_nms.py             # Depth 1 group → Depth 2 code + title gate
│   ├── field_heatmap.py            # Category column cosine, title suppression, arena apply
│   ├── nms_arena.py                # Patch-level exclusive competition, territories
│   ├── vision_nms.py               # Components → crop plans → coverage merge → text snap
│   ├── text_boxes.py               # Text box detection (PP-OCR det + CV fallback)
│   ├── text_upscale.py             # Text-aware upscale, VLM fit, overlap tiles
│   ├── ocr_extract.py              # Crop → line read → refine → record
│   ├── value_grounding.py          # In/out surprisal grounding verdicts
│   └── pipeline.py                 # VisionPipeline — STEP 1~5 orchestration
│
├── text_pipeline/                  # Plain text → structured record
│   ├── __init__.py
│   ├── chunker.py                  # Chunk, dual-track window generation
│   ├── field_bank.py               # FieldBank, phrase embedding, cross prejudice
│   ├── surprisal.py                # Dual-bank scoring, label prefix stripping
│   ├── nms_battle.py               # Winner absorption, gap bridging, coverage
│   ├── format_gate.py              # Field format inference and rejection
│   ├── plinko.py                   # Cliff-detection window growth + override
│   ├── exclusive_assign.py         # 1:1 assignment, record materialization
│   └── pipeline.py                 # TextPipeline — PHASE A~F orchestration
│
├── schema/
│   └── bias.json                   # Master dictionary: trade_schema, comics_schema,
│                                   # search_bridge, per-language field banks,
│                                   # operators, time/season/status filters
│
├── ui/
│   ├── index.html                  # Pywebview shell
│   └── app.js                      # Console hook, model manager, in-app debugger
│
├── ax-ve/                          # A.X-VE remote code (configuration/modeling/processing)
├── models/                         # Downloaded weights (see Model File Paths)
└── output/                         # results.json, record.json, extracted.json, crops
```

---

## Model File Paths

```
nms-ocr/models/
├── .active_language.json                         # Last committed language + source
│
├── ax-ve/                                        # A.X-VE vision encoder (base)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── configuration_ax_ve.py
│   ├── modeling_ax_ve.py
│   ├── image_processing_ax_ve.py
│   └── processing_ax_ve.py
│
├── PP-OCRv5_mobile_det_safetensors/              # Text region detection (base)
│   ├── config.json
│   ├── inference.yml
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── korean_PP-OCRv5_mobile_rec_safetensors/       # Text line recognition (language scope)
│   ├── config.json
│   ├── inference.yml                             # character_dict, use_space_char, rec_image_shape
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── siglip2-large-patch16-512-{code}-16384/       # Joint vision-text tower (language scope)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── Qwen3-Embedding-{code}-16384/                 # 1024d text embedding (language scope)
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── modules.json
│   ├── tokenizer.json
│   ├── merges.txt
│   └── vocab.json
│
├── Qwen3.5-4B-{code}-16384/                      # Refiner VLM (language scope)
│   ├── chat_template.jinja
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── video_preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── merges.txt
│   └── vocab.json
│
├── siglip2-base-patch16-naflex/                  # NaFlex image processor config (support)
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── alphaedge-ai/                                 # Optional, manual placement only
│   ├── config.json
│   └── model.safetensors
│
├── stanza/                                       # Optional NLP gate
│   ├── ko/
│   │   ├── resources.json
│   │   └── models/{tokenize,pos,lemma,depparse,mwt,ner}/*.pt
│   └── en/
│
└── zvec/                                         # Local vector store
    ├── zvec_index.fields.d1024.{tag}.bin
    ├── zvec_index.docs.d1024.{tag}.bin
    └── zvec_index.phrases.d1024.{tag}.bin        # Anchor embedding cache
```

### PP-OCRv5 Language Slug Map

`{prefix}` is empty for Chinese/Japanese (`PaddlePaddle/PP-OCRv5_mobile_rec_safetensors`) and `{slug}_` otherwise.

| ISO 639-3 | Paddle slug | Repository |
|-----------|-------------|------------|
| `zho`, `jpn` | *(none)* | `PaddlePaddle/PP-OCRv5_mobile_rec_safetensors` |
| `kor` | `korean` | `PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors` |
| `eng` | `en` | `PaddlePaddle/en_PP-OCRv5_mobile_rec_safetensors` |
| `rus`, `ukr`, `bel` | `eslav` | `PaddlePaddle/eslav_PP-OCRv5_mobile_rec_safetensors` |
| `bul`, `srp`, `mkd`, `kaz`, `mon` | `cyrillic` | `PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_safetensors` |
| `ara`, `fas`, `urd` | `arabic` | `PaddlePaddle/arabic_PP-OCRv5_mobile_rec_safetensors` |
| `hin`, `mar`, `nep`, `ben` | `devanagari` | `PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_safetensors` |
| `tam` | `ta` | `PaddlePaddle/ta_PP-OCRv5_mobile_rec_safetensors` |
| `tel` | `te` | `PaddlePaddle/te_PP-OCRv5_mobile_rec_safetensors` |
| `ell` | `el` | `PaddlePaddle/el_PP-OCRv5_mobile_rec_safetensors` |
| `tha` | `th` | `PaddlePaddle/th_PP-OCRv5_mobile_rec_safetensors` |
| everything else | `latin` | `PaddlePaddle/latin_PP-OCRv5_mobile_rec_safetensors` |

PaddleOCR ships no Japanese-only recognizer, so `jpn` falls back to the Han-shared Chinese model and the substitution is logged.

---

## AI Model Architecture

### Model Roles

| Model | Approx. Size | Role | Slot | Load Timing |
|-------|--------------|------|------|-------------|
| **PP-OCRv5 mobile rec** | ~10 MB | Text line recognition (CTC) | `ocr` | Every task |
| **PP-OCRv5 mobile det** | ~5 MB | Text region detection (DB) | shared singleton | Every task |
| **SigLIP2 Large** | ~1.3 GB | Patch grid + text anchors in one contrastive space | `joint` | Classification, heatmap, grounding |
| **Qwen3-Embedding** | ~0.9 GB | 1024d multilingual text embedding | `embedder` | Anchor bank, text axis, zvec |
| **Qwen3.5 4B** | ~7.3 GB | Crop reading, JSON field refinement | `refiner` | Extraction (lazy) |
| **A.X-VE** | ~0.4 GB | Fallback patch grid when SigLIP2 is unavailable | `vision` | Fallback only |
| **Stanza** | ~50 MB/lang | Tokenize, POS, lemma, depparse noise gate | *(inline)* | Optional |

### Step Requirements

| Step | Base models | Language models | Stanza |
|------|-------------|-----------------|--------|
| `bootstrap` | — | `ppocr`, `qwen3emb` | — |
| `patch_grid` | `ax-ve` | `siglip2` | — |
| `doc_type` | — | `siglip2`, `qwen3emb` | — |
| `language` | — | `ppocr`, `qwen3emb` | — |
| `field_heatmap` | — | `siglip2`, `qwen3emb` | — |
| `ocr_extract` | `ppocr-det` | `ppocr` | optional |
| `refine` | — | `qwen35` | — |
| `text_pipeline` | — | `qwen3emb` | optional |

### Vision Pipeline Flow

```
[Input image]
   │
   ├─ Language      Unicode block → Latin profile → CTC confidence race → commit
   │
   ├─ STEP 1        SigLIP2 letterbox → dense patch features → pool to ≤256 patches
   │                Legibility map (blank / illegible / legible via Otsu)
   │
   ├─ STEP 2        Category column cosine → field-count debias → category recalibration
   │                Title row suppression (identity categories exempt)
   │
   ├─ STEP 2.5      NMS Arena — patch-level exclusive competition, absent detection
   │                Spatial residual for over-diffuse categories
   │
   ├─ STEP 3        Components → row band → split → IoU suppress → coverage merge
   │                Text snap to detected boxes → coverage guarantee
   │
   ├─ STEP 4        Per crop: detect lines → read windows → weighted vote → refine
   │
   └─ STEP 5        Value grounding (in vs out surprisal) → record_shape → JSON + zvec
```

### Document Type NMS

```
Depth 1 (group)   comics / trade
   bank-neutral key matrix over patches, chrome anchors subtracted
   decisive_margin() vs. noise band → if tied, lower groups stay in the pool

Depth 2 (code)    55 trade codes + 6 comic codes
   fused = code_surprisal + 2.0 × title_gate + text_axis_z
   raw cosine advantage required — normalization-only margins are refused
```

---

## Line-by-Line Reading

Long strings degrade toward the end of a VLM generation. The engine never asks for a whole balloon at once.

```
crop
 └─ PP-OCRv5 det boxes ──► lines[0..n]        (fallback: ink projection + multi-round resplit)
       │
       ├─ per line: PP-OCRv5 rec ──► (text, ctc_confidence)
       │
       └─ sliding windows  span=1, stride=1, overlap=1
             window[0:2] ─► VLM read_raw ─► "야! 이거 봤어?\n드디어 그 작가"
             window[1:3] ─► VLM read_raw ─► "드디어 그 작가\n신작 떴대!"
             window[2:3] ─► VLM read_raw ─► "신작 떴대!"
                     │
                     ▼
             weighted vote per line index
```

### Vote Weights

| Source | Weight | Condition |
|--------|--------|-----------|
| PP-OCRv5 rec | `2.6 × ctc_confidence` (floor `0.30`) | Always |
| VLM, row count matches window span | `1.00` | Aligned |
| VLM, row count mismatched | `0.35` | Realigned by character overlap against the OCR hint |
| VLM, OCR confidence ≥ 0.50 but character overlap < 0.25 | `× 0.25` | Hallucination penalty |

When the top two weights differ by less than `0.80`, the engine falls back to **per-character weighted consensus** over same-length candidates instead of picking the longer string.

### Guards
- **Prompt echo** — Replies containing `script rule`, `do not romanize`, `transcribe now`, … are discarded outright.
- **Romanization block** — For non-Latin scripts, replies with ≥55 % Latin characters are discarded and re-read once with a stricter prompt.
- **Script latch** — The Unicode block verdict is latched into every crop prompt; a probe read runs only if language detection never committed.
- **OCR rescue** — Lines the VLM failed to read are filled from the recognizer alone.
- **OCR-only mode** — When the refiner cannot load, recognized lines are promoted directly into array fields, so accuracy survives even without the VLM.

---

## Memory Management — VRAM / RAM Staging

### Why RAM, not just VRAM

`safetensors` weights are mapped into **system RAM** before quantization and transfer to the GPU. A 4 GB card is not the binding constraint — a 7.3 GB checkpoint on a machine with 2.7 GB free RAM is killed by the OS before CUDA is ever touched.

### Three Admission Gates

```
┌────────────────────────────────────────────────────────────┐
│ 1. memory.can_stage(need_gb)                               │
│    usable = min(available_phys, commit_available)          │
│    need   = weight × 1.15 + 1.2 GB safety                  │
│    4-bit ready → staging estimate = weight × 0.35          │
│    Fails → MissingModelError, never a process kill         │
├────────────────────────────────────────────────────────────┤
│ 2. CrossoverSwitch._make_room_for(slot)                    │
│    vram_need = est_gb + headroom                           │
│    ram_need  = est_gb × 2.4 + 1.5 GB                       │
│    Evicts by (EVICT_LAST rank, -est_gb) until both clear   │
│    Thrash detector warns when a slot is evicted ≥3 of 6    │
├────────────────────────────────────────────────────────────┤
│ 3. Graceful degradation                                    │
│    Refiner unavailable → use_refiner = False               │
│    → PP-OCRv5 output is promoted straight to fields        │
└────────────────────────────────────────────────────────────┘
```

### RAM Probe Sources

| Platform | Source | Fields |
|----------|--------|--------|
| Windows | `GlobalMemoryStatusEx` via `ctypes` | `ullAvailPhys`, `ullAvailPageFile` |
| Any | `psutil.virtual_memory` + `swap_memory` | `available`, `free` |
| Fallback | — | Admission passes with a note |

### Phase Plan

```
PHASE_EMBEDDING   keep [joint, vision, ocr, embedder, nlp]   release [refiner]
PHASE_GENERATION  keep [ocr, refiner, nlp]                   release [joint, vision, embedder]
PHASE_IDLE        keep []                                    release [all]
```

### FP8 Weight Storage

```
Linear weights ──► quantize_fp8(axis=1) ──► float8_e4m3fn + per-row scale
   SigLIP2: 289 Linear layers → 1274 MB → 671 MB measured
   Skipped: head.*, embeddings, logit
   Ampere (sm_86) has no fp8 tensor cores → storage-only, upcast at matmul
```

### KV Cache Planning

```
per_token = 2 × layers × kv_heads × head_dim × elem_size
Qwen3.5 4B: 24 × 2 × 256 × 2 B = 49,152 B/token → 4096 tokens ≈ 201 MB
Placement: VRAM when free > need + 512 MB, else CPU
FP8 KV (E4M3 quantize-dequantize) is disabled for multimodal generation,
because positional indexing depends on exact cache length.
```

### Quantization — Verified, Not Assumed

`import bitsandbytes` succeeding does not mean 4-bit works. The bundled CUDA binary frequently mismatches the installed PyTorch build, and the failure only surfaces halfway through `from_pretrained` — after several gigabytes have already been staged in RAM.

`core/quant_bootstrap.py` therefore runs a real kernel before committing:

The probe builds a `Linear4bit(64, 64)`, moves it to CUDA, runs a forward pass, and checks the output is finite. Only then does the refiner receive a `BitsAndBytesConfig`.

| Stage | Action on failure |
|-------|-------------------|
| Package missing or below `0.43.0` | `pip install "bitsandbytes>=0.43.0"` at runtime |
| 4-bit probe fails | Retry with `Linear8bitLt` (8-bit LLM.int8) |
| 8-bit probe fails | Disable quantization, log the reinstall command, fall back to CPU offload |
| RAM insufficient even at 4-bit | Skip the refiner entirely, promote PP-OCRv5 output to fields |

### Memory Budget by Mode

Ratios are derived from the verified capability, not guessed:

| Mode | VRAM ratio | RAM staging ratio | Qwen3.5-4B (7.3 GB on disk) |
|------|-----------|-------------------|------------------------------|
| `4bit` (NF4 + double quant) | 0.42 | 0.35 | ~3.1 GB VRAM / ~2.6 GB RAM |
| `8bit` (LLM.int8) | 0.70 | 0.60 | ~5.1 GB VRAM / ~4.4 GB RAM |
| `offload` (no quantization) | 1.15 | 1.00 | ~8.4 GB VRAM / ~7.3 GB RAM |

### Compute dtype

`bnb_4bit_compute_dtype` follows the device, not a hardcoded constant:

| Compute capability | dtype | Rationale |
|--------------------|-------|-----------|
| sm_80 and above (Ampere+) | `bfloat16` | Native bf16 tensor cores; matches the checkpoint dtype, no cast |
| Below sm_80 | `float16` | bf16 would be emulated |

`bnb_4bit_quant_storage` is set to the same dtype so `accelerate` can shard the quantized weights without an intermediate conversion.

### Optimization Checklist

- [x] `low_cpu_mem_usage=True` on every `from_pretrained`
- [x] `max_memory` split derived from measured free VRAM and free RAM
- [x] Architecture pre-detection from `config.json` — no duplicate loader attempts
- [x] `unload()` drops references directly, never `to("cpu")` (which doubles RAM)
- [x] `release_vision()` fully frees the tower when RAM cannot hold a CPU copy
- [x] Shared `PaddleTextDetector` singleton — one model init per process
- [x] Anchor embedding cache in zvec — repeated runs skip the encoder entirely
- [x] Eviction failure cleans `slot.instance` and reclaims before re-raising
- [x] Quantization capability verified with a live kernel, never assumed
- [x] 4-bit → 8-bit → offload degradation chain, each step logged
- [x] Compute dtype matched to the device (bf16 on Ampere+)
- [x] VRAM and RAM estimates derived from the verified mode

---

## Build & Run

### Prerequisites

- **Python** 3.9 – 3.13 (3.12 verified)
- **CUDA Toolkit** ≥ 11.8 — optional, for GPU acceleration
- **Windows / Linux / macOS** — the batch installer is Windows-only

### Windows One-Shot

```bat
setup_and_run.bat
```

Steps: venv → pip upgrade → PyTorch (CUDA auto-detect) → `requirements.txt` → PaddlePaddle from its own index → memory check → `bitsandbytes` if VRAM < 6 GB → verify → launch.

### Manual

```bash
python -m venv venv
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# paddlepaddle is NOT on PyPI — use the official index
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr

# 4-bit quantization for low-VRAM machines (CUDA only)
pip install -r requirements-gpu.txt

python app.py
```

### CLI

```bash
python app.py                                  # pywebview UI
python app.py document.pdf --save              # headless, write to output/
python app.py --check-only                     # model status report
python app.py --fetch-all --lang kor           # download everything for Korean
python app.py img.jpg --line-overlap 0         # strict one-line-per-read
python app.py img.jpg --no-refine              # OCR only, skip the VLM
python app.py img.jpg --ram-limit 8            # cap staging admission at 8 GB
python app.py doc.pdf --page 3                 # process page 3
python app.py doc.pdf --pdf-dpi 300            # higher render resolution
python app.py doc.pdf --pdf-force-vision       # ignore the embedded text layer
python app.py img.jpg --quant 4bit             # force NF4, fail loudly if broken
python app.py img.jpg --quant off              # disable quantization entirely
python app.py --check-only                     # includes the quantization self-test
```

---

## Runtime Flags

| Flag / Variable | Default | Effect |
|-----------------|---------|--------|
| `--line-span` / `NMS_LINE_SPAN` | `1` | Lines per read window |
| `--line-overlap` / `NMS_LINE_OVERLAP` | `1` | Lines shared between neighbouring windows |
| `NMS_LINE_STRIDE` | `1` | Window advance in lines |
| `--no-line-read` / `NMS_LINE_READ=0` | on | Read the whole crop in one call |
| `--ram-limit` / `NMS_RAM_LIMIT_GB` | auto | Hard cap on staging admission |
| `--allow-low-ram` / `NMS_ALLOW_LOW_RAM=1` | off | Bypass the RAM gate (process-kill risk) |
| `--paddle-device` / `NMS_PADDLE_DEVICE` | `cpu` | PaddleOCR execution device |
| `--paddle-mkldnn` / `NMS_PADDLE_MKLDNN=1` | off | Re-enable oneDNN (see note) |
| `--no-paddle-install` / `NMS_PADDLE_AUTOINSTALL=0` | on | Disable runtime PaddlePaddle installation |
| `NMS_PADDLE_THREADS` | `min(8, cpu_count)` | PaddleOCR CPU threads |
| `--pdf-dpi` / `NMS_PDF_DPI` | `200` | PDF render resolution (72–400, auto-capped at 4000 px on the long side) |
| `--pdf-pages` / `NMS_PDF_MAX_PAGES` | `32` | Maximum pages processed per document |
| `--pdf-force-vision` / `NMS_PDF_FORCE_VISION=1` | off | Ignore the embedded text layer, use the vision pipeline |
| `--page` | `1` | Page number to process (1-based) |
| `NMS_PDF_BACKEND` | auto | Pin the PDF backend (`pypdfium2` / `pypdf`) |
| `--quant` / `NMS_QUANT_MODE` | `auto` | `auto` / `4bit` / `8bit` / `off` |
| `--no-quant-install` / `NMS_QUANT_AUTOINSTALL=0` | on | Disable runtime bitsandbytes installation |
| `NMS_QUANT_TIMEOUT` | `600` | Seconds allowed for the bitsandbytes install |
| `NMS_DIAG` | `1` | `0` quiet, `1` summary, `2` per-tensor |
| `--vram-budget` | auto | Override detected VRAM |
| `--low-vram` | auto (< 6 GB) | Force quantized / offloaded refiner |
| `--no-crossover` | off | Keep every model resident |
| `--devtools-port` | `9222` | Chrome remote debugging port |

**oneDNN note** — `FLAGS_use_mkldnn` defaults to `0`. The PP-OCRv5 detection graph carries a `pir::ArrayAttribute<pir::DoubleAttribute>` that the oneDNN PIR executor cannot convert, raising `ConvertPirAttribute2RuntimeAttribute not support`. The plain CPU kernels handle it correctly.

---

## License

**Apache License 2.0** — Copyright 2026 Qntology. See [`LICENSE`](./LICENSE).

### Model Licenses

| Model | Component | Role in this project | License |
|-------|-----------|----------------------|---------|
| [PaddlePaddle/PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors) | Text line recognition (zh / ja) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (ko) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/en_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (en) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/latin_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Latin) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/eslav_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (East Slavic) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Cyrillic) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/arabic_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Arabic) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Devanagari) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/ta_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/ta_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Tamil) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/te_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/te_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Telugu) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/el_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/el_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Greek) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/th_PP-OCRv5_mobile_rec_safetensors](https://huggingface.co/PaddlePaddle/th_PP-OCRv5_mobile_rec_safetensors) | Text line recognition (Thai) | `ocr` slot | Apache-2.0 |
| [PaddlePaddle/PP-OCRv5_mobile_det_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors) | Text region detection | Line splitting, crop snap | Apache-2.0 |
| [google/siglip2-large-patch16-512](https://huggingface.co/google/siglip2-large-patch16-512) | Joint vision-text encoder | Patch grid, anchors, grounding | Apache-2.0 |
| [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) | NaFlex image processor | Preprocessing config | Apache-2.0 |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Text embedding | Anchor bank, text axis, zvec | Apache-2.0 |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Vision-language model | Crop reading, JSON refinement | Apache-2.0 |
| [skt/A.X-VE](https://huggingface.co/skt/A.X-VE) | Vision encoder | Fallback patch grid | See model card |
| [stanfordnlp/stanza-ko](https://huggingface.co/stanfordnlp/stanza-ko) · [stanza-en](https://huggingface.co/stanfordnlp/stanza-en) | NLP pipeline | Noise gate, lemma | Apache-2.0 |

Weights under the `alphaedge-ai/` organization (`siglip2-large-patch16-512-{code}-16384`, `Qwen3-Embedding-{code}-16384`, `Qwen3.5-4B-{code}-16384`) are language-scoped derivatives with a reduced 16,384-token vocabulary. They inherit the upstream license of the base model they derive from.

No model weights are redistributed with this repository. Every checkpoint is fetched at runtime from its original publisher, and its own license terms apply to the downloaded artifact.

### Python Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| [transformers](https://github.com/huggingface/transformers) | Model loading, generation, processors | Apache-2.0 |
| [torch](https://github.com/pytorch/pytorch) | Tensor runtime, CUDA kernels | BSD-3-Clause |
| [torchvision](https://github.com/pytorch/vision) | Image transforms | BSD-3-Clause |
| [numpy](https://github.com/numpy/numpy) | Cosine, NMS, connected components | BSD-3-Clause |
| [pillow](https://github.com/python-pillow/Pillow) | Crop, resize, letterbox | MIT-CMU (HPND) |
| [safetensors](https://github.com/huggingface/safetensors) | Weight serialization | Apache-2.0 |
| [tokenizers](https://github.com/huggingface/tokenizers) | Fast tokenization | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Repository resolution | Apache-2.0 |
| [accelerate](https://github.com/huggingface/accelerate) | `device_map`, `max_memory` sharding | Apache-2.0 |
| [sentencepiece](https://github.com/google/sentencepiece) | SigLIP2 / Gemma tokenizer | Apache-2.0 |
| [protobuf](https://github.com/protocolbuffers/protobuf) | Tokenizer model parsing | BSD-3-Clause |
| [requests](https://github.com/psf/requests) | Resumable HF downloads | Apache-2.0 |
| [pywebview](https://github.com/r0x0r/pywebview) | Desktop shell, JS bridge | BSD-3-Clause |
| [stanza](https://github.com/stanfordnlp/stanza) | Tokenize / POS / lemma / depparse | Apache-2.0 |
| [PyYAML](https://github.com/yaml/pyyaml) | `inference.yml` charset fallback | MIT |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | PaddleOCR image preprocessing | Apache-2.0 (OpenCV) · MIT (wrapper) |
| [psutil](https://github.com/giampaolo/psutil) | Cross-platform RAM probe | BSD-3-Clause |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | PDF rasterization + text layer | Apache-2.0 OR BSD-3-Clause |
| [PDFium](https://pdfium.googlesource.com/pdfium/) | Bundled PDF engine (inside pypdfium2 wheels) | BSD-3-Clause |
| [pypdf](https://github.com/py-pdf/pypdf) | Pure-Python PDF text fallback | BSD-3-Clause |

### Optional Dependencies

| Library | Purpose | Auto-installed | License |
|---------|---------|----------------|---------|
| [paddlepaddle](https://github.com/PaddlePaddle/Paddle) | PP-OCRv5 execution runtime | `core/paddle_bootstrap.py` | Apache-2.0 |
| [paddleocr](https://github.com/PaddlePaddle/PaddleOCR) | `TextRecognition`, `TextDetection` | `core/paddle_bootstrap.py` | Apache-2.0 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4 4-bit / LLM.int8 8-bit quantization | `core/quant_bootstrap.py` | MIT |

### PDF Handling — Fully Permissive

PDF support runs on **PDFium**, the engine Chrome uses, via the `pypdfium2` wrapper. PDFium is **BSD-3-Clause**; the wrapper is dual-licensed **Apache-2.0 OR BSD-3-Clause**. The published wheels embed the prebuilt native library, so there is no Poppler, no Ghostscript, and no external binary to install.

Backend resolution order in `core/pdf_render.py`:

| Order | Backend | Render | Text | License |
|-------|---------|--------|------|---------|
| 1 | `pypdfium2` | ✅ | ✅ | Apache-2.0 / BSD-3-Clause |
| 2 | `pypdf` | ❌ | ✅ | BSD-3-Clause |

Both are installed by `requirements.txt`. If neither is present, PDF input is disabled with an actionable message while image and text input keep working.

### Text Layer Shortcut

Digital PDFs already carry an extractable text layer. When at least half the processed pages qualify (≥40 characters and ≥20 alphanumerics each), the engine skips OCR and the VLM entirely and feeds the embedded text straight into the text pipeline:

### License Compatibility — No Copyleft

Every runtime dependency of this project is permissively licensed. There is **no GPL, no LGPL, and no AGPL** component in the dependency graph, so the Apache-2.0 terms of this repository apply cleanly to any redistribution, including closed-source binaries.

| License family | Count | Examples |
|----------------|-------|----------|
| Apache-2.0 | 8 | transformers, safetensors, tokenizers, accelerate, sentencepiece, stanza, opencv, paddleocr |
| BSD-3-Clause | 7 | torch, torchvision, numpy, pywebview, pypdfium2, pypdf, psutil, protobuf |
| MIT | 2 | PyYAML, bitsandbytes |
| HPND (MIT-CMU) | 1 | pillow |
| **Copyleft (GPL / LGPL / AGPL)** | **0** | — |

Two copyleft dependencies existed in earlier revisions and have been removed:

| Removed | License | Replacement |
|---------|---------|-------------|
| `PyMuPDF` (`import fitz`) | AGPL-3.0-only | `pypdfium2` — BSD-3-Clause PDFium |
| `pdf2image` + Poppler binary | MIT wrapper, GPL-2.0/3.0 binary | `pypdf` — BSD-3-Clause text fallback |

Neither `fitz` nor `pdf2image` appears anywhere in the codebase, and neither is referenced by `core/pdf_render.py`'s backend resolution. A clean checkout installs no copyleft package.

To verify after installation:

```bash
pip install pip-licenses
pip-licenses --format=markdown --order=license
```

Model weights are fetched at runtime from their original publishers and are not redistributed here; each carries its own license as listed above. All currently referenced checkpoints are Apache-2.0 except `skt/A.X-VE`, which is optional and used only as a fallback patch-grid provider.

### GPU Notices

#### NVIDIA CUDA Toolkit

This application utilizes the NVIDIA CUDA Toolkit through PyTorch.
Portions of this software are copyrighted by NVIDIA Corporation.
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)

#### AMD ROCm

This application can utilize components from the AMD ROCm Platform on Linux.
Portions of this software are copyrighted by Advanced Micro Devices, Inc.
Licensed under the MIT License and/or Apache License 2.0.
[ROCm License](https://rocm.docs.amd.com/en/latest/about/license.html)

#### Intel oneDNN

PaddlePaddle bundles oneDNN (MKL-DNN), licensed under **Apache-2.0**.
This project disables the oneDNN execution path by default — see the note under [Runtime Flags](#runtime-flags).
[oneDNN License](https://github.com/oneapi-src/oneDNN/blob/main/LICENSE)

All GPU-related components listed above are permissively licensed or vendor EULAs governing the driver stack, not the application source. None imposes copyleft obligations on this repository.

---

## Contributing

Issues and Pull Requests are always welcome.