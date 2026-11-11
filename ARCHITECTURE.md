# avatar-media-mcp — repository architecture

> Status: **map of the code as it exists now** · verified against the tree: **2026-11-11**
> Renewal rule: on any doc change, re-verify claims against the code and update both fields.

This file is the map of the `avatar-media-mcp` code as it exists now. It is an appendix to [ARCHITECTURE-AOE.md](ARCHITECTURE-AOE.md) (the host environment) and follows its conventions: the code is the source of truth; this file is secondary to it — when they disagree, fix this file, never bend the code to match a document.

Markers on this file: state of **this tree now**.

- ✅ working and verified
- ⬜ planned — stub in the code (`unimplemented!("M*")`), tagged with its milestone

Companion documents (do not collapse these roles):

| File | Role |
| --- | --- |
| [ARCHITECTURE-AOE.md](ARCHITECTURE-AOE.md) | Host environment map: CHA0S AOE crates, agent runtime, MCP contour. |
| This file | Tree map of the media server. What is on disk. If this file and the code disagree, the code wins. |
| `IDENTITY-CORE-BRIEF.md` | The brief the identity core was written against (history, not a spec). |
| `avatar-media-mcp/README.md` | Build + AOE wiring quickstart. |

---

## 1. What this is

`avatar-media-mcp` is a Rust MCP server (`avatar-media-mcp/`, cargo crate) — the hands of the CHA0S AOE IDE agent for generative media with avatars: from single photos and portrait sets to full-length films. The agent is the brain (script, dialogue, approvals); the server executes everything that costs money or touches files: stores avatars, generates voiceover/frames/clips through external models, assembles the final cut through ffmpeg, tracks spend, and refuses to exceed the budget.

**State today:** the identity core (mathematics + local ArcFace inference) works and is tested; the data skeleton (domain types, configs, Postgres schema, cache, ffmpeg availability check) is in place; the generation pipeline (TTS / keyframes / TALK / BROLL / assembly) and the MCP interface are stubs tagged M1–M7.

The engineering idea everything rests on:

> **The face lives in a reference frame, the voice lives in one cloned voiceover, motion comes from the cheapest model that can handle a given shot, and assembly is deterministic ffmpeg, not a neural network.**

---

## 2. Repository tree

```
Media_mcp/
├── ARCHITECTURE.md            # this file
├── ARCHITECTURE-AOE.md        # CHA0S AOE map (host environment, not this project)
├── PRODUCT.md                 # AOE product definition
├── IDENTITY-CORE-BRIEF.md     # brief the identity core was written against (history)
├── identity-core-brief.zip    # brief + files archive (history)
├── Ksenia/                    # test asset set: image_01..image_22 (jpg/png)
└── avatar-media-mcp/          # THE SERVER — Rust crate
    ├── Cargo.toml             # rmcp, tokio, serde+schemars, reqwest, sqlx(postgres),
    │                          # image, ndarray; ort (load-dynamic) + ndarray under feature local-identity
    ├── README.md              # build, AOE wiring (context_servers), structure
    ├── .env.example           # FAL_KEY, ELEVENLABS_API_KEY, DATABASE_URL + optional
    ├── .gitignore             # target/, .env, models not in git
    ├── config/
    │   ├── models.yaml        # model registry: role → tier → {provider, model_id, params, price}
    │   └── platforms.yaml     # platform profiles + export rules (ai_disclosure, no auto-publish)
    ├── migrations/
    │   └── 0001_init.sql      # Postgres schema (§7)
    ├── models/                # NOT in git, downloaded locally:
    │   ├── insightface/buffalo_l/   # det_10g.onnx (SCRFD 16.9MB), w600k_r50.onnx (ArcFace R50 174MB),
    │   │                            # + 1k3d68, 2d106det, genderage (downloaded, not wired)
    │   └── ort/lib/                 # libonnxruntime.1.31.0.dylib (macOS arm64)
    ├── src/
    │   ├── lib.rs             # library root: config, domain, identity, providers, pipeline, infra, series
    │   ├── main.rs            # MCP server entry — STUB unimplemented!("M1")
    │   ├── bin/identity_test.rs     # working CLI: image → vector → cosine → core verdict
    │   ├── config/            # ✅ models.rs, platforms.rs, settings.rs
    │   ├── domain/            # ✅ avatar.rs, project.rs, script.rs, job.rs, cost.rs, series.rs
    │   ├── identity/          # ✅ CORE: scoring.rs (1667 lines), embeddings.rs, arcface.rs, threed.rs
    │   ├── providers/         # base.rs (traits) ✅, registry.rs (skeleton) ✅, fal/elevenlabs/fish/byteplus — empty structs
    │   ├── pipeline/          # avatar_build, audio, keyframes, shots, qc, assemble — STUBS M1–M6
    │   ├── infra/             # store.rs (Postgres) ✅, cache.rs ✅, ffmpeg.rs ✅(partial), ledger.rs — stub M2
    │   └── series.rs          # film serialization — stub M7
    └── tests/integration.rs   # 5 integration tests (cache-key, cosine, stage1, assignment)
```

Counters: `cargo test --features local-identity` → **18/18 green** (13 unit in `scoring.rs`, 2 unit in `arcface.rs`, 5 in `tests/integration.rs`). `cargo check` — 11 warnings (dead fields in stubs), 0 errors.

---

## 3. Wiring into CHA0S AOE

The MCP stdio server is registered in `~/.config/cha0s/settings.json`:

```jsonc
"context_servers": {
  "avatar-media": {
    "command": "/abs/path/to/avatar-media-mcp",
    "env": { "FAL_KEY": "...", "ELEVENLABS_API_KEY": "...", "DATABASE_URL": "..." }
  }
}
```

Zero changes to AOE code. Provider keys travel through the process env; `.env` is a second layer. The agent reads pipeline state directly from Postgres through the stock `kb`/`database` tools (§7).

---

## 4. Identity core — ✅ working

The hardest mathematics in the project. Written against `IDENTITY-CORE-BRIEF.md` (6 tasks, all implemented). Pure functions, no I/O; newtype wrappers (`CosineScore`, `GeometryDistance`, `EmbeddingSource`) make mixing incompatible units a compile error.

### 4.1 `src/identity/scoring.rs` (1667 lines) — verdicts and thresholds

| API | What it does |
| --- | --- |
| `IdentityThresholds` | pass/fail thresholds **per embedding source** (ArcFaceOnnx / VisionFallback — different vector spaces; mixing is an error); `for_source()`, `with_avatar_pass_threshold()` (per-avatar calibration), `validate_source()` |
| `stage1()` / `stage1_score()` / `stage1_embedding()` | QC stage 1: Pass / Borderline (→ judge) / Fail |
| `calibrate_thresholds()` | Bayesian calibration from positive/negative samples: Jeffreys prior (Beta), FPR upper bound via a self-contained incomplete beta + Lanczos lgamma (no statistics crates). **Key decision: on small samples, refuse auto-pass** — if FPR ≤ 1% is not provable, the gate goes to the judge rather than being trusted. An empty grey zone (fail == pass) is a valid outcome |
| `analyze_drift()` | drift model over a shot series: Theil–Sen slope (≥3 points — on 2 points the slope is just the endpoint difference and is not computed), robust medians for baseline/recent, step detection, single-outlier isolation, view-angle normalization (`FaceViewAngle`; offsets arrive from M0). Verdicts: Stable / DriftStarting / DriftCritical |
| `assign_faces_to_canons()` | multi-character assignment: **Hungarian algorithm** (O(n³), own implementation) over the cosine matrix; dummy rows for K≠M; per-face thresholds; anomalies: ExtraFace / MissingCharacter / TwinCanons |
| `aggregate_clip_frames()` | QC-2 for a clip: median = base level, intra-clip drift tracked separately (endpoint drop + slope); dropping an artifact frame does not produce a false drift |
| `fit_fusion_model()` / `fuse_identity_signals()` | channel fusion (embedding + FLAME geometry): logistic regression with view-angle interaction terms (weights are not hardcoded); channel conflict → Borderline (to the judge), never averaging; works without geometry with no API degradation |

### 4.2 `src/identity/arcface.rs` (376 lines) — local inference ✅

The full InsightFace pipeline in pure Rust — no Python, no OpenCV:

1. **SCRFD** (`det_10g.onnx`) — face detection + 5 landmarks on a 640px letterbox, threshold 0.5, best score wins; face selection by inter-ocular distance (main person first).
2. **Alignment** — least-squares similarity transform onto the canonical ArcFace 112×112 template: own implementation (solve6 — 6×6 Gaussian elimination with partial pivoting; 2×3 affine inversion; bilinear sampling — image 0.25 has no affine warp).
3. **ArcFace** (`w600k_r50.onnx`) — 512-d vector, L2-normalized (cosine = dot product).

Inference via `ort` (ONNX Runtime 2.0-rc.15, load-dynamic, dylib in `models/ort/lib/`). **Critical:** `AutoDevicePolicy::PreferCPU` is mandatory — the CoreML EP on macOS breaks SCRFD's dynamic outputs (rank mismatch). CPU is sufficient (~50 ms/face).

Verified on a real photo: self-similarity 1.0000; different frames of the same person → 0.8554 → PASS.

Run: `ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib ./target/debug/identity_test <img1> [<img2>]`

### 4.3 `src/identity/embeddings.rs` — metrics ✅

`cosine_similarity()` — working. `compute_embeddings()` — stub M1 (FaceEngine adapter for batch processing of a reference folder).

### 4.4 `src/identity/threed.rs` — FLAME/SMPL ⬜ (M3/M4)

`fit_face()` / `fit_body()` return an explicit error. The FLAME fit is the second fusion channel; SMPL is the figure check for wide shots.

### 4.5 Text channel (prototype, outside the tree)

Dense appearance description (33 fields) via a local Ollama vision model (`qwen3.6-a3b:vision-q8`) — manually verified: description → reverse verification match 1.0. Role: filling `avatar.json.identity`, prompt blocks, judge checklist. **Does not replace** the numeric lock — complements it. Wiring into the server (`identity/textpass.rs`) — not started.

---

## 5. Domain (`src/domain/`) — ✅ types in place

| File | Contents |
| --- | --- |
| `avatar.rs` | `Avatar` (schema_version, 33-field identity, persona, voice, prompt_blocks, refs), `Consent` (self/third_party/fictional — hard gate), `IdentityModel` (embeddings + face_model + body_model + per-avatar threshold) |
| `project.rs` | `Brief` (cast ≥1, format: photo/portrait_set/video_short/video_long/film, platform, language — default en, budget_cap_usd, series_id), `ProjectState` (DRAFT→…→DONE) |
| `script.rs` | `Script` (title, language, platform, hook, cast, shots), `Shot` (type TALK/BROLL/SCREEN/TITLE, vo, visual, characters, speaking, framing, wardrobe, motion, est_seconds) |
| `job.rs` | `Job` (state, stage, percent, spent_usd, provider_requests for resumability) |
| `cost.rs` | `LedgerEntry` (provider, model, units, estimated vs actual), `Estimate`/`EstimateLine` |
| `series.rs` | `Series` (shared cast, style_rules, series-wide budget, episodes) |

---

## 6. Configs (`config/`) — ✅

- **`models.yaml`**: `defaults` (tier, budget_cap_usd, max_concurrency) + 8 roles × 3 tiers with placeholder models (fal/elevenlabs/anthropic) and hypothetical prices; the `m0_results` section (watermark_check, moderation_threshold, blind_test_winner) is filled by M0. `ModelsConfig::resolve(role, tier)` — working.
- **`platforms.yaml`**: youtube_shorts (1080×1920, -14 LUFS, word-level subtitles), youtube_long, vertical_generic; `export.ai_disclosure_default: true`, `auto_publish: false`.
- **`settings.rs`**: `Settings` (default_tier, project_budget_usd, max_concurrency) + `ProviderKeys::from_env()` — keys are never logged and never written to the database.

---

## 7. State: Postgres, one contour with the agent — ✅ schema

`src/infra/store.rs`: `Store::connect(DATABASE_URL)` — sqlx pool + `migrate!("./migrations")`. Without a DSN the server refuses to start (state must not live in two places).

`migrations/0001_init.sql` — tables:

| Table | Purpose |
| --- | --- |
| `projects` | brief (JSONB), state, script (JSONB), script_hash, approved_at, budget_cap_usd |
| `shots` | (project_id, id), state, tier, retries, identity_score, output_path |
| `jobs` | state, stage, percent, spent_usd, provider_requests (JSONB) — restart-safe |
| `ledger` | every spend: provider, model, role, tier, units, estimated vs actual |
| `identity_scores` | **vector(512) + HNSW index**: embeddings and scores; "find shots with face drift above threshold" is one SQL query |
| `m0_results` | watermark, moderation_threshold, blind_test_winner per model |

The AOE agent reads these tables directly through `kb`/`database` — `cost_ledger` as an MCP tool remains a thin wrapper. Provider keys are never written to the database. Artifacts (audio/frames/clips/cache) live on disk only; the database holds paths and metadata.

---

## 8. Infrastructure (`src/infra/`)

| File | Status | Contents |
| --- | --- | --- |
| `cache.rs` | ✅ | content-addressed cache: `key(provider, model, params, input_hashes)` = SHA-256; `get/put` on the filesystem; `file_hash()` |
| `ffmpeg.rs` | ✅ partial | `check_available()` (async, ffmpeg+ffprobe in PATH) — works; `probe()` — stub M2; `FfmpegOp` enum (Trim/Concat/CaptionsAss/Loudnorm/SidechainDuck/ColorGradeGrain/NormalizeFps) — argv builders are stubs |
| `ledger.rs` | ⬜ M2 | `record()` / `spent()` — stubs; data goes to the `ledger` table |
| `store.rs` | ✅ | see §7 |

---

## 9. Providers (`src/providers/`)

- **`base.rs`** ✅ — five traits: `ImageProvider` (prompt + negative + reference images + seed), `VideoProvider` (first_frame, duration, with_audio), `AvatarVideoProvider` (frame + audio + motion_prompt), `TtsProvider` (synthesize → audio + WordTimings; clone_voice), `JudgeProvider` (judge_identity → JudgeVerdict). All async-trait, mockable.
- **`registry.rs`** ✅ skeleton — `ProviderRegistry` (models + keys + reqwest client), `entry(role, tier)` works; `provider(name)` — stub.
- **`fal.rs` / `elevenlabs.rs` / `fish.rs` / `byteplus.rs`** — empty structs with http/key fields. REST client implementation: M1–M2. Google is excluded (decision #8).

---

## 10. Pipeline (`src/pipeline/`) — ⬜ stubs tagged with milestones

| File | Milestone | What it will do |
| --- | --- | --- |
| `avatar_build.rs` | M1 | normalize_photo (sRGB, ≤2048px, EXIF strip), check_photo_quality, build_canon (sheet of ~6–8 frames with ArcFace scoring) |
| `audio.rs` | M2 | synthesize_script (TTS per shot → word timings), align_words_local (whisper fallback) |
| `keyframes.rs` | M3 | generate_keyframes (one frame per shot in the target aspect, QC-1, retries ≤2) |
| `shots.rs` | M3/M4 | render_shots (TALK→avatar model, BROLL→i2v/r2v, SCREEN/TITLE→local $0; parallel with a concurrency ceiling) |
| `qc.rs` | M3/M6 | qc_keyframe, qc_shot, qc_final — two-stage gates on the §4 core |
| `assemble.rs` | M2/M6 | assemble (ffmpeg: concat, J/L-cuts, ASS subtitles, color grade + grain, ducking, loudnorm), export_package (mp4 + metadata.json, ai_disclosure) |

---

## 11. MCP interface — ⬜ M1

`main.rs` — stub. Plan (~18 tools): avatar_* (6), project_create, series_create/status, script_submit/estimate/approve, render_preview, render_start/job_status/job_cancel/shot_regenerate, assemble, export_package, cost_ledger, providers_check. Resources: `avatar://<slug>/profile`, `project://<id>/script`, `platform://<name>`. Prompts: `write_script`, `review_storyboard`. Long operations: `render_start` → job_id → `job_status` polling (the base, works everywhere); Tasks (SEP-1686) — optional add-on.

---

## 12. Tests — ✅ 18/18

- **Unit in `scoring.rs`** (13): calibration (synthetic samples with known separability, auto-pass refusal), drift (stable / linear decline / step / outlier), assignment (K≠M, twin canons, swap), clip aggregation, fusion (training, channel conflict), source incompatibility.
- **Unit in `arcface.rs`** (2): affine roundtrip, solve6.
- **`tests/integration.rs`** (5): cache-key determinism, cosine, stage1 thresholds, each_to_own_canon (greedy version kept as a fast path).

---

## 13. Work plan — what remains

| Milestone | Contents | State |
| --- | --- | --- |
| M0 | fal+ElevenLabs keys, blind bake-off 3 TALK / 3 BROLL / 2 image / 2 TTS, watermark and moderation tests → `models.yaml` + `m0_results` | ⬜ no keys yet |
| M1 | MCP skeleton (main.rs), avatar_create→canon→voice, `compute_embeddings` adapter | ⬜ (ArcFace inference already done — §4.2) |
| M2 | audio + assemble (animatic <$0.5) + ledger in Postgres | ⬜ |
| M3 | keyframes + TALK, threshold calibration on a real face, two-character scene | ⬜ |
| M4 | BROLL + avatar in action | ⬜ |
| M5 | retries, escalation, resume after kill -9, providers_check | ⬜ |
| M6 | platform profiles, metadata.json, long 5+ min | ⬜ |
| M7 | photo/portrait_set with no separate code, film serialization | ⬜ |

---

## 14. Accepted decisions (fixed)

1. **Any avatar**: self/third_party (consent) / fictional (statement) — `consent.json` is mandatory.
2. **Default language is English**; others are explicit.
3. **Full script-driven generation**: cast, character interaction, B-roll, graphics. Not a "talking head".
4. **Budget $5…$500** — tier and cap are chosen before `script_submit`; the approval gate is mandatory.
5. **Music is optional** — absent from materials → the video is validly assembled without it.
6. **No auto-publishing** — the server exports a package; the operator uploads.
7. **AI-disclosure is on by default** in metadata.json; removal is an explicit decision.
8. **Google is excluded** (input moderation, SynthID on output). Replacements: Seedream/FLUX (canon/keyframes), Seedance/Kling/Wan (B-roll), Qwen3-VL/Claude (judge). Models with forced output marking never enter the registry.
9. **Language is Rust** (rmcp Tier-1); providers over REST (reqwest), no vendor SDKs; heavy ML is ONNX (`ort`) or API.
10. **State is Postgres** (one contour with the agent via kb/database), no SQLite; pgvector for identity scores; DSN is mandatory.
11. **ArcFace uses open buffalo_l weights** (InsightFace); the licensing question was waived by the operator; per-character fine-tuning comes later, the hook is in place (`IdentityModel`, per-avatar thresholds).
12. **The project is monolingual** — no Python; Python prototypes are forbidden.

---

## 15. Known limitations of the current code

- 11 `cargo check` warnings — dead fields in stubs (providers, ledger, ffmpeg argv).
- `each_to_own_canon` (greedy version) is kept next to the Hungarian `assign_faces_to_canons` — use the Hungarian one.
- `models/` (~370 MB) is not in git — re-download when moving machines (source: HuggingFace ApacheOne/insightface buffalo_l).
- `identity_test` requires `ORT_DYLIB_PATH` — the dylib is not bundled; resolve for production (bundle or system path).
- The text appearance-description channel (Ollama) is a prototype in `/tmp/idtest/`, not wired into the server.
- `Ksenia/` is the test asset set (22 photos, image_01..image_22) for the first full avatar_create run.
