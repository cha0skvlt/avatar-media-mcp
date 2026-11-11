# avatar-media-mcp — code architecture

> **Rules for the agent editing this file**
> 1. Write only what the code confirms. Every claim must be checkable in the tree; when code and this file disagree, the code wins.
> 2. Write only what exists in the project. Never name a library, model, provider, transport, file or feature that is not in the tree — not even to say it was removed or rejected.
> 3. Code never references documentation. No `.md` path, section number or decision number in a comment, a string, a config or a migration.
> 4. No changelog, no history, no "previously", no "no longer". This file describes the tree as it is today.

> Status: **map of the code as it exists now** · verified against the tree: **2026-11-12**
> Renewal rule: on any doc change, re-verify claims against the code and update both fields.

This file maps the `avatar-media-mcp` crate — what is on disk and what the code actually does. The code is the source of truth; when this file and the code disagree, fix this file, never bend the code to match a document.

Markers:

- ✅ implemented and exercised
- ◐ partial — some functions real, some stubs in the same file
- ⬜ stub — an explicit `bail!`, tagged with its milestone

---

## 1. What this is

`avatar-media-mcp` is a Rust MCP server (cargo crate, edition 2024) — the hands of the CHA0S AOE IDE agent for generative media with avatars. The agent is the brain (script, dialogue, approvals); the server executes everything that costs money or touches files: stores avatars, generates voiceover/frames/clips through external models, assembles the final cut through ffmpeg, tracks spend, and refuses to exceed the budget.

**State today:** the identity core (mathematics + local ArcFace inference + reference clustering + the text-channel contract) works and is tested; the face-swap pipeline (§10, FS1) works; `avatar_create` is a live MCP tool that produces `avatar.json` + an identity model from a photo set; the data skeleton (domain types, configs, Postgres schema, cache, ffmpeg availability check) is in place. The generation pipeline — TTS, keyframes, TALK, BROLL, assembly — is stubbed at M2–M6, and 14 of the 17 registered MCP tools answer with an explicit "planned for M<n>".

The engineering idea everything rests on:

> **The face lives in a reference frame, the voice lives in one cloned voiceover, motion comes from the cheapest model that can handle a given shot, and assembly is deterministic ffmpeg, not a neural network.**

---

## 2. Repository tree

```
avatar-media-mcp/                # the crate — 9092 lines of Rust
├── Cargo.toml                   # rmcp 0.3, tokio, tracing, serde+schemars, reqwest,
│                                # sqlx(postgres), image, sha2/hex, chrono, serde_yaml,
│                                # figment, futures; ort 2.0-rc.15 (load-dynamic) +
│                                # ndarray under `local-identity`
├── README.md                    # build, CLI entry points, AOE wiring
├── ARCHITECTURE.md              # this file — the code map
├── .env.example                 # FAL_KEY, ELEVENLABS_API_KEY, DATABASE_URL + optional keys
├── .gitignore                   # target/, .env, avatars/, models/, editor noise
├── config/
│   ├── models.yaml              # 8 roles × tiers → {provider, model_id, params, price}
│   ├── platforms.yaml           # 3 platform profiles + export rules
│   └── settings.yaml            # runtime knobs: tier, budget, concurrency
├── migrations/
│   ├── 0001_init.sql            # 6 tables (§7)
│   └── 0002_integrity.sql       # unique request_id, identity_scores FK, ON DELETE CASCADE
├── models/                      # insightface/buffalo_l, faceswap, ort/lib — the weights
├── avatars/                     # <slug>/{avatar.json, consent.json, identity_model.json, refs/}
├── examples/
│   └── pairwise.rs              #  54  cosine distribution over an identity_model.json
├── src/
│   ├── lib.rs                   #  10  module roots
│   ├── main.rs                  # 285  rmcp stdio server: 17 tools, 3 live
│   ├── bin/
│   │   ├── avatar_build.rs      # 198  CLI build: normalize → embed → dedupe → AvatarStore
│   │   ├── faceswap.rs          #  68  FS1 CLI
│   │   ├── faceswap_diag.rs     #  28  inswapper aligned-crop diagnostic
│   │   └── identity_test.rs     #  39  embed → cosine → stage1 verdict
│   ├── config/                  # 385  models.rs, platforms.rs, settings.rs (figment loader)
│   ├── domain/                  # 380  avatar, project, script, job, cost, series
│   ├── identity/                # 3526 scoring 1769, arcface 629, textpass 492, clusters 336,
│   │                            #      embeddings 171, threed 120
│   ├── providers/               # 176  base 87, registry 40, fal/elevenlabs/fish/byteplus 37
│   ├── pipeline/                # 1674 faceswap 1296, qc 160, avatar_build 134, assemble 24,
│   │                            #      audio 19, keyframes 14, shots 14
│   ├── infra/                   # 983  avatar_store 532, cache 106, ledger 96, store 50,
│   │                            #      image_io 69, paths 62, ffmpeg 59
│   ├── tools/                   # 552  avatar.rs 392 — `avatar_create`; health.rs 155 — preflight
│   └── series.rs                #  16  M7 stubs
└── tests/
    ├── integration.rs           #  50  5 tests
    ├── smoke.rs                 # 482  19 tests — public surface, hermetic
    ├── db.rs                    #  92  1 test — skipped unless DATABASE_URL
    └── e2e_models.rs            #  94  1 test — #[ignore], needs weights + photos
```

Counters: `cargo test --features local-identity` → **111/111 green** (86 lib unit + 5 integration + 19 smoke + 1 db), plus **1 ignored** (`e2e_models`). `cargo check --features local-identity --all-targets` → **0 warnings**. `cargo clippy --all-targets` → **0 warnings**.

Runtime path rule: the server resolves `models/insightface/buffalo_l`, `models/faceswap` and writes `avatars/<slug>/` **relative to the process cwd**. It must be launched with `cwd = avatar-media-mcp`.

---

## 3. Wiring into CHA0S AOE

The MCP stdio server is registered as a `context_servers` entry in `~/.config/cha0s/settings.json`:

```jsonc
"context_servers": {
  "avatar-media": {
    "command": "/abs/path/to/avatar-media-mcp",
    "args": [],
    "env": { "FAL_KEY": "...", "ELEVENLABS_API_KEY": "...", "DATABASE_URL": "..." }
  }
}
```

Zero changes to AOE code. Provider keys travel through the **process env**: `ProviderKeys::from_env()` reads eight keys from the `context_servers.env` block, empty strings treated as absent. Keys are never logged and never written to the database.

`main.rs` opens Postgres lazily: `Store::connect` (§7) is called by the stateful tools (M2).

---

## 4. Identity core — ✅ working

Pure functions, no I/O; newtype wrappers (`CosineScore`, `GeometryDistance`, `EmbeddingSource`) make mixing incompatible units a compile error.

### 4.1 `src/identity/scoring.rs` (1769 lines) — verdicts and thresholds

| API | What it does |
| --- | --- |
| `IdentityThresholds` | pass/fail thresholds **per embedding source** (ArcFaceOnnx / VisionFal / LegacyUnspecified — different vector spaces; mixing is an error); `for_source()`, `with_avatar_pass_threshold()` (per-avatar calibration), `validate_source()` |
| `stage1()` / `stage1_score()` / `stage1_embedding()` | QC stage 1: Pass / Borderline (→ judge) / Fail |
| `calibrate_thresholds()` | Bayesian calibration from positive/negative samples: Jeffreys prior (Beta), FPR upper bound via a self-contained incomplete beta + Lanczos lgamma. **Key decision: on small samples, refuse auto-pass** — if FPR ≤ 1% is not provable, the gate goes to the judge rather than being trusted. An empty grey zone (fail == pass) is a valid outcome |
| `analyze_drift()` | drift model over a shot series: Theil–Sen slope (≥3 points — on 2 points the slope is just the endpoint difference and is not computed), robust medians for baseline/recent, step detection, single-outlier isolation, view-angle normalization (`FaceViewAngle`). Verdicts: Stable / DriftStarting / DriftCritical |
| `assign_faces_to_canons()` | multi-character assignment: **Hungarian algorithm** (O(n³), own implementation) over the cosine matrix; dummy rows for K≠M; per-face thresholds; anomalies: ExtraFace / MissingCharacter / TwinCanons |
| `aggregate_clip_frames()` | QC-2 for a clip: median = base level, intra-clip drift tracked separately (endpoint drop + slope); dropping an artifact frame does not produce a false drift |
| `fit_fusion_model()` / `fuse_identity_signals()` | channel fusion (embedding + FLAME geometry): logistic regression with view-angle interaction terms (weights are not hardcoded); channel conflict → Borderline (to the judge), never averaging; works without geometry with no API degradation |
| `each_to_own_canon()` | greedy fast path kept next to the Hungarian assignment; the server calls the Hungarian one |

### 4.2 `src/identity/arcface.rs` (629 lines) — local inference ✅

The full InsightFace pipeline in pure Rust:

1. **SCRFD** (`det_10g.onnx`) — face detection + 5 landmarks on a 640px letterbox, threshold 0.5, best score wins; face selection by inter-ocular distance (main person first).
2. **Alignment** — least-squares similarity transform onto the canonical ArcFace 112×112 template: own implementation (solve6 — 6×6 Gaussian elimination with partial pivoting; 2×3 affine inversion; bilinear sampling).
3. **ArcFace** (`w600k_r50.onnx`) — 512-d vector, L2-normalized (cosine = dot product).

Public surface: `FaceEngine::load(dir)`, `embed_image(path) -> Vec<arcface::FaceEmbedding>` (vector + 5 landmarks), `detect_faces(path) -> Vec<DetectedFaceBox>` (bbox, kps, score).

Inference via `ort` (ONNX Runtime 2.0-rc.15, load-dynamic, dylib in `models/ort/lib/`). **Critical:** `AutoDevicePolicy::PreferCPU` is mandatory — the CoreML EP on macOS breaks SCRFD's dynamic outputs (rank mismatch). CPU is sufficient (~50 ms/face).

Verified on real photos: self-similarity 1.0000; different frames of the same person → 0.8554 → PASS.

Run: `ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib ./target/debug/identity_test <img1> [<img2>]`

### 4.3 `src/identity/embeddings.rs` (171 lines) — metrics + batch adapter ✅

- `EmbeddingSource` (`ArcFaceOnnx` / `VisionFal` / `LegacyUnspecified`), `CosineScore` (bounded, validated), `FaceEmbedding { vector, source }`.
- `cosine_similarity_checked()` — f64 accumulation, rejects empty/NaN/zero-norm/dimension mismatch. `cosine_similarity()` is the untyped compatibility wrapper (returns 0.0 on error).
- `compare_embeddings()` — refuses to compare different encoder spaces.
- `compute_embeddings(&[PathBuf]) -> (BTreeMap<path, Vec<f32>>, skipped)` — loads its own `FaceEngine`, embeds the **main face** of each image, reports images without a face in `skipped` instead of failing. Errors only when no image yields a face.

### 4.4 `src/identity/threed.rs` (120 lines) — FLAME/SMPL ⬜ (M3/M4)

`GeometryModel` (Flame/Smpl/SmplX), `GeometryDistance` (non-negative RMS in one model's parameter space), typed parameter validation. The FLAME fit is the second fusion channel, SMPL the figure check for wide shots.

### 4.5 Text channel — ArcFace-first identity pipeline ◐ (implemented, agent transport only)

The numeric lock is built first and governs the text; the text never precedes the numbers.

Four phases, in fixed order:

1. **Numeric lock (mandatory, first).** `photos → FaceEngine → 512-d vectors → clustering → IdentityModel`. Implemented in `identity/clusters.rs` + `identity/embeddings.rs`, driven by `tools/avatar.rs`.
2. **Grounded text description.** A vision channel fills the `Identity` struct (§5) from (a) one representative image per cluster and (b) a `NumericSummary` (cluster count, per-cluster member count and min intra-cluster cosine, inter-cluster mean, per-avatar threshold). The channel must cover every cluster.
3. **Two transports, one contract.** `trait TextChannel { describe, verify }`. `AgentChannel` (default) cannot block on the operator: `describe` fails with the exact prompt + numeric summary, the agent fills `identity_json` on the follow-up `avatar_create` call, and `verify` fails with the verification round-trip — the server never verifies on the agent's behalf. `CloudChannel` (fal vision) bails until M0/M2 keys exist. `channel_order()` maps `identity.text_channel` → `["agent","cloud"]` (default) or `["cloud","agent"]`.
4. **Closed-loop validation.** `grounded_description()` — describe → verify → regenerate ≤2 (`MAX_REGENERATIONS`), pass at `VERIFY_PASS_THRESHOLD = 0.8`, then flag. `filter_do_not_change()` bars the variable traits (hair, beard, makeup, glasses) when the set has more than one appearance state.

`clusters.rs` — greedy centroid clustering over cosine, seeded from the most central reference so the result is independent of input order:

- `DEFAULT_CLUSTER_THRESHOLD = 0.85`; `auto_threshold()` sweeps 12 candidates (0.50…0.97) and maximizes `inter_cluster_mean − intra_cluster_mean`, tie-breaking toward fewer clusters.
- `cluster_references()` returns per-cluster members, L2-normalized centroid, representative (closest to centroid), min intra-cluster cosine (NaN for singletons), plus `inter_cluster_mean`, `intra_cluster_mean`, and `avatar_threshold = clamp(intra_mean − 0.04, 0.55, 0.93)` (`None` when the spread is unknown).

`textpass.rs` also carries `base64_encode()` (RFC 4648) for the cloud transport, and `parse_agent_identity()` which tolerates JSON wrapped in prose.

Status: `avatar_create` drives the agent transport with `describe` + `parse_agent_verification`, then renders `prompt_blocks` with `build_prompt_blocks()`: the identity block is a deterministic rendering of the filled fields, `do_not_change` carries only cluster-stable traits, and the variable traits are dropped when the set has more than one appearance state.

---

## 5. Domain (`src/domain/`) — ✅ types in place

| File | Contents |
| --- | --- |
| `avatar.rs` | `Avatar` (schema_version, slug, display_name, consent_ref, identity, persona, voice, prompt_blocks, refs, created_at, canon_approved_at), `Identity`, `Hair`, `Persona`, `VoiceRef`, `PromptBlocks` (+`RefBinding`), `Refs`, `Consent` (+`ConsentSubject`: self / third_party / fictional — hard gate), `IdentityModel` |
| `project.rs` | `Brief` (cast ≥1, format: photo/portrait_set/video_short/video_long/film, platform, language, goal, budget_cap_usd, series_id, materials), `ProjectState` (Draft→ScriptSubmitted→Estimated→Approved→Rendering→Assembled→Done) |
| `script.rs` | `Script` (title, language, platform, hook, cast, shots), `Shot` (id, type TALK/BROLL/SCREEN/TITLE, vo, visual, characters, speaking, framing CU/MCU/MS/WS, wardrobe, motion, est_seconds) |
| `job.rs` | `Job` (id, project_id, state, stage, percent, spent_usd, provider_requests for resumability, error, timestamps) |
| `cost.rs` | `LedgerEntry` (provider, model, role, tier, units, unit_name, estimated vs actual, request_id), `EstimateTier` |
| `series.rs` | `Series` (shared cast, style_rules, series-wide budget, episodes) |

`Identity` is **14 leaf fields**: `apparent_age_range`, `gender_presentation`, `face_shape`, `skin_tone`, `eyes`, `brows`, `nose`, `lips`, `hair.{color,length,style}`, `facial_hair`, `distinguishing_marks`, `build`.

`IdentityModel` = `face_embeddings: BTreeMap<path, Vec<f32>>`, `face_model`, `body_model`, `identity_threshold`.

Path rule: every path inside `avatar.json` and `identity_model.json` is **relative to the avatar directory** (`refs/01.jpg`, `consent.json`). `AvatarStore` (§8) enforces it on write and resolves it on read.

---

## 6. Configs (`config/`) — ✅

- **`models.yaml`**: `defaults` (tier=standard) + 8 roles (`avatar_canon`, `keyframe`, `talk`, `broll`, `broll_reference`, `tts`, `word_timing`, `qc_judge`) with placeholder models (fal/elevenlabs/anthropic) and hypothetical prices; `avatar_canon` defines `standard` and `premium`. The `m0_results` section (watermark_check, moderation_threshold, blind_test_winner) is empty until M0. `ModelsConfig::resolve(role, tier)` — working.
- **`platforms.yaml`**: youtube_shorts (1080×1920, 30 fps, ≤180 s, −14 LUFS, word-level ASS, safe area 0.12/0.18), youtube_long (1920×1080, chapters from script), vertical_generic (1080×1920, ≤90 s); `export.ai_disclosure_default: true`, `auto_publish: false`.
- **`settings.yaml` + `settings.rs`**: the three runtime knobs (`default_tier`, `project_budget_usd`, `max_concurrency`) have exactly one home. `Settings::load(path)` = defaults → optional YAML → `CHA0S_MEDIA_*` env, then `validate()` (budget > 0 and finite, concurrency ≥ 1). Budget and concurrency live only in `settings.yaml`. `ProviderKeys::from_env()` reads eight keys from the process env.
- **Loaders**: `ModelsConfig::from_path` / `from_yaml_str`, `PlatformsConfig::from_path` / `from_yaml_str`, `Settings::from_path` / `load`. `ModelsConfig::resolve` falls back to `defaults.tier` when a role has no entry for the requested tier (`avatar_canon` defines `standard` and `premium`).

---

## 7. State: Postgres, one contour with the agent — ✅ schema

`src/infra/store.rs`: `Store::connect(DATABASE_URL)` — sqlx pool (max 10) + `migrate!("./migrations")`; `health()`, `spent_usd()`.

`migrations/0001_init.sql` — tables:

| Table | Purpose |
| --- | --- |
| `projects` | brief (JSONB), state, script (JSONB), script_hash, approved_at, budget_cap_usd |
| `shots` | (project_id, id), script_hash, state, tier, retries, identity_score, output_path |
| `jobs` | state, stage, percent, spent_usd, provider_requests (JSONB) — restart-safe |
| `ledger` | every spend: provider, model, role, tier, units, estimated vs actual; index on project_id; unique `request_id` |
| `identity_scores` | **vector(512) + HNSW index** (pgvector): embeddings and scores; "find shots with face drift above threshold" is one SQL query |
| `m0_results` | watermark, moderation_threshold, blind_test_winner per model |

The AOE agent reads these tables directly through `kb`/`database`; the `cost_ledger` MCP tool is a stub at M2. Provider keys are never written to the database. Artifacts (audio/frames/clips/cache) live on disk only; the database holds paths and metadata.

---

## 8. Infrastructure (`src/infra/`)

| File | Status | Contents |
| --- | --- | --- |
| `avatar_store.rs` | ✅ | the only avatar writer/reader: `save` refuses absolute or missing refs and a primary outside `refs.set`; `load` refuses a directory whose refs disappeared; `list`; atomic writes. Layout: `avatar.json`, `consent.json`, `identity_model.json`, `refs/NN.jpg` |
| `image_io.rs` | ✅ | `open_limited` / `open_rgb8`: every decode is bounded (≤12000 px side, ≤512 MB) — a decompression bomb is an error |
| `paths.rs` | ✅ | `confine_output`: tool writes must resolve inside the workspace; reads stay unrestricted |
| `cache.rs` | ✅ | content-addressed cache: `key(provider, model, params, input_hashes)` = SHA-256 over length-prefixed fields (no boundary collisions); `put` writes a temp file and renames (a crash cannot leave a truncated hit); `get`, `file_hash()` |
| `ffmpeg.rs` | ◐ | `check_available()` (async, ffmpeg+ffprobe in PATH) ✅; `probe()` bails at M2; `FfmpegOp` enum (Trim/Concat/CaptionsAss/Loudnorm/SidechainDuck/ColorGradeGrain/NormalizeFps) — `argv()` returns `Result` and bails until the M2 builders land |
| `ledger.rs` | ✅ | `record(entry, cap)` locks the project row, sums the ledger and inserts in one transaction — the cap is enforced under concurrency; `spent()`; data goes to the `ledger` table |
| `store.rs` | ✅ | see §7 |

---

## 9. Providers (`src/providers/`)

- **`base.rs`** ✅ — five traits: `ImageProvider` (prompt + negative + reference images + w/h + seed), `VideoProvider` (prompt, first_frame, duration, with_audio), `AvatarVideoProvider` (frame + audio + motion_prompt), `TtsProvider` (synthesize → audio + `WordTiming`; clone_voice), `JudgeProvider` (judge_identity → `JudgeVerdict`). All async-trait.
- **`registry.rs`** ◐ — `ProviderRegistry` (models + keys + reqwest client), `entry(role, tier)` works; `provider(name)` bails until the M1/M2 constructors land.
- **`fal.rs` / `elevenlabs.rs` / `fish.rs` / `byteplus.rs`** — structs with `http` + `key` fields; REST client implementation: M1–M2.

---

## 10. Pipeline (`src/pipeline/`) — ✅ faceswap (FS1); the rest stubs M2–M6

| File | Milestone | State |
| --- | --- | --- |
| `avatar_build.rs` | M1/M3 | ◐ `normalize_photo` (≤2048 px long side, JPEG q95 or PNG, metadata stripped) ✅; `check_photo` / `check_photo_quality` (min 256 px, mean brightness 40…235) ✅; `build_canon` bails at M3 |
| `audio.rs` | M2 | ⬜ `synthesize_script`, `align_words_local` — `bail!` |
| `keyframes.rs` | M3 | ⬜ `generate_keyframes` — `bail!` |
| `shots.rs` | M3/M4 | ⬜ `render_shots` — `bail!` |
| `qc.rs` | M3/M6 | ◐ pure gates ✅: `qc1_identity_gate`, `qc1_multi_character_gate`, `qc2_identity_gate`, `qc1_score_gate` (return `QcAction::AutoPass/AutoFail/VisionJudge`); async adapters `qc_keyframe`, `qc_shot`, `qc_final` bail — they need frame extraction + judge providers |
| `assemble.rs` | M2/M6 | ⬜ `assemble` (M2), `export_package` (M6) — `bail!` |
| `faceswap.rs` | ✅ FS1 | fast face swap, pure Rust `ort`: SCRFD detect + `2d106det` dense landmarks, `inswapper_128` (aligned 128 crop + source ArcFace embedding through the `emap` buff2fs matrix), Reinhard per-channel color transfer, optional GPEN bfr-512 restore (blend 0.8), landmark face-contour paste mask with open-mouth keep hole, inverse-affine composite. `SwapEngine::swap_face(target, bbox, kps5, emb)`; helpers `umeyama_similarity`, `warp_sample`, `reinhard_color_match`, `convex_hull`, `mouth_aspect_ratio`, `face_contour_mask`, `mouth_keep_mask`, `ellipse_fallback_mask`. Weights in `models/faceswap/` (`2d106det.onnx`, `inswapper_128.onnx`, `inswapper_emap.f32`, optional `gpen_bfr_512.onnx`) |

---

## 11. MCP interface — ◐ M1

`main.rs` — rmcp stdio server, `ServerCapabilities::enable_tools()`. Engines load lazily on the first call and are cached in `Arc<Mutex<Option<…>>>`; a failed load leaves `None` so the next call retries.

| Tool | State | Notes |
| --- | --- | --- |
| `faceswap` | ✅ | `source_path`, `target_path`, `out_path` (confined to the workspace); swaps exactly one face (best-scoring detection); 120 s timeout |
| `avatar_create` | ✅ | `photos` (1–30), `slug`, `consent`, `text_channel`, `identity_json`, `verification_json`, `force`; three-call flow; 900 s timeout |
| `providers_check` | ✅ | preflight: config files, key presence, ORT dylib, model dirs, ffmpeg, avatars root, DATABASE_URL — reads config and env only |
| `avatar_list`, `avatar_status`, `avatar_approve_canon` | ⬜ M1 | |
| `project_create`, `script_submit` | ⬜ M1 | |
| `script_estimate`, `script_approve`, `assemble`, `cost_ledger` | ⬜ M2 | |
| `render_start`, `job_status` | ⬜ M3 | |
| `job_cancel`, `shot_regenerate` | ⬜ M5 | |
| `export_package` | ⬜ M6 | |

Stubs answer with `invalid_params` and the milestone they belong to — the contour is visible to the agent.

**Face swap is agent-callable two ways.** The MCP tool above, and the CLI `src/bin/faceswap.rs` (`<face> <target> <out.png>`). Same crate, same `SwapEngine`, same weights — both wrap `SwapEngine::swap_face` unchanged.

**`avatar_create` flow** (`tools/avatar.rs`): slug validation → approved-canon guard (`force`) → consent gate (third_party parses and checks `consent.json`) → `normalize_photo` into a **staging** `refs/NN.jpg` → per-photo quality gate → `compute_embeddings` → near-duplicate removal (cosine > 0.9999) → `auto_threshold` → `cluster_references` (medoid-ordered) → `NumericSummary` → channel loop → **verification** → `build_prompt_blocks` → `Staging::commit` (swap into place) → report (clusters, representatives, threshold, skipped/duplicate images, channel used, verdict, flag, warnings). A failed rebuild leaves the live avatar untouched. Blocking ONNX work runs inside `spawn_blocking` under a timeout; the async channel calls are driven by `futures::executor::block_on`.

**Verification is a second round-trip.** With `identity_json` but no `verification_json`, `avatar_create` returns the verification prompt (description + the same representatives) and writes nothing. The agent answers `{"match_score": 0..1, "mismatches": [...]}`; `parse_agent_verification` rejects anything outside `[0, 1]`; `verification_flagged = match_score < 0.8`. `AgentChannel::verify` fails with that round-trip request.

---

## 12. Tests — ✅ 111/111 (+1 ignored)

`cargo test --features local-identity` → 86 lib unit + 5 integration + 19 smoke + 1 db, all green; `e2e_models` is `#[ignore]`d. The default suite is hermetic: pure functions and temp directories. The `db` test skips itself when `DATABASE_URL` is unset.

- **`pipeline/faceswap.rs`** (18): affine invert/scale, Umeyama (similarity, identity, isotropic covariance), Reinhard (noop, mean shift), hull/fill/masks, mouth MAR and keep hole, ellipse fallback, warp identity/translation, paste mask and forward-sampler round-trip, unusable-landmark contour returns an empty mask.
- **`identity/scoring.rs`** (13): calibration (known separability, auto-pass refusal, disabled-fail flag consistency), drift (stable / linear decline / step / outlier, angle offsets), assignment (K≠M, twin canons, swap), clip aggregation, fusion (training, channel conflict, source mismatch), per-avatar pass threshold cannot invert the gate.
- **`identity/textpass.rs`** (7): base64 RFC 4648 vectors, wrapped-JSON parsing, `do_not_change` filtering, agent describe round-trip, agent verify refuses to verify on the agent's behalf, verification payload validation (`match_score` outside `[0,1]` rejected), prompt-block rendering (empty identity stays empty, hair dropped on multi-state).
- **`identity/arcface.rs`** (5): affine roundtrip, solve6, bbox/SCRFD component indexing, ambiguous-shape rejection, IoU.
- **`identity/clusters.rs`** (6): two appearance states, single state, empty input, closest-cluster assignment, invalid vectors rejected, order-independence (reversed input → same clusters).
- **`tools/avatar.rs`** (5): slug traversal rejected, near-duplicate removal, consent gate before I/O, third-party consent file required, photo count bounded.
- **`infra/avatar_store.rs`** (12): save/load roundtrip, absolute refs refused, missing ref refused on save, primary must be in the set, a ref that disappeared is reported on load, missing model file is an error, slug traversal refused, staging commit replaces the live avatar, dropped staging leaves it untouched, schema_version mismatch refused, slug mismatch refused, consent parsed not just present.
- **`infra/image_io.rs`** (3): oversized image refused, normal photo accepted, missing file is an error.
- **`infra/paths.rs`** (3): output inside the workspace allowed, outside refused, traversal refused.
- **`config/settings.rs`** (4): defaults valid, file overrides defaults, missing file falls back, zero/NaN budget and zero concurrency refused.
- **`infra/cache.rs`** (2): field boundaries do not collide, atomic put leaves no temp file.
- **`config/models.rs`** (2): tier fallback, flattened price spec.
- **`config/platforms.rs`** (1): export rules conservative, shorts profile parsed from the real file.
- **`pipeline/avatar_build.rs`** (2): quality remarks for small/dark photos, normalize caps the long side at 2048.
- **`identity/embeddings.rs`** (2): checked cosine rejects invalid vectors, source mismatch is an error.
- **`identity/threed.rs`** (1): geometry parameter validation.
- **`tests/integration.rs`** (5): cache-key determinism (×2), cosine of identical, stage1 boundaries, `each_to_own_canon` swap detection.
- **`tests/smoke.rs`** (19): config files parse and resolve, settings load from the real file, canonical avatar layout is self-contained (save → load → refs resolve), verification round-trip is explicit, migration declares the six tables + pgvector, `avatar.json`/`IdentityModel` roundtrip, consent gate and slug traversal refused before I/O, every stub returns an error instead of panicking, ffmpeg availability reported, identity core determinism, clustering, text-channel contract, cache roundtrip, QC action mapping, empty identity is a valid document, parsers never panic on 500 pseudo-random payloads, preflight reports the real configuration, the per-avatar threshold reaches the QC gate.
- **`tests/db.rs`** (1, skipped without `DATABASE_URL`): ledger cap enforced inside a transaction, duplicate `request_id` rejected, unknown project refused, `ON DELETE CASCADE` removes ledger rows.
- **`tests/e2e_models.rs`** (1, `#[ignore]`): the full three-call `avatar_create` on real photos — round-trip 1, round-trip 2, write, then `AvatarStore::load` with every ref inside the avatar directory.

---

## 13. Milestones — where the stubs sit

Every row names symbols that exist in the tree; the milestone is the tag their `bail!` carries.

| Milestone | Stubbed at this milestone | State |
| --- | --- | --- |
| M0 | `models.yaml` prices and `m0_results` hold placeholder data | ⬜ |
| M1 | `avatar_list`, `avatar_status`, `avatar_approve_canon`, `project_create`, `script_submit`, `build_canon`, `TtsProvider::clone_voice`, `CloudChannel` | ◐ `avatar_create` live |
| M2 | `audio::synthesize_script`, `audio::align_words_local`, `assemble::assemble`, `script_estimate`, `script_approve`, `cost_ledger` | ◐ `ledger` live |
| M3 | `keyframes::generate_keyframes`, `shots::render_shots` (TALK), `qc::qc_keyframe`, `qc::qc_shot`, `render_start`, `job_status` | ⬜ |
| M4 | `shots::render_shots` (BROLL) | ⬜ |
| M5 | `job_cancel`, `shot_regenerate` | ⬜ |
| M6 | `assemble::export_package`, `qc::qc_final` | ⬜ |
| M7 | `series::create_series`, `series::status` | ⬜ |

## 14. Accepted decisions (fixed)

1. **Any avatar**: self/third_party (consent) / fictional (statement) — `consent.json` is mandatory.
2. **Default language is English**; others are explicit.
3. **Full script-driven generation**: cast, character interaction, B-roll, graphics. Not a "talking head".
4. **Budget $5…$500** — tier and cap are chosen before `script_submit`; the approval gate is mandatory.
5. **Music is optional** — absent from materials → the video is validly assembled without it.
6. **No auto-publishing** — the server exports a package; the operator uploads.
7. **AI-disclosure is on by default** in metadata.json; removal is an explicit decision.
8. **The registry carries fal, elevenlabs, fish, byteplus and anthropic models** — one provider per role, priced per unit.
9. **Language is Rust** (rmcp Tier-1); providers over REST (reqwest); heavy ML is ONNX (`ort`) or API.
10. **State is Postgres** (one contour with the agent via kb/database); pgvector for identity scores; DSN is mandatory once a stateful tool lands.
11. **ArcFace uses open buffalo_l weights** (InsightFace); the per-character hook is `IdentityModel` + per-avatar thresholds.
12. **The project is monolingual — Rust.**
13. **ArcFace-first identity pipeline** (§4.5): the numeric lock is built first and grounds the text; the text description is produced by a vision channel over cluster artifacts and validated by reverse verification; it never precedes the numbers. **Transports**: the multimodal agent (default, round-trip via `identity_json` + `verification_json`) and cloud vision (M0/M2).

---

## 15. Properties of the current code

- `AvatarStore` reads and writes one layout: `avatar.json`, `consent.json`, `identity_model.json`, `refs/NN.jpg`; every path inside the JSON is relative to the avatar directory.
- Two `FaceEmbedding` types: `identity::arcface::FaceEmbedding { vector, landmarks }` and `identity::embeddings::FaceEmbedding { vector, source }`; callers convert explicitly.
- `each_to_own_canon` (greedy) sits next to the Hungarian `assign_faces_to_canons`; the server uses the Hungarian one.
- `models/` holds ~370 MB of weights — buffalo_l, faceswap, ort/lib — downloaded per machine from HuggingFace `ApacheOne/insightface`.
- `identity_test`, `faceswap`, `faceswap_diag`, `avatar_build` and the MCP server read `ORT_DYLIB_PATH`; the dylib lives in `models/ort/lib/`.
- faceswap swaps exactly one face — the best-scoring detection (`detections[0]`).
- `avatar_create` accepts 1–30 photos and embeds the main face of each.
- `normalize_photo` decodes to RGB8 and treats the pixels as sRGB.
- `avatar_create` loads its own `FaceEngine` through `compute_embeddings`; the server's cached engine serves `faceswap`.
- `tokio::time::timeout` returns control to the agent at 120 s (`faceswap`) and 900 s (`avatar_create`); the blocking ONNX task runs to completion.
- `qc::thresholds_for_avatar` applies `IdentityModel.identity_threshold` to the encoder space.
- `avatar_create` and the CLI resolve `models/` and `avatars/` relative to the process cwd.
