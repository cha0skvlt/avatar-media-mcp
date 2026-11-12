# Integration plan — adk-rust-mcp-toolkit

> Source: `../_incoming/adk-rust-mcp-toolkit-main` (Apache-2.0, 11 crates, ~30k lines).
> Target: this crate. Status: **plan, nothing ported yet**.

## 1. Principle: capabilities, not vendors

The server is provider-neutral. Pipeline code calls the traits in `src/providers/base.rs` and never names a vendor; the registry resolves `role → tier → provider` from `config/models.yaml` and constructs the adapter. Adding a provider means adding one file that implements a trait — no pipeline change.

The toolkit is useful as a **reference implementation of the shape**, not as a provider choice: its crates show how a long-running job is polled, how a generation request is shaped, how TTS parameters are validated. The vendor payloads, auth and model ids stay behind.

## 2. What we take and what we leave

| Toolkit crate | Lines | Take (provider-independent) | Leave (vendor-bound) |
| --- | --- | --- | --- |
| `adk-rust-mcp-avtool` | 2415 | `run_ffprobe` + `MediaInfo`/`StreamInfo`, argv builders for concat / combine / overlay / volume, `VolumeValue` parsing, stderr in errors | GCS input/output, tool schemas |
| `adk-rust-mcp-video` | 3171 | LRO polling loop (`poll_lro`), backoff constants, operation-status error extraction | request payloads, auth, model ids |
| `adk-rust-mcp-image` | 2371 | request shape: prompt, negative, aspect ratio, seed, reference images, save bytes | request payloads, auth |
| `adk-rust-mcp-speech` | 1764 | TTS parameter surface: voice, language, rate, pitch, pronunciations, output file; `list_voices` | request payloads, auth |
| `adk-rust-mcp-common` | 3993 | error taxonomy: `api(endpoint, status, body)`, `timeout(secs)`, `ffmpeg(stderr)`, `validation` | transport, config, GCS, OTel |
| `composer`, `graphics`, `diagrams`, `artist`, `education`, `music` | ~1500 | — | all of it |

What the toolkit does not have and we keep writing ourselves: word timings, identity QC, canon, budget, multi-character assignment.

## 3. Waves

Each wave is independently shippable, ends with `cargo test --features local-identity` green and `cargo clippy --all-targets` at zero warnings, and touches only the files listed.

### Wave 1 — ffmpeg core

**Goal:** `infra/ffmpeg.rs` stops being a stub. No provider, no network.

| From | To |
| --- | --- |
| `avtool/src/handler.rs:458` `run_ffprobe` | `src/infra/ffmpeg.rs` `probe()` |
| `avtool/src/handler.rs:37` `MediaInfo`, `:48` `StreamInfo` | `src/infra/ffmpeg.rs` same structs, `serde` |
| `avtool/src/handler.rs:758` `concatenate` | `FfmpegOp::Concat` argv |
| `avtool/src/handler.rs:650` `combine_audio_video` | `FfmpegOp::Combine` argv |
| `avtool/src/handler.rs:693` `overlay_image` | `FfmpegOp::Overlay` argv |
| `avtool/src/handler.rs:811` `adjust_volume` | `FfmpegOp::Loudnorm` argv |
| `avtool/src/handler.rs:238` `VolumeValue::parse` | `src/infra/ffmpeg.rs` volume parsing |

**Files:** `src/infra/ffmpeg.rs`, `tests/smoke.rs` (argv assertions).

**Acceptance:** `probe()` returns duration/streams for a real file; every `FfmpegOp::argv()` returns a complete argument list; ffmpeg stderr reaches the error text. Tests: argv unit tests need no ffmpeg; one `#[ignore]` test probes a generated file.

**Unblocks:** `assemble`, `export_package`, QC frame extraction.

### Wave 2 — TTS capability

**Goal:** `pipeline/audio.rs` produces audio and word timings through `TtsProvider`, with one adapter behind the registry.

| From | To |
| --- | --- |
| `speech/src/handler.rs:104` `SpeechSynthesizeParams` | `TtsRequest` in `src/providers/base.rs` |
| `speech/src/handler.rs:308` `synthesize` | `TtsProvider::synthesize` contract |
| `speech/src/handler.rs:395` `list_voices` | `TtsProvider::list_voices` |
| `speech/src/handler.rs` range validation (rate 0.25–4.0, pitch ±20) | same checks in the adapter, `bail!` |

**Files:** `src/providers/base.rs` (request/response types), `src/providers/registry.rs` (construct by name), one adapter file, `src/pipeline/audio.rs`.

**Acceptance:** `synthesize_script` writes one audio file per shot and returns `WordTiming` per word; a missing key is a clear error, not a panic; the pipeline names no vendor. Tests: request building and response parsing against a fixture; no network.

**Unblocks:** M2 audio, subtitles in Wave 3.

### Wave 3 — assembly

**Goal:** animatic from stills + voiceover, deterministic ffmpeg.

| From | To |
| --- | --- |
| `avtool` concat + overlay + volume (Wave 1) | `pipeline/assemble.rs` `assemble()` |
| `composer/src/short.rs` 9:16 + bottom caption layout | `assemble` for `youtube_shorts` |

**Files:** `src/pipeline/assemble.rs`, `src/infra/ffmpeg.rs` (ASS subtitles op).

**Acceptance:** a project with shots and audio assembles into one mp4 at the platform profile's resolution, with burned-in word-level subtitles and loudness normalised to −14 LUFS. Test: `#[ignore]`, needs ffmpeg.

**Unblocks:** M2 complete.

### Wave 4 — job skeleton + image capability

**Goal:** a reusable long-running-operation client, then keyframes on top of it.

| From | To |
| --- | --- |
| `video/src/handler.rs:39` `LRO_*` constants | `src/providers/job.rs` |
| `video/src/handler.rs` `poll_lro` | `src/providers/job.rs` `poll()` |
| `image/src/handler.rs:33` `ImageGenerateParams` | `ImageRequest` in `src/providers/base.rs` |

**Files:** `src/providers/job.rs` (new), `src/providers/base.rs`, one image adapter, `src/pipeline/keyframes.rs`.

**Acceptance:** `poll()` retries with backoff, caps attempts, returns a typed timeout, and surfaces the operation's own error message. Test: the loop runs against a fake status source — no network, no sleeps longer than a millisecond.

**Unblocks:** keyframes (M3), video (Wave 5).

### Wave 5 — video capability

**Goal:** TALK and BROLL shots through `VideoProvider` / `AvatarVideoProvider`.

| From | To |
| --- | --- |
| Wave 4 `poll()` | adapter implementations of both traits |
| `video/src/handler.rs` param surface (duration, aspect, seed, first frame) | `VideoRequest` in `src/providers/base.rs` |

**Files:** `src/providers/base.rs`, one video adapter, `src/pipeline/shots.rs`.

**Acceptance:** `render_shots` writes one clip per shot, records the provider request id for resume, and respects the concurrency ceiling from settings; the pipeline names no vendor.

**Unblocks:** M3 TALK, M4 BROLL.

### Wave 6 — QC adapters

**Goal:** the pure gates in `pipeline/qc.rs` get their inputs.

| From | To |
| --- | --- |
| Wave 1 `probe()` + frame extraction | `qc_keyframe`, `qc_shot`, `qc_final` |
| `providers/base.rs` `JudgeProvider` | judge call for Borderline verdicts |

**Files:** `src/pipeline/qc.rs`, one judge adapter.

**Acceptance:** a keyframe with a low identity score goes to the judge, not to auto-pass; a clip's frames aggregate through `aggregate_clip_frames`.

**Unblocks:** M3/M6 quality gates.

### Wave 7 — export package

**Goal:** the deliverable.

| From | To |
| --- | --- |
| Wave 1 + Wave 3 | `export_package`: mp4 + `metadata.json` with `ai_disclosure` |

**Files:** `src/pipeline/assemble.rs`, `src/domain/project.rs` (metadata shape).

**Acceptance:** the package contains the video, the metadata with disclosure on, and the platform profile used. No upload.

**Unblocks:** M6.

## 4. Order and dependencies

```
Wave 1 (ffmpeg) ──┬─> Wave 3 (assembly) ──> Wave 7 (export)
                  └─> Wave 6 (QC)
Wave 2 (TTS) ─────> Wave 3
Wave 4 (job+image) ─> Wave 5 (video) ─> Wave 6
```

M2 closes after Wave 3. M3 after Wave 5. M6 after Wave 7.

## 5. Rules for every wave

1. Port code, do not add the toolkit as a dependency.
2. No vendor endpoint, auth or model id outside an adapter file.
3. Pipeline code calls traits only; a vendor name in `src/pipeline/` is a defect.
4. No documentation reference in code — no `.md` path, no section number.
5. Every wave adds tests that run without network; anything needing ffmpeg or a key is `#[ignore]`d.
6. A wave is done when `cargo test --features local-identity` and `cargo clippy --all-targets` are clean.

## 6. Risks

- **Provider drift.** Each adapter owns its payload; a change is local to one file.
- **Word timings.** The `TtsProvider` contract requires them; an adapter whose API does not return them falls back to `align_words_local`.
- **ffmpeg versions.** The argv builders assume a recent ffmpeg; `check_available` reports the binary, and the `#[ignore]` tests catch a mismatch.
- **Porting cost.** Waves 1–2 are the bulk of the copied logic (~600 lines); Waves 4–5 are mostly our own code on a small skeleton.
