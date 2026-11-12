# avatar-media-mcp

Rust MCP server for generative media with avatars. The agent is the brain
(script, dialogue, approvals); this server executes everything that costs money
or touches files.

> **Status: active build (work in progress).** The identity core, the face-swap
> pipeline and `avatar_create` are implemented and tested; the generation
> pipeline (TTS, keyframes, shots, assembly) is stubbed at M2–M6. See
> `ARCHITECTURE.md` for the per-module status map.

**Live today:** `faceswap` (FS1) and `avatar_create` (M1), plus `providers_check`
(preflight). The remaining 14 tools answer with an explicit "planned for M<n>"
error.

## Build

```bash
cargo build --release --features local-identity
```

`local-identity` enables the ONNX path (`ort`, load-dynamic) that `faceswap`
and `avatar_create` load their models through.

## Run

The server resolves `models/`, `avatars/` and `config/` **relative to the
process cwd**: launch it from this directory.

```bash
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  ./target/release/avatar-media-mcp
```

Logs go to stderr; `RUST_LOG=debug` raises the level.

## Wiring into CHA0S AOE

`~/.config/cha0s/settings.json`:

```jsonc
"context_servers": {
  "avatar-media": {
    "command": "/abs/path/to/avatar-media-mcp",
    "args": [],
    "env": {
      "FAL_KEY": "...",
      "ELEVENLABS_API_KEY": "...",
      "DATABASE_URL": "postgres://..."
    }
  }
}
```

Keys are read from the **process env** (`context_servers.env`); `.env.example`
lists the names.

## Layout

```
config/            models.yaml (registry), platforms.yaml (export rules), settings.yaml (runtime knobs)
migrations/        0001_init.sql (schema), 0002_integrity.sql (unique request_id, FKs, cascade)
models/            insightface/buffalo_l, faceswap, ort/lib
avatars/           <slug>/{avatar.json, identity_model.json, refs/NN.jpg}
src/config/        models, platforms, settings (figment loader)
src/domain/        avatar, project, script, job, cost, series — types only
src/identity/      scoring (decisions), arcface (SCRFD + ArcFace), clusters, textpass, embeddings, threed
src/providers/     base traits; fal/elevenlabs/fish/byteplus structs (M1/M2)
src/pipeline/      faceswap, avatar_build (normalize + quality), qc (pure gates); audio/keyframes/shots/assemble at M2–M6
src/infra/         avatar_store, image_io (bounded decode), paths (write confinement), cache, ffmpeg, ledger, store
src/tools/         avatar_create, health (preflight)
src/bin/           avatar_build, faceswap, faceswap_diag, identity_test
tests/             integration, smoke, db (needs DATABASE_URL), e2e_models (ignored, needs weights)
```

## CLI entry points

```bash
# Avatar from a folder of photos (writes through the same AvatarStore as the tool)
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  cargo run --features local-identity --bin avatar_build -- \
  --name <slug> --photos <dir>

# Face swap
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  cargo run --features local-identity --bin faceswap -- <face> <target> <out.png>

# Embedding sanity check
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  cargo run --features local-identity --bin identity_test -- <img1> [<img2>]
```

## Tests

```bash
cargo test --features local-identity            # 107 tests, hermetic (+1 ignored)
DATABASE_URL=postgres://localhost/avatar_media_qa \
  cargo test --features local-identity --test db
AVATAR_E2E_PHOTOS=/path/to/photos ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  cargo test --features local-identity --test e2e_models -- --ignored --nocapture
```

The `db` test skips itself when `DATABASE_URL` is unset. The `e2e_models` test is
`#[ignore]`d: it runs the full three-call `avatar_create` flow on real photos.

## Requirements

- `ffmpeg` + `ffprobe` in PATH — used by assembly (M2).
- `ORT_DYLIB_PATH` — required by `faceswap`, `avatar_create` and the CLIs.
- `models/` (~370 MB) — download from HuggingFace `ApacheOne/insightface`
  (buffalo_l) plus the faceswap weights.
- Postgres with pgvector — only for stateful tools (M2). `CREATE EXTENSION vector`
  in `0001_init.sql` needs a superuser; on a managed instance create the
  extension first.

## License

PolyForm Shield License 1.0.0 — see `LICENSE`. Source-available: you may read,
run, modify and redistribute the code, but not use it to provide a product that
competes with this software or with a product the licensor provides using it.

Required Notice: Copyright 2026 cha0skvlt (https://github.com/cha0skvlt)
