# avatar-media-mcp

MCP server for generative media with avatars. Rust + rmcp (official MCP SDK).
Architecture: [ARCHITECTURE.md](/ARCHITECTURE.md). Code follows the work plan, section 13 (M0–M7).

## Build

```sh
cargo build --release          # single static binary target/release/avatar-media-mcp
cargo test                     # providers are mocked; ffmpeg tests on synthetic clips
```

Requirements: Rust 1.97+ (edition 2024), `ffmpeg`/`ffprobe` in PATH.

Identity inference (optional feature, local, no network):

```sh
cargo build --features local-identity --bin identity_test
cargo build --features local-identity --bin avatar_build
# models: models/insightface/buffalo_l/ + models/ort/lib/ (not in git, see ARCHITECTURE §15)
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  ./target/debug/avatar_build --name <slug> --photos <dir> --consent self
```

Face swap (FS1), two entry points over the same `SwapEngine`:

```sh
cargo build --features local-identity --bins
# CLI:
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  ./target/debug/faceswap <source_face_img> <target_img> [out.png]
# MCP server (tool `faceswap`: source_path, target_path, out_path):
cd avatar-media-mcp && ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  ./target/debug/avatar-media-mcp
```

## Wiring into CHA0S AOE

```jsonc
// ~/.config/cha0s/settings.json
"context_servers": {
  "avatar-media": {
    "command": "/abs/path/to/avatar-media-mcp",
    "args": [],
    "env": { "FAL_KEY": "...", "ELEVENLABS_API_KEY": "...", "DATABASE_URL": "..." }
  }
}
```

## Structure

```
src/
├── main.rs        # rmcp stdio server (M1): tool `faceswap` live; the rest M1–M7 stubs
├── bin/           # faceswap (FS1 CLI), identity_test, avatar_build, faceswap_diag
├── config/        # models.yaml, platforms.yaml, .env
├── domain/        # Avatar, Project, Script, Shot, Job, Cost (serde + schemars)
├── identity/      # ArcFace embeddings, FLAME/SMPL fits, scoring (ARCHITECTURE §4)
├── providers/     # traits + fal/byteplus/elevenlabs/fish REST clients, role+tier registry
├── pipeline/      # faceswap (FS1, ✅), avatar_build, audio, keyframes, shots, qc, assemble
├── infra/         # Postgres store (sqlx), file cache, cost ledger, ffmpeg ops
└── series.rs      # serialization for the film format
config/            # models.yaml (model registry), platforms.yaml (platform profiles)
tests/             # integration tests
```

## Keys

See `.env.example`. MVP minimum: `FAL_KEY`, `ELEVENLABS_API_KEY`, `DATABASE_URL`.
`.env` is gitignored; the server never writes keys to logs or the database.

## What is not in git

`avatars/` (personal data), `models/` (weights, ~370 MB), test asset sets (`Ksenia/`, `Eugene/`).
