# avatar-media-mcp

**A Rust MCP server that gives [CHA0S AOE](https://github.com/cha0skvlt/CHA0S_AOE) hands for generative media with avatars** — identity, face swap, voice, frames, clips, the final cut, and a budget it refuses to exceed.

![Rust](https://img.shields.io/badge/rust-edition_2024-000000?logo=rust&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-rmcp_0.3-6E4AFF)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-ort_2.0--rc.15-005CED?logo=onnx&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-4169E1?logo=postgresql&logoColor=white)
![Tests](https://img.shields.io/badge/tests-107_passing-2ea043)
![License](https://img.shields.io/badge/license-PolyForm_Shield_1.0.0-blue)
![Status](https://img.shields.io/badge/status-under_construction-orange)

---

> # 🚧 UNDER CONSTRUCTION 🚧
>
> **This is an active build, not a release.** The identity core, the face-swap
> pipeline and `avatar_create` work and are tested. The generation pipeline —
> TTS, keyframes, shots, assembly — is stubbed at M2–M6, and 14 of the 17
> registered MCP tools answer with an explicit `planned for M<n>` error.
>
> The contour is real and the stubs are honest: every unimplemented path returns
> the milestone it belongs to instead of pretending. `ARCHITECTURE.md` carries
> the per-module status map (✅ / ◐ / ⬜).

---

## What works today

| Piece | State |
| --- | --- |
| InsightFace pipeline in pure Rust — SCRFD detect, 112×112 alignment (`solve6`), ArcFace 512-d | ✅ |
| Identity decisions — Bayesian threshold calibration, Theil–Sen drift, Hungarian multi-character assignment | ✅ |
| `faceswap` — InSwapper + Reinhard color transfer + landmark paste mask, `ort` on CPU | ✅ |
| `avatar_create` — photos → clusters → identity model → `avatar.json`, agent round-trip | ✅ |
| `providers_check` — preflight over config, keys, ORT dylib, models, ffmpeg, Postgres | ✅ |
| TTS, keyframes, TALK/BROLL shots, assembly, export | ⬜ M2–M6 |

## Quick start

```bash
cargo build --release --features local-identity
```

`local-identity` enables the ONNX path (`ort`, load-dynamic). The server resolves
`models/`, `avatars/` and `config/` **relative to the process cwd** — launch it
from this directory:

```bash
ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
  ./target/release/avatar-media-mcp
```

Logs go to stderr; `RUST_LOG=debug` raises the level.

### Wire it into CHA0S AOE

`~/.config/cha0s/settings.json` — zero changes to the IDE itself:

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

Keys travel through the process env (`context_servers.env`); `.env.example`
lists the names. They are never logged and never written to the database.

## Design in three lines

- **Provider-neutral.** Models are data in `config/models.yaml` (role → tier → provider/price); every call goes through the traits in `src/providers/base.rs`. A vendor name lives only in an adapter file.
- **Typed where it matters.** `CosineScore`, `EmbeddingSource`, `GeometryDistance` — mixing incompatible vector spaces is a compile error, not a silent 0.9.
- **Money is transactional.** `ledger.rs` locks the project row and enforces the cap inside one transaction; `confine_output` keeps tool writes inside the workspace; every image decode is bounded.

## Requirements

- `ffmpeg` + `ffprobe` in PATH (assembly, M2).
- `ORT_DYLIB_PATH` — required by `faceswap`, `avatar_create` and the CLIs.
- `models/` (~370 MB) — buffalo_l from HuggingFace `ApacheOne/insightface` plus the faceswap weights. Not in the repo.
- Postgres with pgvector — only for stateful tools (M2).

## Tests

```bash
cargo test --features local-identity    # 107 passing, hermetic (+1 ignored)
```

## Docs

`ARCHITECTURE.md` — the code map: modules, status per file, accepted decisions.

## License

[PolyForm Shield 1.0.0](LICENSE) — source-available. Read it, run it, modify it,
redistribute it; do not use it to ship a product that competes with this software
or with a product the licensor provides using it.

Required Notice: Copyright 2026 cha0skvlt (https://github.com/cha0skvlt)
