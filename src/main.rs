//! avatar-media-mcp — MCP server entry point.
//!
//! rmcp server: tools/resources/prompts over stdio (ARCHITECTURE.md 9.1–9.3).
//! Tool registration follows section 9.3: avatar, project, series, script,
//! preview, render, assemble, export, ops groups.

fn main() {
    // M1: rmcp ServerHandler with tool_router, stdio transport serve.
    // Startup checks: .env permissions, ffmpeg/ffprobe in PATH, config load.
    unimplemented!("M1")
}
