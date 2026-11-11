//! MCP stdio server (M1 skeleton, first real tool: `faceswap`).
//!
//! Run: `cargo run --features local-identity --bin avatar-media-mcp`
//! with `cwd = avatar-media-mcp` and `ORT_DYLIB_PATH=models/ort/lib/…`
//! (ort `load-dynamic` reads the env var directly).
//!
//! The tool wraps the same `SwapEngine::swap_face` contract as the CLI
//! binary `src/bin/faceswap.rs` — one implementation, two entry points.

use std::sync::{Arc, Mutex};

use anyhow::Context;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    model::{
        CallToolResult, Content, ServerCapabilities, ServerInfo,
    },
    tool, tool_handler, tool_router,
};

#[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
struct FaceswapArgs {
    /// Path of the SOURCE image whose face identity is transferred.
    source_path: String,
    /// Path of the TARGET image whose face is replaced.
    target_path: String,
    /// Output PNG path.
    out_path: String,
}

use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::pipeline::faceswap::SwapEngine;

#[derive(Clone)]
struct AvatarMediaServer {
    engine: Arc<Mutex<Option<FaceEngine>>>,
    swap: Arc<Mutex<Option<SwapEngine>>>,
    tool_router: Arc<ToolRouter<AvatarMediaServer>>,
}

/// Lazily build the shared ORT environment + both model bundles on the
/// first call; a failed load poisons the option back to None so the next
/// call retries (e.g. after the operator fixes ORT_DYLIB_PATH).
fn with_engines<T>(
    engine: &Mutex<Option<FaceEngine>>,
    swap: &Mutex<Option<SwapEngine>>,
    f: impl FnOnce(&mut FaceEngine, &mut SwapEngine) -> anyhow::Result<T>,
) -> Result<T, McpError> {
    let mut engine_guard = engine.lock().map_err(|e| McpError::internal_error(e.to_string(), None))?;
    let mut swap_guard = swap.lock().map_err(|e| McpError::internal_error(e.to_string(), None))?;
    if engine_guard.is_none() {
        let e = FaceEngine::load(std::path::Path::new("models/insightface/buffalo_l"))
            .map_err(|err| McpError::internal_error(format!("face engine load: {err}"), None))?;
        *engine_guard = Some(e);
    }
    if swap_guard.is_none() {
        let s = SwapEngine::load(std::path::Path::new("models/faceswap"))
            .map_err(|err| McpError::internal_error(format!("swap engine load: {err}"), None))?;
        *swap_guard = Some(s);
    }
    f(engine_guard.as_mut().unwrap(), swap_guard.as_mut().unwrap())
        .map_err(|err| McpError::internal_error(err.to_string(), None))
}

fn err(e: McpError) -> McpError {
    e
}

#[tool_router]
impl AvatarMediaServer {
    fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            swap: Arc::new(Mutex::new(None)),
            tool_router: Arc::new(Self::tool_router()),
        }
    }

    /// Swap the identity from `source_path` (its main face) onto the face
    /// in `target_path`, save PNG to `out_path`.
    #[tool(description = "Face swap: transfer the identity from the source image's main face onto the face in the target image (pose, hair, body and background of the target are kept). Saves a PNG and reports the detection + timing. Exactly one face (best-scoring detection) is swapped.")]
    async fn faceswap(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            FaceswapArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let src = std::path::PathBuf::from(&args.source_path);
        let tgt = std::path::PathBuf::from(&args.target_path);
        let out = std::path::PathBuf::from(&args.out_path);
        // ort sessions are not Send-friendly across await points; run the
        // whole pipeline inside one blocking task from the engines' view.
        let engine = self.engine.clone();
        let swap = self.swap.clone();
        let report = tokio::task::spawn_blocking(move || {
            with_engines(&engine, &swap, |engine, swap| {
                let src_faces = engine.embed_image(&src)?;
                anyhow::ensure!(!src_faces.is_empty(), "no face in source {}", src.display());
                let source_embedding = src_faces[0].vector.clone();

                let detections = engine.detect_faces(&tgt)?;
                anyhow::ensure!(!detections.is_empty(), "no face in target {}", tgt.display());
                let det = &detections[0];

                let target = image::open(&tgt)
                    .with_context(|| format!("open {}", tgt.display()))?
                    .to_rgb8();
                let result = swap.swap_face(&target, det.bbox, det.kps, &source_embedding)?;
                result.save(&out)?;
                Ok(format!(
                    "source: dim {}, norm {:.4}\ntarget: {} face(s), best score {:.3}, bbox [{:.0},{:.0},{:.0},{:.0}]\ngpen available: {}\nsaved: {}",
                    source_embedding.len(),
                    source_embedding.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    detections.len(),
                    det.score,
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    swap.gpen_available(),
                    out.display(),
                ))
            })
        })
        .await
        .map_err(|e| McpError::internal_error(format!("swap task: {e}"), None))?
        .map_err(err)?;
        Ok(CallToolResult::success(vec![Content::text(report)]))
    }
}

#[tool_handler]
impl ServerHandler for AvatarMediaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let server = AvatarMediaServer::new();
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
