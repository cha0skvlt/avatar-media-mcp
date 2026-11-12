//! MCP stdio server (M1): `faceswap` and `avatar_create` are live; the rest
//! of the M1–M7 tool surface answers with honest "not implemented yet"
//! errors so the contour is visible to the agent.
//!
//! Run: `cargo run --features local-identity --bin avatar-media-mcp`
//! with `cwd = avatar-media-mcp` and `ORT_DYLIB_PATH=models/ort/lib/…`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};

use avatar_media_mcp::identity::arcface::FaceEngine;
use avatar_media_mcp::infra::paths::confine_output;
use avatar_media_mcp::pipeline::faceswap::SwapEngine;
use avatar_media_mcp::tools::avatar::{AgentStep, AvatarCreateArgs, avatar_create};
use avatar_media_mcp::tools::health::preflight;

/// A hung ONNX session must not hang the agent forever.
const SWAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
const AVATAR_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(900);

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct FaceswapArgs {
    /// Path of the SOURCE image whose face identity is transferred.
    source_path: String,
    /// Path of the TARGET image whose face is replaced.
    target_path: String,
    /// Output PNG path.
    out_path: String,
}

#[derive(Clone)]
struct AvatarMediaServer {
    engine: Arc<Mutex<Option<FaceEngine>>>,
    swap: Arc<Mutex<Option<SwapEngine>>>,
    tool_router: Arc<ToolRouter<Self>>,
}

/// Lazily build the shared ORT environment + both model bundles on the
/// first call; a failed load leaves None so the next call retries.
fn with_engines<T>(
    engine: &Mutex<Option<FaceEngine>>,
    swap: &Mutex<Option<SwapEngine>>,
    f: impl FnOnce(&mut FaceEngine, &mut SwapEngine) -> anyhow::Result<T>,
) -> Result<T, McpError> {
    let mut engine_guard = engine
        .lock()
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    let mut swap_guard = swap
        .lock()
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    if engine_guard.is_none() {
        let e = FaceEngine::load(Path::new("models/insightface/buffalo_l"))
            .map_err(|err| McpError::internal_error(format!("face engine load: {err}"), None))?;
        *engine_guard = Some(e);
    }
    if swap_guard.is_none() {
        let s = SwapEngine::load(Path::new("models/faceswap"))
            .map_err(|err| McpError::internal_error(format!("swap engine load: {err}"), None))?;
        *swap_guard = Some(s);
    }
    f(engine_guard.as_mut().unwrap(), swap_guard.as_mut().unwrap())
        .map_err(|err| McpError::internal_error(err.to_string(), None))
}

/// A tool that ran and failed. MCP wants execution failures inside the tool
/// result (`isError`), where the model sees the text; a JSON-RPC error is for
/// protocol problems (bad arguments, unknown tool) and clients may swallow it.
fn tool_failure(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}

fn not_yet(tool: &str, milestone: &str) -> McpError {
    McpError::invalid_params(
        format!("tool `{tool}` is planned for milestone {milestone}; not implemented yet"),
        None,
    )
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

    /// FS1 face swap (live).
    #[tool(description = "Face swap: transfer the identity from the source image's main face onto the face in the target image (pose, hair, body and background of the target are kept). Saves a PNG and reports the detection + timing. Exactly one face (best-scoring detection) is swapped.")]
    async fn faceswap(&self, rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<FaceswapArgs>) -> Result<CallToolResult, McpError> {
        let src = std::path::PathBuf::from(&args.source_path);
        let tgt = std::path::PathBuf::from(&args.target_path);
        // Writes are confined to the workspace: an injected call must not be
        // able to overwrite arbitrary files.
        let workspace = std::env::current_dir()
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let out = confine_output(std::path::Path::new(&args.out_path), &workspace)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let engine = self.engine.clone();
        let swap = self.swap.clone();
        let started = std::time::Instant::now();
        let report = tokio::task::spawn_blocking(move || {
            with_engines(&engine, &swap, |engine, swap| {
                let src_faces = engine.embed_image(&src)?;
                anyhow::ensure!(!src_faces.is_empty(), "no face in source {}", src.display());
                let source_embedding = src_faces[0].vector.clone();

                let detections = engine.detect_faces(&tgt)?;
                anyhow::ensure!(!detections.is_empty(), "no face in target {}", tgt.display());
                let det = &detections[0];

                let target = avatar_media_mcp::infra::image_io::open_rgb8(&tgt)?;
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
        });
        let report = match tokio::time::timeout(SWAP_TIMEOUT, report).await {
            Err(_) => {
                return Ok(tool_failure(format!(
                    "faceswap exceeded {}s",
                    SWAP_TIMEOUT.as_secs()
                )));
            }
            Ok(Err(join_error)) => return Ok(tool_failure(format!("swap task: {join_error}"))),
            Ok(Ok(Err(error))) => return Ok(tool_failure(error.to_string())),
            Ok(Ok(Ok(report))) => report,
        };
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "faceswap done");
        Ok(CallToolResult::success(vec![Content::text(report)]))
    }

    /// avatar_create: the M1 end-to-end path (live).
    #[tool(description = "Create an avatar from reference photos. Multi-call protocol; every step answers with JSON {status, message, next}: (1) no identity_json → status needs_identity: cluster representatives + numeric summary; (2) identity_json → status needs_verification: the verification prompt; (3) identity_json + verification_json {match_score, mismatches} → if match_score >= 0.8 writes avatars/<slug>/{avatar.json, identity_model.json, refs/NN.jpg} atomically; below 0.8 → status verification_failed and NOTHING is written: fix identity_json and verify again (only the operator may pass accept_flagged=true to write anyway). Photos are normalized and quality-gated; an approved canon is never replaced without force=true.")]
    async fn avatar_create(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            AvatarCreateArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let slug = args.slug.clone();
        let started = std::time::Instant::now();
        let joined = tokio::time::timeout(
            AVATAR_TIMEOUT,
            tokio::task::spawn_blocking(move || avatar_create(&args)),
        )
        .await;
        let outcome = match joined {
            Err(_) => {
                return Ok(tool_failure(format!(
                    "avatar_create exceeded {}s",
                    AVATAR_TIMEOUT.as_secs()
                )));
            }
            Ok(Err(join_error)) => {
                return Ok(tool_failure(format!("avatar_create task: {join_error}")));
            }
            Ok(Ok(outcome)) => outcome,
        };
        let report = match outcome {
            Ok(report) => report,
            Err(error) => {
                // The round-trip steps and the verification gate are part of the
                // protocol: hand the agent a normal result with the next call.
                if let Some(step) = error.downcast_ref::<AgentStep>() {
                    let text = serde_json::to_string_pretty(step)
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                    return Ok(CallToolResult::success(vec![Content::text(text)]));
                }
                return Ok(tool_failure(format!("{error:#}")));
            }
        };
        tracing::info!(
            slug = %slug,
            clusters = report.clusters,
            flagged = report.verification_flagged,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "avatar_create done"
        );
        let text = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    // ---- Planned surface: honest stubs, tagged with their milestone ----

    #[tool(description = "List avatars with status. Planned for M1 (post-skeleton).")]
    async fn avatar_list(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("avatar_list", "M1"))
    }

    #[tool(description = "Avatar status: canon sheet, voice, prompt blocks. Planned for M1 (post-skeleton).")]
    async fn avatar_status(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("avatar_status", "M1"))
    }

    #[tool(description = "Operator approval gate for the canon sheet. Planned for M1 (post-skeleton).")]
    async fn avatar_approve_canon(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("avatar_approve_canon", "M1"))
    }

    #[tool(description = "Create a project from a brief. Planned for M1 (post-skeleton).")]
    async fn project_create(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("project_create", "M1"))
    }

    #[tool(description = "Submit a script for estimate/approval. Planned for M1 (post-skeleton).")]
    async fn script_submit(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("script_submit", "M1"))
    }

    #[tool(description = "Cost estimate per tier. Planned for M2 (needs ledger).")]
    async fn script_estimate(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("script_estimate", "M2"))
    }

    #[tool(description = "Mandatory approval gate before any spend. Planned for M2.")]
    async fn script_approve(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("script_approve", "M2"))
    }

    #[tool(description = "Start rendering a project (returns job_id). Planned for M3.")]
    async fn render_start(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("render_start", "M3"))
    }

    #[tool(description = "Poll render job status. Planned for M3.")]
    async fn job_status(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("job_status", "M3"))
    }

    #[tool(description = "Cancel a render job. Planned for M5.")]
    async fn job_cancel(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("job_cancel", "M5"))
    }

    #[tool(description = "Regenerate one shot. Planned for M5.")]
    async fn shot_regenerate(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("shot_regenerate", "M5"))
    }

    #[tool(description = "Assemble the final cut with ffmpeg. Planned for M2.")]
    async fn assemble(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("assemble", "M2"))
    }

    #[tool(description = "Export mp4 + metadata.json package. Planned for M6.")]
    async fn export_package(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("export_package", "M6"))
    }

    #[tool(description = "Spend ledger query. Planned for M2.")]
    async fn cost_ledger(&self) -> Result<CallToolResult, McpError> {
        Err(not_yet("cost_ledger", "M2"))
    }

    #[tool(description = "Preflight: config files, provider keys (presence only), ORT dylib, model directories, ffmpeg, avatars directory, DATABASE_URL. No network calls.")]
    async fn providers_check(&self) -> Result<CallToolResult, McpError> {
        let report = preflight().await;
        let text = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for AvatarMediaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    tracing::info!("avatar-media-mcp starting");
    let server = AvatarMediaServer::new();
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
