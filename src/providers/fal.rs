//! fal.ai REST client (queue API): images, avatar models, Kling/Seedance/Wan.
//!
//! Primary transport: one key covers most roles. Queue + request_id for
//! resumability (ARCHITECTURE.md 6.2, 9.2).

/// M1: implement ImageProvider, VideoProvider, AvatarVideoProvider, JudgeProvider
/// against https://queue.fal.run with FAL_KEY.
pub struct FalClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
