//! fal.ai REST client (queue API): images, avatar models, video.
//!
//! Primary transport: one key covers most roles. Queue + request_id for
//! resumability.

/// M1: implement ImageProvider, VideoProvider, AvatarVideoProvider, JudgeProvider
/// against https://queue.fal.run with FAL_KEY.
#[allow(dead_code)] // fields are read by the M1/M2 REST impls
pub struct FalClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
