//! Fish Audio REST client: alternative voice clone provider.

/// Optional; implement TtsProvider if M0 picks Fish over ElevenLabs.
pub struct FishClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
