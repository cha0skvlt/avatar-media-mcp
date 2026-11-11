//! Fish Audio REST client: alternative voice clone provider.

/// Optional; implement TtsProvider if M0 picks Fish over ElevenLabs.
#[allow(dead_code)] // fields are read by the M1/M2 REST impls
pub struct FishClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
