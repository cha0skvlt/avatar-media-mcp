//! ElevenLabs REST client: voice cloning + TTS with word timings.

/// M1: implement TtsProvider against api.elevenlabs.io with ELEVENLABS_API_KEY.
#[allow(dead_code)] // fields are read by the M1/M2 REST impls
pub struct ElevenLabsClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
