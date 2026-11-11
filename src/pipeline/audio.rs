//! Audio-first stage.
//!
//! TTS per shot (cloned voice) → exact durations → word timings. Video is
//! fitted to sound, never the reverse.

/// Synthesize voice-over for every shot; returns audio files + word timings.
pub async fn synthesize_script(
    _script: &crate::domain::script::Script,
    _tier: crate::config::models::Tier,
) -> anyhow::Result<Vec<(String, std::path::PathBuf, Vec<crate::providers::base::WordTiming>)>> {
    // M2.
    anyhow::bail!("synthesize_script lands with M2 (TTS provider + word timings)")
}

/// Local whisper fallback for word alignment when TTS provides none.
pub async fn align_words_local(_audio: std::path::PathBuf, _text: &str) -> anyhow::Result<Vec<crate::providers::base::WordTiming>> {
    // M2 (optional feature).
    anyhow::bail!("align_words_local lands with M2 (local whisper fallback)")
}
