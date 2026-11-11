//! Script schema written by the IDE agent.
//!
//! Validation rules enforced by the server: TALK 3–12 s, shorts rhythm 2–5 s,
//! hook in the first shot, ≤150–170 wpm, speaking ∈ cast, voice configured.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Script {
    pub title: String,
    pub language: String,
    pub platform: String,
    pub hook: String,
    /// Avatar slugs, ≥1.
    pub cast: Vec<String>,
    pub shots: Vec<Shot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShotType {
    Talk,
    Broll,
    Screen,
    Title,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Framing {
    Cu,
    Mcu,
    Ms,
    Ws,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Shot {
    pub id: String,
    #[serde(rename = "type")]
    pub shot_type: ShotType,
    /// Voice-over text of this shot.
    #[serde(default)]
    pub vo: Option<String>,
    pub visual: String,
    /// Characters in frame; empty for pure B-roll.
    #[serde(default)]
    pub characters: Vec<String>,
    /// Whose voice sounds in this shot; must reference cast.
    #[serde(default)]
    pub speaking: Option<String>,
    #[serde(default)]
    pub framing: Option<Framing>,
    #[serde(default)]
    pub wardrobe: Option<String>,
    /// Motion prompt for avatar models.
    #[serde(default)]
    pub motion: Option<String>,
    pub est_seconds: f64,
}
