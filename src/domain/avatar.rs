//! Avatar identity profile.
//!
//! `avatars/<slug>/` holds `avatar.json` (identity, persona, voice,
//! prompt_blocks, refs), `consent.json` (hard gate), `identity_model.json`
//! (ArcFace embeddings + per-avatar threshold) and `refs/NN.jpg` (the canon).
//! Every path inside `avatar.json` is relative to the avatar directory;
//! `AvatarStore` enforces that on write and resolves it on read.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

impl Identity {
    /// Number of filled leaf fields (hair counts as three).
    pub fn filled_fields(&self) -> usize {
        [
            self.apparent_age_range.as_str(),
            self.gender_presentation.as_str(),
            self.face_shape.as_str(),
            self.skin_tone.as_str(),
            self.eyes.as_str(),
            self.brows.as_str(),
            self.nose.as_str(),
            self.lips.as_str(),
            self.hair.color.as_str(),
            self.hair.length.as_str(),
            self.hair.style.as_str(),
            self.facial_hair.as_deref().unwrap_or(""),
            self.build.as_str(),
        ]
        .iter()
        .filter(|value| !value.trim().is_empty())
        .count()
            + usize::from(!self.distinguishing_marks.is_empty())
    }

    /// An identity with no filled field is not a description.
    pub fn is_empty(&self) -> bool {
        self.filled_fields() == 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Avatar {
    pub schema_version: u32,
    pub slug: String,
    pub display_name: String,
    pub consent_ref: String,
    pub identity: Identity,
    #[serde(default)]
    pub persona: Option<Persona>,
    #[serde(default)]
    pub voice: Option<VoiceRef>,
    pub prompt_blocks: PromptBlocks,
    pub refs: Refs,
    pub created_at: String,
    #[serde(default)]
    pub canon_approved_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Identity {
    pub apparent_age_range: String,
    pub gender_presentation: String,
    pub face_shape: String,
    pub skin_tone: String,
    pub eyes: String,
    pub brows: String,
    pub nose: String,
    pub lips: String,
    pub hair: Hair,
    #[serde(default)]
    pub facial_hair: Option<String>,
    #[serde(default)]
    pub distinguishing_marks: Vec<String>,
    pub build: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Hair {
    pub color: String,
    pub length: String,
    pub style: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Persona {
    pub energy: String,
    pub gestures: String,
    pub default_expression: String,
    pub speaking_pace_wpm: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct VoiceRef {
    pub provider: String,
    pub voice_id: String,
    pub language: String,
    #[serde(default)]
    pub stability: Option<f64>,
    #[serde(default)]
    pub style: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PromptBlocks {
    pub identity: String,
    pub negative: String,
    pub do_not_change: Vec<String>,
    pub ref_binding: RefBinding,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RefBinding {
    #[serde(default)]
    pub gemini_image: Option<String>,
    #[serde(default)]
    pub omni_video: Option<String>,
    pub generic: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Refs {
    pub primary: String,
    pub set: Vec<String>,
}

/// Consent gate. avatar_create refuses without it.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Consent {
    pub subject: ConsentSubject,
    pub statement: String,
    pub date: String,
    pub scope: Vec<String>,
    /// Required for third_party: path to the written consent file.
    #[serde(default)]
    pub written_consent_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConsentSubject {
    Self_,
    ThirdParty,
    Fictional,
}

/// Mathematical identity lock, stored in identity_model.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityModel {
    /// ArcFace 512-d embeddings per reference image.
    pub face_embeddings: std::collections::BTreeMap<String, Vec<f32>>,
    /// FLAME fit parameters (face geometry passport).
    #[serde(default)]
    pub face_model: Option<serde_json::Value>,
    /// SMPL/BLADE body fit parameters.
    #[serde(default)]
    pub body_model: Option<serde_json::Value>,
    /// Calibrated identity threshold (M0).
    #[serde(default)]
    pub identity_threshold: Option<f32>,
}
