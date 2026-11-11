//! Provider abstraction (ARCHITECTURE.md 6.2, 9.1).
//!
//! Providers are behind traits; models live in config/models.yaml. All REST
//! via reqwest — no vendor SDKs. Google is excluded from the stack
//! (operator decision #9): strict input moderation and SynthID watermarking.

pub mod base;
pub mod registry;

pub mod fal;
pub mod elevenlabs;
pub mod fish;
pub mod byteplus;
