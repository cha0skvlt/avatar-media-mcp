//! Provider abstraction.
//!
//! Providers are behind traits; models live in `config/models.yaml`. All REST
//! via reqwest — no vendor SDKs.

pub mod base;
pub mod registry;

pub mod fal;
pub mod elevenlabs;
pub mod fish;
pub mod byteplus;
