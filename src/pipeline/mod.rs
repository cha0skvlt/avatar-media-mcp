//! Pipeline stages (ARCHITECTURE.md 3, 7, 8, 10, 11).
//!
//! Order: audio first → keyframes → animatic → shot render → assemble.
//! Every stage is idempotent and cached by hash; retries ≤2 per shot.

pub mod avatar_build;
pub mod audio;
pub mod keyframes;
pub mod shots;
pub mod qc;
pub mod assemble;
