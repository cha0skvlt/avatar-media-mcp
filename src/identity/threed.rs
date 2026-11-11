//! Typed FLAME/SMPL geometry interface and pure distance metrics.
//!
//! The fitting operation belongs to a local/API adapter (the fitted vectors
//! are data, not a dependency of the maths core). This module provides strict
//! model/dimension checks and a normalized RMS parameter distance. RMS is used
//! instead of raw Euclidean distance so that values remain comparable when a
//! model's parameter count changes; a fitted model's version and parameter
//! normalization still need to be fixed by the adapter.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryModel {
    Flame,
    Smpl,
    SmplX,
}

/// Non-negative RMS distance in a single geometry model's parameter space.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct GeometryDistance(pub f32);

impl GeometryDistance {
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            bail!("geometry distance must be finite and non-negative, got {value}");
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// Versioned/typed model parameters produced by an external fitting adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryFit {
    pub model: GeometryModel,
    pub parameters: Vec<f32>,
}

impl GeometryFit {
    pub fn new(model: GeometryModel, parameters: Vec<f32>) -> Result<Self> {
        if parameters.is_empty() {
            bail!("geometry fit must contain at least one parameter");
        }
        if parameters.iter().any(|value| !value.is_finite()) {
            bail!("geometry fit contains NaN or infinity");
        }
        Ok(Self { model, parameters })
    }
}

/// RMS distance; comparing FLAME to SMPL or vectors of different shape is an
/// explicit error rather than a numerically plausible but meaningless score.
pub fn geometry_distance(a: &GeometryFit, b: &GeometryFit) -> Result<GeometryDistance> {
    if a.model != b.model {
        bail!("geometry model mismatch: {:?} vs {:?}", a.model, b.model);
    }
    if a.parameters.len() != b.parameters.len() {
        bail!(
            "geometry parameter dimensions differ: {} vs {}",
            a.parameters.len(),
            b.parameters.len()
        );
    }
    if a.parameters.is_empty()
        || a.parameters.iter().chain(&b.parameters).any(|value| !value.is_finite())
    {
        bail!("geometry parameters must be non-empty and finite");
    }

    let sum_squared: f64 = a
        .parameters
        .iter()
        .zip(&b.parameters)
        .map(|(left, right)| (f64::from(*left) - f64::from(*right)).powi(2))
        .sum();
    GeometryDistance::new((sum_squared / a.parameters.len() as f64).sqrt() as f32)
}

/// Map a calibrated non-negative geometry distance to a bounded similarity
/// proxy. The scale is an explicit M0 calibration parameter, never inferred
/// from a single frame.
pub fn distance_to_similarity(distance: GeometryDistance, scale: f32) -> Result<f32> {
    if !scale.is_finite() || scale <= 0.0 {
        bail!("geometry distance scale must be finite and positive");
    }
    Ok((-(distance.0 / scale)).exp())
}

/// Fit FLAME parameters from 2–3 best photos. Actual fitting is adapter work;
/// no FLAME runtime, licensed weights, or provider contract was included in
/// the supplied files, so this boundary returns an actionable error.
pub async fn fit_face(_photos: &[PathBuf]) -> Result<serde_json::Value> {
    bail!("FLAME fitting adapter/model is not included in the supplied source set")
}

/// Fit SMPL/BLADE parameters from full-body references.
pub async fn fit_body(_photos: &[PathBuf]) -> Result<serde_json::Value> {
    bail!("SMPL/BLADE fitting adapter/model is not included in the supplied source set")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_distance_is_normalized_and_model_checked() {
        let a = GeometryFit::new(GeometryModel::Flame, vec![0.0, 0.0]).unwrap();
        let b = GeometryFit::new(GeometryModel::Flame, vec![3.0, 4.0]).unwrap();
        assert!((geometry_distance(&a, &b).unwrap().get() - 3.535_534).abs() < 1e-5);
        let other = GeometryFit::new(GeometryModel::Smpl, vec![0.0, 0.0]).unwrap();
        assert!(geometry_distance(&a, &other).is_err());
    }
}
