//! Identity decision mathematics: calibration, multi-person assignment, drift,
//! clip aggregation, and embedding/geometry fusion.
//!
//! The core is deterministic and has no database/provider/filesystem access.
//! All encoder provenance, angle offsets, thresholds, and fitting scales are
//! explicit inputs. QC-1 is intentionally tri-state: only statistically safe
//! scores auto-pass/auto-fail; the uncertain interval is sent to the paid judge.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::embeddings::{CosineScore, EmbeddingSource, FaceEmbedding, compare_embeddings};
use super::threed::{GeometryDistance, distance_to_similarity};

/// A threshold in the same numeric space as the score it compares against.
/// For identity thresholds this is cosine similarity; for fusion thresholds
/// it is a calibrated probability in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ScoreThreshold(pub f32);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage1Verdict {
    Pass(f32),
    Borderline(f32),
    Fail(f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDiagnostics {
    pub source: EmbeddingSource,
    pub positive_samples: usize,
    pub negative_samples: usize,
    pub target_fpr: f64,
    pub target_fnr: f64,
    pub confidence_level: f64,
    /// One-sided Bayesian credible upper bound for false acceptance at `pass`.
    pub false_accept_upper_bound: f64,
    /// One-sided Bayesian credible upper bound for false rejection at `fail`.
    pub false_reject_upper_bound: f64,
    pub auto_pass_enabled: bool,
    pub auto_fail_enabled: bool,
}

/// Per-avatar thresholds. `source = Some(..)` binds them to one encoder space.
/// `None` is reserved for the conservative legacy/default thresholds before
/// an avatar has a calibrated source-specific data set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityThresholds {
    /// Similarity at/above this value may bypass the judge.
    pub pass: ScoreThreshold,
    /// Similarity below this value may be rejected locally.
    pub fail: ScoreThreshold,
    pub source: Option<EmbeddingSource>,
    /// False when the small calibration set cannot statistically support the
    /// requested FPR/FNR. In that case the system abstains to QC-2/judge.
    pub auto_pass_enabled: bool,
    pub auto_fail_enabled: bool,
    #[serde(default)]
    pub calibration: Option<CalibrationDiagnostics>,
}

impl Default for IdentityThresholds {
    fn default() -> Self {
        // Explicit M0 bootstrap values from the original design. They remain
        // conservative placeholders until the first per-avatar calibration.
        Self {
            pass: ScoreThreshold(0.72),
            fail: ScoreThreshold(0.58),
            source: None,
            auto_pass_enabled: true,
            auto_fail_enabled: true,
            calibration: None,
        }
    }
}

impl IdentityThresholds {
    /// Bind the default thresholds to an encoder without changing their M0
    /// bootstrap values. VisionFal must never inherit ArcFace's calibrated set.
    pub fn for_source(source: EmbeddingSource) -> Self {
        Self {
            source: Some(source),
            ..Self::default()
        }
    }

    /// Apply the existing `IdentityModel.identity_threshold` field as the
    /// avatar-specific PASS threshold. The FAIL threshold remains the default
    /// until enough labelled positives support its own calibration.
    pub fn with_avatar_pass_threshold(
        source: EmbeddingSource,
        stored_pass_threshold: Option<f32>,
    ) -> Result<Self> {
        let mut thresholds = Self::for_source(source);
        if let Some(value) = stored_pass_threshold {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                bail!("avatar identity_threshold must be in [-1, 1], got {value}");
            }
            thresholds.pass = ScoreThreshold(value);
            // A per-avatar pass threshold below the default fail boundary would
            // otherwise invert the gate: every score >= pass would auto-pass,
            // including scores the fail boundary was meant to reject. Collapse
            // the grey zone instead of silently disabling the reject gate.
            if thresholds.fail.0 > thresholds.pass.0 {
                thresholds.fail = ScoreThreshold(thresholds.pass.0);
            }
        }
        Ok(thresholds)
    }

    /// The scalar to persist in `IdentityModel.identity_threshold`.
    pub fn avatar_pass_threshold(&self) -> f32 {
        self.pass.0
    }

    pub fn validate_source(&self, source: EmbeddingSource) -> Result<()> {
        if let Some(expected) = self.source
            && expected != source
        {
            bail!(
                "threshold source mismatch: thresholds are for {:?}, score is from {:?}",
                expected,
                source
            );
        }
        Ok(())
    }
}

/// Compatibility API retained for existing callers that pass raw f32 scores.
pub fn stage1(score: f32, thresholds: &IdentityThresholds) -> Stage1Verdict {
    stage1_score(CosineScore(score), thresholds)
}

/// Typed QC-1 decision. Disabled auto decisions deliberately become Borderline
/// instead of pretending a tiny calibration sample proves a 1% error target.
pub fn stage1_score(score: CosineScore, thresholds: &IdentityThresholds) -> Stage1Verdict {
    if thresholds.auto_pass_enabled && score.0 >= thresholds.pass.0 {
        Stage1Verdict::Pass(score.0)
    } else if thresholds.auto_fail_enabled && score.0 < thresholds.fail.0 {
        Stage1Verdict::Fail(score.0)
    } else {
        Stage1Verdict::Borderline(score.0)
    }
}

pub fn stage1_embedding(
    score: CosineScore,
    source: EmbeddingSource,
    thresholds: &IdentityThresholds,
) -> Result<Stage1Verdict> {
    thresholds.validate_source(source)?;
    Ok(stage1_score(score, thresholds))
}

// -----------------------------------------------------------------------------
// 1. Per-avatar calibration
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimilaritySample {
    pub score: CosineScore,
    pub source: EmbeddingSource,
}

impl SimilaritySample {
    pub fn new(score: f32, source: EmbeddingSource) -> Result<Self> {
        Ok(Self {
            score: CosineScore::new(score)?,
            source,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Maximum tolerated false acceptance rate (different person auto-passed).
    pub target_fpr: f64,
    /// Maximum tolerated false rejection rate (same person auto-failed).
    pub target_fnr: f64,
    /// One-sided posterior confidence/credible level.
    pub confidence_level: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            target_fpr: 0.01,
            target_fnr: 0.05,
            confidence_level: 0.95,
        }
    }
}

/// Choose thresholds by enumerating observed score boundaries. The unknown
/// error rates use Jeffreys Beta(1/2, 1/2) priors; posterior upper bounds, not
/// raw empirical rates, must satisfy the requested FPR/FNR. This is much less
/// brittle than a Gaussian fit for small, non-normal biometric score samples.
///
/// With only a few dozen negatives, even zero observed false accepts cannot
/// establish FPR <= 1% at 95% confidence. Rather than overclaim, this returns
/// a disabled auto-pass gate (all such frames go to the judge) until M0 has
/// enough negatives. The same logic applies to unsafe auto-fail thresholds.
pub fn calibrate_thresholds(
    same_person: &[SimilaritySample],
    different_person: &[SimilaritySample],
    config: CalibrationConfig,
) -> Result<IdentityThresholds> {
    validate_calibration_config(config)?;
    if same_person.len() < 2 || different_person.len() < 2 {
        bail!("calibration requires at least two positive and two negative samples");
    }

    let source = same_person[0].source;
    if same_person.iter().chain(different_person).any(|sample| sample.source != source) {
        bail!("calibration sample set mixes embedding sources; calibrate each encoder separately");
    }
    for sample in same_person.iter().chain(different_person) {
        CosineScore::new(sample.score.0)?;
    }

    let positives: Vec<f32> = same_person.iter().map(|sample| sample.score.0).collect();
    let negatives: Vec<f32> = different_person.iter().map(|sample| sample.score.0).collect();
    let mut candidates = positives.clone();
    candidates.extend(negatives.iter().copied());
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let mut best_pass: Option<(f32, usize, f64)> = None;
    for threshold in candidates.iter().copied() {
        let false_accepts = negatives.iter().filter(|score| **score >= threshold).count();
        let upper_bound = beta_quantile(
            config.confidence_level,
            false_accepts as f64 + 0.5,
            (negatives.len() - false_accepts) as f64 + 0.5,
        );
        if upper_bound <= config.target_fpr {
            let true_accepts = positives.iter().filter(|score| **score >= threshold).count();
            let candidate = (threshold, true_accepts, upper_bound);
            let replace = best_pass
                .map(|best| true_accepts > best.1 || (true_accepts == best.1 && threshold < best.0))
                .unwrap_or(true);
            if replace {
                best_pass = Some(candidate);
            }
        }
    }

    let mut best_fail: Option<(f32, usize, f64)> = None;
    for threshold in candidates.iter().copied() {
        let false_rejects = positives.iter().filter(|score| **score < threshold).count();
        let upper_bound = beta_quantile(
            config.confidence_level,
            false_rejects as f64 + 0.5,
            (positives.len() - false_rejects) as f64 + 0.5,
        );
        if upper_bound <= config.target_fnr {
            let true_rejects = negatives.iter().filter(|score| **score < threshold).count();
            let candidate = (threshold, true_rejects, upper_bound);
            let replace = best_fail
                .map(|best| true_rejects > best.1 || (true_rejects == best.1 && threshold > best.0))
                .unwrap_or(true);
            if replace {
                best_fail = Some(candidate);
            }
        }
    }

    let auto_pass_enabled = best_pass.is_some();
    let mut auto_fail_enabled = best_fail.is_some();
    let pass = best_pass.map(|value| value.0).unwrap_or(1.0);
    let mut fail = best_fail.map(|value| value.0).unwrap_or(-1.0);
    if fail > pass {
        // Overlapping classes do not justify an inverted grey zone.
        // Keep the more safety-critical PASS boundary and abstain on FAIL.
        // (fail == pass is allowed: an empty grey zone is a valid outcome —
        // every score is then decided locally and none goes to the judge.)
        fail = -1.0;
        // The flag must follow the decision: a disabled gate that still
        // reports `auto_fail_enabled` would make diagnostics lie.
        auto_fail_enabled = false;
    }
    let diagnostics = CalibrationDiagnostics {
        source,
        positive_samples: positives.len(),
        negative_samples: negatives.len(),
        target_fpr: config.target_fpr,
        target_fnr: config.target_fnr,
        confidence_level: config.confidence_level,
        false_accept_upper_bound: best_pass.map(|value| value.2).unwrap_or(1.0),
        false_reject_upper_bound: best_fail.map(|value| value.2).unwrap_or(1.0),
        auto_pass_enabled,
        auto_fail_enabled,
    };

    Ok(IdentityThresholds {
        pass: ScoreThreshold(pass),
        fail: ScoreThreshold(fail),
        source: Some(source),
        auto_pass_enabled,
        auto_fail_enabled,
        calibration: Some(diagnostics),
    })
}

fn validate_calibration_config(config: CalibrationConfig) -> Result<()> {
    if !config.target_fpr.is_finite()
        || !config.target_fnr.is_finite()
        || !config.confidence_level.is_finite()
        || !(0.0..1.0).contains(&config.target_fpr)
        || !(0.0..1.0).contains(&config.target_fnr)
        || !(0.5..1.0).contains(&config.confidence_level)
    {
        bail!("invalid calibration config: rates must be in (0, 1), confidence in [0.5, 1)");
    }
    Ok(())
}

/// Inverse Beta CDF using a bounded bisection over the regularized incomplete
/// beta function. This avoids a statistics dependency and is stable at the
/// small sample sizes expected by M0 (dozens to hundreds of examples).
fn beta_quantile(probability: f64, alpha: f64, beta: f64) -> f64 {
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..90 {
        let middle = (low + high) * 0.5;
        if regularized_incomplete_beta(middle, alpha, beta) < probability {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) * 0.5
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 240;
    const EPSILON: f64 = 3.0e-14;
    const MIN_DENOMINATOR: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_DENOMINATOR {
        d = MIN_DENOMINATOR;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }
    h
}

fn ln_gamma(value: f64) -> f64 {
    // Lanczos approximation, g=7, adequate for the fractional Beta shapes
    // used here. Reflection keeps the approximation accurate for z < 1/2.
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - ln_gamma(1.0 - value);
    }
    let z = value - 1.0;
    let mut x = 0.999_999_999_999_809_9;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
        x += coefficient / (z + index as f64 + 1.0);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

// -----------------------------------------------------------------------------
// Shared camera-angle metadata
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaceViewAngle {
    Frontal,
    ThreeQuarter,
    Profile,
    #[default]
    Unknown,
}

fn normalize_angle_score(score: f32, angle: FaceViewAngle, offsets: &BTreeMap<FaceViewAngle, f32>) -> f32 {
    // Offsets are learned/declared from M0 angle-stratified data and are the
    // expected deficit (negative for a profile, e.g. -0.08). Subtracting the
    // offset restores the frontal scale. Absent a calibration entry, the core
    // does not invent an angle correction.
    score - offsets.get(&angle).copied().unwrap_or(0.0)
}

// -----------------------------------------------------------------------------
// 2. Small-N drift monitoring
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DriftObservation {
    /// Render order, not wall-clock time.
    pub shot_index: u64,
    pub score: CosineScore,
    pub view_angle: FaceViewAngle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftVerdict {
    Stable,
    DriftStarting,
    DriftCritical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftConfig {
    pub fail_threshold: f32,
    pub warning_slope_per_shot: f32,
    pub critical_slope_per_shot: f32,
    pub warning_drop: f32,
    pub critical_drop: f32,
    pub forecast_horizon_shots: f32,
    pub angle_offsets: BTreeMap<FaceViewAngle, f32>,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            fail_threshold: 0.58,
            warning_slope_per_shot: 0.005,
            critical_slope_per_shot: 0.025,
            warning_drop: 0.05,
            critical_drop: 0.14,
            forecast_horizon_shots: 5.0,
            angle_offsets: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftReport {
    pub verdict: DriftVerdict,
    /// Dimensionless severity; 1.0 is the configured warning boundary.
    pub drift_measure: f32,
    pub robust_slope_per_shot: f32,
    pub baseline_median: f32,
    pub recent_median: f32,
    pub last_score: f32,
    pub ignored_outliers: usize,
    pub observations_used: usize,
}

/// Theil–Sen slope (median of pairwise slopes) plus robust baseline/recent
/// medians. Theil–Sen resists a single outlier much better than OLS; a median
/// level shift detects a step change that a slope-only warning could delay.
/// For N<5 there is insufficient history for a trend verdict.
pub fn analyze_drift(observations: &[DriftObservation], config: &DriftConfig) -> Result<DriftReport> {
    validate_drift_config(config)?;
    if observations.is_empty() {
        bail!("drift analysis requires at least one observation");
    }
    for pair in observations.windows(2) {
        if pair[1].shot_index <= pair[0].shot_index {
            bail!("drift observations must be in strictly increasing render order");
        }
    }
    let normalized: Vec<(f64, f32)> = observations
        .iter()
        .map(|observation| {
            (
                observation.shot_index as f64,
                normalize_angle_score(observation.score.0, observation.view_angle, &config.angle_offsets),
            )
        })
        .collect();
    if normalized.iter().any(|(_, value)| !value.is_finite()) {
        bail!("angle normalization produced a non-finite score");
    }

    let values: Vec<f32> = normalized.iter().map(|(_, value)| *value).collect();
    let outlier_indices = isolated_outlier_indices(&values, 0.12, 0.035);
    let filtered: Vec<(f64, f32)> = normalized
        .iter()
        .enumerate()
        .filter(|(index, _)| !outlier_indices.contains(index))
        .map(|(_, observation)| *observation)
        .collect();
    let filtered = if filtered.is_empty() { normalized.clone() } else { filtered };
    let filtered_values: Vec<f32> = filtered.iter().map(|(_, value)| *value).collect();
    let last_score = *filtered_values.last().unwrap_or(&values[values.len() - 1]);

    if filtered.len() < 5 {
        let center = median(&filtered_values);
        return Ok(DriftReport {
            verdict: DriftVerdict::Stable,
            drift_measure: 0.0,
            robust_slope_per_shot: 0.0,
            baseline_median: center,
            recent_median: center,
            last_score,
            ignored_outliers: outlier_indices.len(),
            observations_used: filtered.len(),
        });
    }

    let baseline_count = (filtered.len() / 3).max(2).min(filtered.len() - 2);
    let recent_count = (filtered.len() / 2).max(2).min(filtered.len() - baseline_count);
    let baseline = median(&filtered_values[..baseline_count]);
    let recent = median(&filtered_values[filtered_values.len() - recent_count..]);
    let slope = theil_sen_slope(&filtered);
    let drop = (baseline - recent).max(0.0);
    let downward_slope = (-slope).max(0.0);
    let slope_measure = downward_slope / config.warning_slope_per_shot.max(f32::EPSILON);
    let drop_measure = drop / config.warning_drop.max(f32::EPSILON);
    let drift_measure = slope_measure.max(drop_measure);

    let projected_score = last_score - downward_slope * config.forecast_horizon_shots;
    let critical = last_score < config.fail_threshold
        || recent < config.fail_threshold
        || drop >= config.critical_drop
        || (downward_slope >= config.critical_slope_per_shot
            && projected_score < config.fail_threshold);
    let warning = critical
        || downward_slope >= config.warning_slope_per_shot
        || drop >= config.warning_drop;
    let verdict = if critical {
        DriftVerdict::DriftCritical
    } else if warning {
        DriftVerdict::DriftStarting
    } else {
        DriftVerdict::Stable
    };

    Ok(DriftReport {
        verdict,
        drift_measure,
        robust_slope_per_shot: slope,
        baseline_median: baseline,
        recent_median: recent,
        last_score,
        ignored_outliers: outlier_indices.len(),
        observations_used: filtered.len(),
    })
}

fn validate_drift_config(config: &DriftConfig) -> Result<()> {
    let values = [
        config.fail_threshold,
        config.warning_slope_per_shot,
        config.critical_slope_per_shot,
        config.warning_drop,
        config.critical_drop,
        config.forecast_horizon_shots,
    ];
    if values.iter().any(|value| !value.is_finite())
        || config.warning_slope_per_shot <= 0.0
        || config.critical_slope_per_shot <= 0.0
        || config.warning_drop <= 0.0
        || config.critical_drop <= 0.0
        || config.forecast_horizon_shots < 0.0
    {
        bail!("invalid drift configuration");
    }
    if config.angle_offsets.values().any(|value| !value.is_finite()) {
        bail!("angle offsets must be finite");
    }
    Ok(())
}

fn isolated_outlier_indices(values: &[f32], min_distance: f32, consensus_tolerance: f32) -> Vec<usize> {
    if values.len() < 3 {
        return Vec::new();
    }
    let mut outliers = Vec::new();
    for index in 0..values.len() {
        let remaining: Vec<f32> = values
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, value)| *value)
            .collect();
        if remaining.is_empty() {
            continue;
        }
        let min = remaining.iter().copied().fold(f32::INFINITY, f32::min);
        let max = remaining.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let center = median(&remaining);
        if max - min <= consensus_tolerance && (values[index] - center).abs() >= min_distance {
            outliers.push(index);
        }
    }
    if outliers.len() == 1 { outliers } else { Vec::new() }
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn theil_sen_slope(points: &[(f64, f32)]) -> f32 {
    let mut slopes = Vec::new();
    for left in 0..points.len() {
        for right in (left + 1)..points.len() {
            let dx = points[right].0 - points[left].0;
            if dx > 0.0 {
                slopes.push((f64::from(points[right].1 - points[left].1) / dx) as f32);
            }
        }
    }
    median(&slopes)
}

// -----------------------------------------------------------------------------
// 3. Multi-character global assignment (Hungarian algorithm)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentVerdict {
    Pass,
    Borderline,
    Fail,
    Unassigned,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceAssignment {
    pub face_index: usize,
    pub character_index: Option<usize>,
    pub similarity: Option<CosineScore>,
    pub verdict: AssignmentVerdict,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssignmentAnomaly {
    ExtraFace { face_index: usize },
    MissingCharacter { character_index: usize },
    SimilarCanons {
        first_character: usize,
        second_character: usize,
        similarity: CosineScore,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignmentResult {
    pub faces: Vec<FaceAssignment>,
    pub anomalies: Vec<AssignmentAnomaly>,
    pub matched_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AssignmentConfig {
    /// Warning threshold for two canon vectors that are too close to
    /// distinguish reliably. Calibrate this on the encoder and cast in M0.
    pub twin_canon_similarity_threshold: f32,
}

impl Default for AssignmentConfig {
    fn default() -> Self {
        Self {
            twin_canon_similarity_threshold: 0.92,
        }
    }
}

/// Maximum-weight one-to-one matching with explicit dummy rows/columns for
/// unmatched detections and missing characters. Hungarian gives the global
/// optimum; the per-avatar FAIL threshold is subtracted from each edge's
/// utility, so a weak face can be left unmatched instead of being forced onto
/// the least-wrong canon. This handles K!=M and K=M impostors uniformly.
pub fn assign_faces_to_canons(
    detected: &[FaceEmbedding],
    canons: &[FaceEmbedding],
    thresholds: &[IdentityThresholds],
    config: AssignmentConfig,
) -> Result<AssignmentResult> {
    if thresholds.len() != canons.len() {
        bail!("one identity threshold set is required per canon");
    }
    if !config.twin_canon_similarity_threshold.is_finite()
        || !(-1.0..=1.0).contains(&config.twin_canon_similarity_threshold)
    {
        bail!("invalid twin-canon similarity threshold");
    }

    for (canon, threshold) in canons.iter().zip(thresholds) {
        threshold.validate_source(canon.source)?;
    }

    let face_count = detected.len();
    let canon_count = canons.len();
    let mut similarities = vec![vec![0.0_f32; canon_count]; face_count];
    for (face_index, face) in detected.iter().enumerate() {
        for (canon_index, canon) in canons.iter().enumerate() {
            let similarity = compare_embeddings(face, canon)?;
            similarities[face_index][canon_index] = similarity.0;
        }
    }

    // Every real row/column gets a dedicated zero-utility dummy alternative.
    // N=K+M square assignment means the Hungarian solver can choose any
    // cardinality from 0 through min(K,M), rather than forcing min(K,M) matches.
    let size = face_count + canon_count;
    let mut weights = vec![vec![0.0_f64; size]; size];
    for face_index in 0..face_count {
        for canon_index in 0..canon_count {
            weights[face_index][canon_index] =
                f64::from(similarities[face_index][canon_index] - thresholds[canon_index].fail.0)
                    + 1.0e-9;
        }
    }
    let assignment = hungarian_max(&weights);

    let mut faces: Vec<FaceAssignment> = (0..face_count)
        .map(|face_index| FaceAssignment {
            face_index,
            character_index: None,
            similarity: None,
            verdict: AssignmentVerdict::Unassigned,
        })
        .collect();
    let mut canon_matched = vec![false; canon_count];
    let mut matched_count = 0;

    for face_index in 0..face_count {
        let column = assignment[face_index];
        if column >= canon_count {
            continue;
        }
        let similarity = similarities[face_index][column];
        if similarity <= thresholds[column].fail.0 {
            continue;
        }
        let score = CosineScore::new(similarity)?;
        let verdict = match stage1_score(score, &thresholds[column]) {
            Stage1Verdict::Pass(_) => AssignmentVerdict::Pass,
            Stage1Verdict::Borderline(_) => AssignmentVerdict::Borderline,
            Stage1Verdict::Fail(_) => AssignmentVerdict::Fail,
        };
        faces[face_index] = FaceAssignment {
            face_index,
            character_index: Some(column),
            similarity: Some(score),
            verdict,
        };
        canon_matched[column] = true;
        matched_count += 1;
    }

    let mut anomalies = Vec::new();
    for face in &faces {
        if face.character_index.is_none() {
            anomalies.push(AssignmentAnomaly::ExtraFace {
                face_index: face.face_index,
            });
        }
    }
    for (character_index, matched) in canon_matched.iter().enumerate() {
        if !matched {
            anomalies.push(AssignmentAnomaly::MissingCharacter { character_index });
        }
    }
    for first in 0..canon_count {
        for second in (first + 1)..canon_count {
            let similarity = compare_embeddings(&canons[first], &canons[second])?;
            if similarity.0 >= config.twin_canon_similarity_threshold {
                anomalies.push(AssignmentAnomaly::SimilarCanons {
                    first_character: first,
                    second_character: second,
                    similarity,
                });
            }
        }
    }

    Ok(AssignmentResult {
        faces,
        anomalies,
        matched_count,
    })
}

/// Original helper preserved for older callers. It now uses optimal global
/// assignment and calibrated default boundaries rather than per-face argmax.
pub fn each_to_own_canon(detected: &[Vec<f32>], canons: &[Vec<f32>]) -> bool {
    if detected.is_empty() {
        return true;
    }
    if canons.is_empty() {
        return false;
    }
    let detected_typed: Result<Vec<FaceEmbedding>> = detected
        .iter()
        .map(|vector| FaceEmbedding::new(vector.clone(), EmbeddingSource::LegacyUnspecified))
        .collect();
    let canons_typed: Result<Vec<FaceEmbedding>> = canons
        .iter()
        .map(|vector| FaceEmbedding::new(vector.clone(), EmbeddingSource::LegacyUnspecified))
        .collect();
    let (Ok(detected_typed), Ok(canons_typed)) = (detected_typed, canons_typed) else {
        return false;
    };
    let thresholds = vec![IdentityThresholds::default(); canons.len()];
    assign_faces_to_canons(
        &detected_typed,
        &canons_typed,
        &thresholds,
        AssignmentConfig::default(),
    )
    .map(|result| result.faces.iter().all(|face| face.verdict == AssignmentVerdict::Pass))
    .unwrap_or(false)
}

/// Hungarian minimum-cost assignment, adapted to a square dense matrix.
/// Returns one selected column per row. O(n^3), negligible for a scene cast
/// bounded to a handful of faces (the project spec currently targets <=5).
fn hungarian_max(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut p = vec![0_usize; n + 1];
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cost = -weights[i0 - 1][j - 1] - u[i0] - v[j];
                    if cost < minv[j] {
                        minv[j] = cost;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![0_usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

// -----------------------------------------------------------------------------
// 4. Clip score aggregation (QC-2)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipFrameScore {
    pub timestamp_seconds: f64,
    pub score: CosineScore,
    pub view_angle: FaceViewAngle,
    /// An upstream frame-quality detector may mark an obvious extraction /
    /// decode artefact. Marked frames are reported but excluded from trend.
    pub marked_artifact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipAggregationConfig {
    pub warning_drop: f32,
    pub critical_drop: f32,
    pub warning_slope_per_second: f32,
    pub critical_slope_per_second: f32,
    pub isolated_outlier_distance: f32,
    pub artifact_consensus_tolerance: f32,
    pub angle_offsets: BTreeMap<FaceViewAngle, f32>,
}

impl Default for ClipAggregationConfig {
    fn default() -> Self {
        Self {
            warning_drop: 0.05,
            critical_drop: 0.14,
            warning_slope_per_second: 0.01,
            critical_slope_per_second: 0.04,
            isolated_outlier_distance: 0.12,
            artifact_consensus_tolerance: 0.035,
            angle_offsets: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipDriftVerdict {
    None,
    DriftStarting,
    DriftCritical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipVerdict {
    Pass,
    Borderline,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipAggregationResult {
    pub verdict: ClipVerdict,
    pub base_level_score: CosineScore,
    pub base_level_verdict: Stage1Verdict,
    pub within_clip_drift: ClipDriftVerdict,
    pub first_valid_score: f32,
    pub last_valid_score: f32,
    pub within_clip_drop: f32,
    pub robust_slope_per_second: f32,
    /// Frames marked as artefacts upstream: excluded from the base level.
    pub ignored_artifacts: usize,
    /// Isolated outliers: kept in the base level, excluded from the trend.
    pub isolated_outliers: usize,
    pub frames_used_for_trend: usize,
}

/// Aggregate any number of timestamped QC frames. Median score is the robust
/// base level; a separate endpoint drop and Theil–Sen slope detect intra-clip
/// drift (including a good first frame and bad final frame). Frames are equally
/// weighted: positional weighting is rejected because the endpoints carry the
/// strongest continuity evidence; median aggregation already limits one
/// isolated bad frame's impact. An isolated outlier is omitted from the trend
/// only when all remaining frames agree tightly, avoiding a single artefact
/// being mistaken for sustained movement.
pub fn aggregate_clip_frames(
    frames: &[ClipFrameScore],
    thresholds: &IdentityThresholds,
    config: &ClipAggregationConfig,
) -> Result<ClipAggregationResult> {
    validate_clip_config(config)?;
    if frames.is_empty() {
        bail!("clip aggregation requires at least one frame score");
    }
    let mut ordered = frames.to_vec();
    ordered.sort_by(|left, right| left.timestamp_seconds.total_cmp(&right.timestamp_seconds));
    for frame in &ordered {
        if !frame.timestamp_seconds.is_finite() {
            bail!("clip frame timestamps must be finite");
        }
    }
    if ordered.windows(2).any(|pair| pair[1].timestamp_seconds <= pair[0].timestamp_seconds) {
        bail!("clip frame timestamps must be unique and strictly increasing");
    }

    let quality_frames: Vec<ClipFrameScore> = ordered
        .iter()
        .copied()
        .filter(|frame| !frame.marked_artifact)
        .collect();
    if quality_frames.is_empty() {
        bail!("all clip frames were marked as artefacts");
    }
    let normalized: Vec<f32> = quality_frames
        .iter()
        .map(|frame| normalize_angle_score(frame.score.0, frame.view_angle, &config.angle_offsets))
        .collect();
    if normalized.iter().any(|score| !score.is_finite()) {
        bail!("angle normalization produced a non-finite clip score");
    }
    let base_level_score = CosineScore::new(median(&normalized).clamp(-1.0, 1.0))?;
    let base_level_verdict = stage1_score(base_level_score, thresholds);

    // `ignored_artifacts` counts frames excluded from the BASE LEVEL (marked
    // upstream). Isolated outliers are still part of the median — they are only
    // dropped from the trend — so they are reported separately.
    let ignored_artifacts = ordered.len() - quality_frames.len();
    let isolated = isolated_outlier_indices(
        &normalized,
        config.isolated_outlier_distance,
        config.artifact_consensus_tolerance,
    );
    let trend_frames: Vec<ClipFrameScore> = quality_frames
        .iter()
        .enumerate()
        .filter(|(index, _)| !isolated.contains(index))
        .map(|(_, frame)| *frame)
        .collect();

    let (first_valid_score, last_valid_score, drop, slope) = if trend_frames.is_empty() {
        (base_level_score.0, base_level_score.0, 0.0, 0.0)
    } else if trend_frames.len() == 1 {
        let score = normalize_angle_score(
            trend_frames[0].score.0,
            trend_frames[0].view_angle,
            &config.angle_offsets,
        );
        (score, score, 0.0, 0.0)
    } else {
        let normalized_trend: Vec<(f64, f32)> = trend_frames
            .iter()
            .map(|frame| {
                (
                    frame.timestamp_seconds,
                    normalize_angle_score(frame.score.0, frame.view_angle, &config.angle_offsets),
                )
            })
            .collect();
        let first = normalized_trend[0].1;
        let last = normalized_trend[normalized_trend.len() - 1].1;
        let decline = (first - last).max(0.0);
        let robust_slope = theil_sen_slope(&normalized_trend);
        (first, last, decline, robust_slope)
    };
    let duration = trend_frames
        .first()
        .zip(trend_frames.last())
        .map(|(first, last)| (last.timestamp_seconds - first.timestamp_seconds) as f32)
        .unwrap_or(0.0);
    // A slope over two points is just the endpoint difference, not a robust
    // estimate — Theil–Sen needs >= 3 points before a slope signal is trusted.
    // With two trend frames only the endpoint drop is used.
    let slope_drop = if trend_frames.len() >= 3 {
        (-slope).max(0.0) * duration
    } else {
        0.0
    };
    let slope_per_second = if trend_frames.len() >= 3 { (-slope).max(0.0) } else { 0.0 };
    let effective_drop = drop.max(slope_drop);
    let current_is_below_fail = last_valid_score < thresholds.fail.0 && thresholds.auto_fail_enabled;
    let critical = current_is_below_fail
        || effective_drop >= config.critical_drop
        || slope_per_second >= config.critical_slope_per_second;
    let warning = critical
        || effective_drop >= config.warning_drop
        || slope_per_second >= config.warning_slope_per_second;
    let within_clip_drift = if critical {
        ClipDriftVerdict::DriftCritical
    } else if warning {
        ClipDriftVerdict::DriftStarting
    } else {
        ClipDriftVerdict::None
    };

    let verdict = if matches!(base_level_verdict, Stage1Verdict::Fail(_))
        || within_clip_drift == ClipDriftVerdict::DriftCritical
    {
        ClipVerdict::Fail
    } else if matches!(base_level_verdict, Stage1Verdict::Borderline(_))
        || within_clip_drift == ClipDriftVerdict::DriftStarting
    {
        ClipVerdict::Borderline
    } else {
        ClipVerdict::Pass
    };

    Ok(ClipAggregationResult {
        verdict,
        base_level_score,
        base_level_verdict,
        within_clip_drift,
        first_valid_score,
        last_valid_score,
        within_clip_drop: effective_drop,
        robust_slope_per_second: slope,
        ignored_artifacts,
        isolated_outliers: isolated.len(),
        frames_used_for_trend: trend_frames.len(),
    })
}

fn validate_clip_config(config: &ClipAggregationConfig) -> Result<()> {
    let values = [
        config.warning_drop,
        config.critical_drop,
        config.warning_slope_per_second,
        config.critical_slope_per_second,
        config.isolated_outlier_distance,
        config.artifact_consensus_tolerance,
    ];
    if values.iter().any(|value| !value.is_finite() || *value < 0.0)
        || config.warning_slope_per_second == 0.0
        || config.critical_slope_per_second == 0.0
        || config.warning_drop == 0.0
        || config.critical_drop == 0.0
    {
        bail!("invalid clip aggregation configuration");
    }
    if config.angle_offsets.values().any(|value| !value.is_finite()) {
        bail!("clip angle offsets must be finite");
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// 5. Embedding + FLAME geometry fusion
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FusionTrainingExample {
    pub embedding_score: CosineScore,
    pub source: EmbeddingSource,
    pub geometry_distance: Option<GeometryDistance>,
    pub view_angle: FaceViewAngle,
    pub same_person: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FusionFitConfig {
    /// M0 scale converting FLAME RMS distance to exp(-distance / scale).
    pub geometry_distance_scale: f32,
    pub learning_rate: f64,
    pub iterations: usize,
    pub l2_regularization: f64,
    pub threshold_calibration: CalibrationConfig,
}

impl Default for FusionFitConfig {
    fn default() -> Self {
        Self {
            geometry_distance_scale: 0.5,
            learning_rate: 0.05,
            iterations: 3_000,
            l2_regularization: 0.05,
            threshold_calibration: CalibrationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionModel {
    pub source: EmbeddingSource,
    pub geometry_distance_scale: f32,
    /// [bias, embedding, geometry, profile×embedding, profile×geometry].
    /// The profile interaction terms make the learned channel weights
    /// explicitly view-angle-dependent without hand-written weighted averages.
    pub coefficients: [f64; 5],
    pub embedding_thresholds: IdentityThresholds,
    pub geometry_thresholds: IdentityThresholds,
    pub fused_thresholds: IdentityThresholds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionResult {
    pub combined_probability: Option<f32>,
    pub verdict: Stage1Verdict,
    pub embedding_verdict: Stage1Verdict,
    pub geometry_verdict: Option<Stage1Verdict>,
    pub channel_conflict: bool,
}

/// Fit an L2-regularized logistic model using tiny deterministic gradient
/// descent. The profile interaction terms allow the data to learn whether
/// geometry deserves more weight on profiles; we do not hard-code fusion
/// weights. L2 regularization is preferred to a high-capacity model for M0's
/// small sample count. Use held-out M0 examples before production rollout.
pub fn fit_fusion_model(
    examples: &[FusionTrainingExample],
    config: FusionFitConfig,
) -> Result<FusionModel> {
    validate_fusion_config(config)?;
    if examples.is_empty() {
        bail!("fusion fit requires labelled examples");
    }
    let source = examples[0].source;
    if examples.iter().any(|example| example.source != source) {
        bail!("fusion calibration mixes embedding sources; fit one model per encoder");
    }
    let complete: Vec<FusionTrainingExample> = examples
        .iter()
        .copied()
        .filter(|example| example.geometry_distance.is_some())
        .collect();
    let positive_count = complete.iter().filter(|example| example.same_person).count();
    let negative_count = complete.len() - positive_count;
    if positive_count < 2 || negative_count < 2 {
        bail!("fusion fit needs at least two positive and two negative paired geometry samples");
    }

    let mut feature_rows = Vec::with_capacity(complete.len());
    let mut labels = Vec::with_capacity(complete.len());
    for example in &complete {
        let geometry = distance_to_similarity(
            example.geometry_distance.expect("filtered to fitted geometry"),
            config.geometry_distance_scale,
        )?;
        feature_rows.push(fusion_features(example.embedding_score.0, geometry, example.view_angle));
        labels.push(if example.same_person { 1.0 } else { 0.0 });
    }

    let mut coefficients = [0.0_f64; 5];
    for _ in 0..config.iterations {
        let mut gradients = [0.0_f64; 5];
        for (features, label) in feature_rows.iter().zip(&labels) {
            let probability = sigmoid(dot5(&coefficients, features));
            let error = probability - label;
            for index in 0..5 {
                gradients[index] += error * features[index];
            }
        }
        let count = feature_rows.len() as f64;
        for index in 0..5 {
            let regularization = if index == 0 { 0.0 } else { config.l2_regularization * coefficients[index] };
            coefficients[index] -= config.learning_rate * (gradients[index] / count + regularization);
        }
    }

    let mut embedding_positive = Vec::new();
    let mut embedding_negative = Vec::new();
    let mut geometry_positive = Vec::new();
    let mut geometry_negative = Vec::new();
    let mut fused_positive = Vec::new();
    let mut fused_negative = Vec::new();
    for (example, features) in complete.iter().zip(&feature_rows) {
        let geometry = distance_to_similarity(
            example.geometry_distance.expect("filtered to fitted geometry"),
            config.geometry_distance_scale,
        )?;
        let predicted = sigmoid(dot5(&coefficients, features)) as f32;
        let embedding_sample = SimilaritySample::new(example.embedding_score.0, source)?;
        let geometry_sample = SimilaritySample::new(geometry, source)?;
        let fused_sample = SimilaritySample::new(predicted, source)?;
        if example.same_person {
            embedding_positive.push(embedding_sample);
            geometry_positive.push(geometry_sample);
            fused_positive.push(fused_sample);
        } else {
            embedding_negative.push(embedding_sample);
            geometry_negative.push(geometry_sample);
            fused_negative.push(fused_sample);
        }
    }

    let embedding_thresholds = calibrate_thresholds(
        &embedding_positive,
        &embedding_negative,
        config.threshold_calibration,
    )?;
    let mut geometry_thresholds = calibrate_thresholds(
        &geometry_positive,
        &geometry_negative,
        config.threshold_calibration,
    )?;
    // Geometry is a separate numeric channel, not an embedding encoder space.
    // Its paired training data is still versioned to the encoder above.
    geometry_thresholds.source = None;
    let fused_thresholds = calibrate_thresholds(
        &fused_positive,
        &fused_negative,
        config.threshold_calibration,
    )?;

    Ok(FusionModel {
        source,
        geometry_distance_scale: config.geometry_distance_scale,
        coefficients,
        embedding_thresholds,
        geometry_thresholds,
        fused_thresholds,
    })
}

fn validate_fusion_config(config: FusionFitConfig) -> Result<()> {
    if !config.geometry_distance_scale.is_finite()
        || config.geometry_distance_scale <= 0.0
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || config.iterations == 0
        || !config.l2_regularization.is_finite()
        || config.l2_regularization < 0.0
    {
        bail!("invalid fusion fit configuration");
    }
    validate_calibration_config(config.threshold_calibration)
}

fn fusion_features(embedding: f32, geometry: f32, angle: FaceViewAngle) -> [f64; 5] {
    let profile_factor = match angle {
        FaceViewAngle::Frontal => 0.0,
        FaceViewAngle::ThreeQuarter => 0.5,
        FaceViewAngle::Profile => 1.0,
        FaceViewAngle::Unknown => 0.25,
    };
    [
        1.0,
        f64::from(embedding),
        f64::from(geometry),
        f64::from(embedding * profile_factor),
        f64::from(geometry * profile_factor),
    ]
}

fn dot5(left: &[f64; 5], right: &[f64; 5]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Fuse already-computed channel scores. If the geometry fit is absent or no
/// fusion model has been calibrated, this falls back to the embedding-only
/// decision with the same API. Opposing confident channel decisions always
/// route to Borderline; the logistic average is not allowed to hide conflict.
pub fn fuse_identity_signals(
    embedding_score: CosineScore,
    embedding_source: EmbeddingSource,
    geometry_distance: Option<GeometryDistance>,
    view_angle: FaceViewAngle,
    fallback_thresholds: &IdentityThresholds,
    model: Option<&FusionModel>,
) -> Result<FusionResult> {
    fallback_thresholds.validate_source(embedding_source)?;
    let Some(distance) = geometry_distance else {
        let embedding_verdict = stage1_score(embedding_score, fallback_thresholds);
        return Ok(FusionResult {
            combined_probability: None,
            verdict: embedding_verdict,
            embedding_verdict,
            geometry_verdict: None,
            channel_conflict: false,
        });
    };
    let Some(model) = model else {
        let embedding_verdict = stage1_score(embedding_score, fallback_thresholds);
        return Ok(FusionResult {
            combined_probability: None,
            verdict: embedding_verdict,
            embedding_verdict,
            geometry_verdict: None,
            channel_conflict: false,
        });
    };
    if model.source != embedding_source {
        bail!(
            "fusion model source mismatch: model={:?}, embedding={:?}",
            model.source,
            embedding_source
        );
    }

    let embedding_verdict = stage1_embedding(embedding_score, embedding_source, &model.embedding_thresholds)?;
    let geometry_similarity = distance_to_similarity(distance, model.geometry_distance_scale)?;
    let geometry_verdict = stage1_score(CosineScore::new(geometry_similarity)?, &model.geometry_thresholds);
    let embedding_pass = matches!(embedding_verdict, Stage1Verdict::Pass(_));
    let embedding_fail = matches!(embedding_verdict, Stage1Verdict::Fail(_));
    let geometry_pass = matches!(geometry_verdict, Stage1Verdict::Pass(_));
    let geometry_fail = matches!(geometry_verdict, Stage1Verdict::Fail(_));
    let channel_conflict = (embedding_pass && geometry_fail) || (embedding_fail && geometry_pass);

    let features = fusion_features(embedding_score.0, geometry_similarity, view_angle);
    let probability = sigmoid(dot5(&model.coefficients, &features)) as f32;
    let fused_verdict = if channel_conflict {
        Stage1Verdict::Borderline(probability)
    } else {
        stage1_score(CosineScore::new(probability)?, &model.fused_thresholds)
    };

    Ok(FusionResult {
        combined_probability: Some(probability),
        verdict: fused_verdict,
        embedding_verdict,
        geometry_verdict: Some(geometry_verdict),
        channel_conflict,
    })
}

// -----------------------------------------------------------------------------
// Unit tests: synthetic known-answer cases for all scoring primitives
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(score: f32, source: EmbeddingSource) -> SimilaritySample {
        SimilaritySample::new(score, source).unwrap()
    }

    fn observations(scores: &[f32]) -> Vec<DriftObservation> {
        scores
            .iter()
            .enumerate()
            .map(|(index, score)| DriftObservation {
                shot_index: index as u64,
                score: CosineScore::new(*score).unwrap(),
                view_angle: FaceViewAngle::Frontal,
            })
            .collect()
    }

    fn embedding(x: f32, y: f32) -> FaceEmbedding {
        FaceEmbedding::new(vec![x, y], EmbeddingSource::ArcFaceOnnx).unwrap()
    }

    #[test]
    fn calibration_is_source_specific_and_disables_unsafe_small_sample_passes() {
        let positives: Vec<_> = (0..24).map(|i| sample(0.88 + i as f32 * 0.001, EmbeddingSource::ArcFaceOnnx)).collect();
        let negatives: Vec<_> = (0..24).map(|i| sample(0.20 + i as f32 * 0.001, EmbeddingSource::ArcFaceOnnx)).collect();
        let thresholds = calibrate_thresholds(&positives, &negatives, CalibrationConfig::default()).unwrap();
        assert!(!thresholds.auto_pass_enabled, "24 negatives cannot certify 1% FPR at 95% confidence");
        assert_eq!(thresholds.source, Some(EmbeddingSource::ArcFaceOnnx));
        assert!(calibrate_thresholds(
            &positives,
            &[sample(0.2, EmbeddingSource::VisionFal), sample(0.3, EmbeddingSource::VisionFal)],
            CalibrationConfig::default()
        ).is_err());

        let positives: Vec<_> = (0..40).map(|i| sample(0.90 + (i % 5) as f32 * 0.001, EmbeddingSource::ArcFaceOnnx)).collect();
        let negatives: Vec<_> = (0..400).map(|i| sample(0.10 + (i % 30) as f32 * 0.001, EmbeddingSource::ArcFaceOnnx)).collect();
        let thresholds = calibrate_thresholds(&positives, &negatives, CalibrationConfig::default()).unwrap();
        assert!(thresholds.auto_pass_enabled);
        assert!(thresholds.pass.0 > 0.85);
        assert!(thresholds.calibration.unwrap().false_accept_upper_bound <= 0.01);
    }

    #[test]
    fn hungarian_assignment_handles_swaps_extra_faces_and_missing_characters() {
        let canons = vec![embedding(1.0, 0.0), embedding(0.0, 1.0)];
        let thresholds = vec![IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx); 2];
        let detected = vec![embedding(0.0, 1.0), embedding(1.0, 0.0)];
        let result = assign_faces_to_canons(&detected, &canons, &thresholds, AssignmentConfig::default()).unwrap();
        assert_eq!(result.faces[0].character_index, Some(1));
        assert_eq!(result.faces[1].character_index, Some(0));
        assert_eq!(result.matched_count, 2);

        let extra = vec![embedding(1.0, 0.0), embedding(0.0, 1.0), embedding(-1.0, 0.0)];
        let result = assign_faces_to_canons(&extra, &canons, &thresholds, AssignmentConfig::default()).unwrap();
        assert!(result.anomalies.iter().any(|anomaly| matches!(anomaly, AssignmentAnomaly::ExtraFace { .. })));

        let one_face = vec![embedding(1.0, 0.0)];
        let result = assign_faces_to_canons(&one_face, &canons, &thresholds, AssignmentConfig::default()).unwrap();
        assert!(result.anomalies.iter().any(|anomaly| matches!(anomaly, AssignmentAnomaly::MissingCharacter { character_index: 1 })));
    }

    #[test]
    fn assignment_warns_for_twin_canons_and_rejects_impostor() {
        let canons = vec![embedding(1.0, 0.0), embedding(0.99, 0.1)];
        let thresholds = vec![IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx); 2];
        let result = assign_faces_to_canons(
            &[embedding(-1.0, 0.0)],
            &canons,
            &thresholds,
            AssignmentConfig::default(),
        ).unwrap();
        assert!(result.faces[0].character_index.is_none());
        assert!(result.anomalies.iter().any(|anomaly| matches!(anomaly, AssignmentAnomaly::SimilarCanons { .. })));
    }

    #[test]
    fn drift_detects_linear_and_step_changes_but_ignores_one_outlier() {
        let stable = analyze_drift(&observations(&[0.90, 0.89, 0.90, 0.89, 0.90, 0.89]), &DriftConfig::default()).unwrap();
        assert_eq!(stable.verdict, DriftVerdict::Stable);
        let linear = analyze_drift(&observations(&[0.94, 0.90, 0.86, 0.82, 0.78, 0.74]), &DriftConfig::default()).unwrap();
        assert_ne!(linear.verdict, DriftVerdict::Stable);
        let step = analyze_drift(&observations(&[0.92, 0.91, 0.70, 0.69, 0.70, 0.69]), &DriftConfig::default()).unwrap();
        assert_eq!(step.verdict, DriftVerdict::DriftCritical);
        let outlier = analyze_drift(&observations(&[0.90, 0.90, 0.52, 0.90, 0.90, 0.90]), &DriftConfig::default()).unwrap();
        assert_eq!(outlier.verdict, DriftVerdict::Stable);
        assert_eq!(outlier.ignored_outliers, 1);
    }

    #[test]
    fn drift_normalizes_calibrated_profile_offsets() {
        let mut config = DriftConfig::default();
        config.angle_offsets.insert(FaceViewAngle::Profile, -0.08);
        let angles = [
            FaceViewAngle::Frontal,
            FaceViewAngle::Profile,
            FaceViewAngle::Frontal,
            FaceViewAngle::Profile,
            FaceViewAngle::Frontal,
            FaceViewAngle::Profile,
        ];
        let values = [0.90, 0.82, 0.90, 0.82, 0.90, 0.82];
        let input: Vec<_> = values.iter().zip(angles).enumerate().map(|(index, (score, angle))| DriftObservation {
            shot_index: index as u64,
            score: CosineScore::new(*score).unwrap(),
            view_angle: angle,
        }).collect();
        assert_eq!(analyze_drift(&input, &config).unwrap().verdict, DriftVerdict::Stable);
    }

    #[test]
    fn clip_aggregation_catches_intra_clip_decline_and_ignores_one_isolated_artifact() {
        let thresholds = IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx);
        let frames = [
            ClipFrameScore { timestamp_seconds: 0.0, score: CosineScore(0.92), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
            ClipFrameScore { timestamp_seconds: 1.0, score: CosineScore(0.82), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
            ClipFrameScore { timestamp_seconds: 2.0, score: CosineScore(0.61), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
        ];
        let drift = aggregate_clip_frames(&frames, &thresholds, &ClipAggregationConfig::default()).unwrap();
        assert_eq!(drift.verdict, ClipVerdict::Fail);
        assert!(drift.within_clip_drift != ClipDriftVerdict::None);

        let artifact = [
            ClipFrameScore { timestamp_seconds: 0.0, score: CosineScore(0.91), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
            ClipFrameScore { timestamp_seconds: 1.0, score: CosineScore(0.90), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
            ClipFrameScore { timestamp_seconds: 2.0, score: CosineScore(0.35), view_angle: FaceViewAngle::Frontal, marked_artifact: false },
        ];
        let robust = aggregate_clip_frames(&artifact, &thresholds, &ClipAggregationConfig::default()).unwrap();
        assert_eq!(robust.within_clip_drift, ClipDriftVerdict::None);
        // The outlier stays in the base level (median) and leaves the trend.
        assert_eq!(robust.ignored_artifacts, 0);
        assert_eq!(robust.isolated_outliers, 1);
        assert_eq!(robust.frames_used_for_trend, 2);
    }

    #[test]
    fn fusion_learns_channel_weights_and_escalates_opposing_signals() {
        let mut examples = Vec::new();
        for index in 0..40 {
            let positive = index % 2 == 0;
            let embedding_score = if positive { 0.90 + (index % 4) as f32 * 0.005 } else { 0.20 + (index % 4) as f32 * 0.005 };
            let geometry_distance = if positive { 0.08 + (index % 3) as f32 * 0.01 } else { 0.95 + (index % 3) as f32 * 0.02 };
            examples.push(FusionTrainingExample {
                embedding_score: CosineScore(embedding_score),
                source: EmbeddingSource::ArcFaceOnnx,
                geometry_distance: Some(GeometryDistance(geometry_distance)),
                view_angle: if index % 3 == 0 { FaceViewAngle::Profile } else { FaceViewAngle::Frontal },
                same_person: positive,
            });
        }
        let config = FusionFitConfig {
            threshold_calibration: CalibrationConfig { target_fpr: 0.2, target_fnr: 0.2, confidence_level: 0.8 },
            ..FusionFitConfig::default()
        };
        let model = fit_fusion_model(&examples, config).unwrap();
        assert!(model.coefficients[1].is_finite());
        let fallback = IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx);
        let contradiction = fuse_identity_signals(
            CosineScore(0.96),
            EmbeddingSource::ArcFaceOnnx,
            Some(GeometryDistance(2.0)),
            FaceViewAngle::Profile,
            &fallback,
            Some(&model),
        ).unwrap();
        assert!(contradiction.channel_conflict);
        assert!(matches!(contradiction.verdict, Stage1Verdict::Borderline(_)));

        let no_geometry = fuse_identity_signals(
            CosineScore(0.90),
            EmbeddingSource::ArcFaceOnnx,
            None,
            FaceViewAngle::Frontal,
            &fallback,
            Some(&model),
        ).unwrap();
        assert_eq!(no_geometry.verdict, Stage1Verdict::Pass(0.90));
        assert!(no_geometry.combined_probability.is_none());
    }

    #[test]
    fn fusion_rejects_different_embedding_sources() {
        let model = FusionModel {
            source: EmbeddingSource::ArcFaceOnnx,
            geometry_distance_scale: 0.5,
            coefficients: [0.0; 5],
            embedding_thresholds: IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx),
            geometry_thresholds: IdentityThresholds::default(),
            fused_thresholds: IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx),
        };
        assert!(fuse_identity_signals(
            CosineScore(0.9),
            EmbeddingSource::VisionFal,
            Some(GeometryDistance(0.1)),
            FaceViewAngle::Frontal,
            &IdentityThresholds::for_source(EmbeddingSource::VisionFal),
            Some(&model),
        ).is_err());
    }

    #[test]
    fn avatar_pass_threshold_never_inverts_the_gate() {
        // A per-avatar threshold below the default fail boundary used to leave
        // pass < fail, so every score >= pass auto-passed — including scores
        // the fail boundary existed to reject.
        let thresholds = IdentityThresholds::with_avatar_pass_threshold(
            EmbeddingSource::ArcFaceOnnx,
            Some(0.55),
        )
        .unwrap();
        assert!(thresholds.fail.0 <= thresholds.pass.0);
        // The grey zone collapses to empty: 0.55 passes, 0.54 is rejected.
        assert!(matches!(
            stage1_score(CosineScore(0.55), &thresholds),
            Stage1Verdict::Pass(_)
        ));
        assert!(matches!(
            stage1_score(CosineScore(0.54), &thresholds),
            Stage1Verdict::Fail(_)
        ));
        assert!(IdentityThresholds::with_avatar_pass_threshold(
            EmbeddingSource::ArcFaceOnnx,
            Some(1.5)
        )
        .is_err());
    }

    #[test]
    fn beta_quantile_matches_known_distributions() {
        // Beta(1,1) is uniform: the quantile is the probability itself.
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let q = beta_quantile(p, 1.0, 1.0);
            assert!((q - p).abs() < 1e-6, "uniform quantile {p} -> {q}");
        }
        // Beta(1/2,1/2) is the arcsine distribution: F(x) = 2/pi * asin(sqrt x).
        // Its median is 0.5 and its 25th percentile is sin^2(pi/8).
        assert!((beta_quantile(0.5, 0.5, 0.5) - 0.5).abs() < 1e-4);
        let expected = (std::f64::consts::PI / 8.0).sin().powi(2);
        assert!((beta_quantile(0.25, 0.5, 0.5) - expected).abs() < 1e-4);
        // Monotone in the probability.
        assert!(beta_quantile(0.9, 2.0, 5.0) > beta_quantile(0.1, 2.0, 5.0));
    }

    #[test]
    fn hungarian_finds_the_global_optimum_not_the_greedy_one() {
        // Greedy (face 0 first) takes c1 at 0.866 and leaves face 1 unmatched
        // (0.367 < fail). The optimum swaps: 0.663 + 0.985.
        let angle = |degrees: f32| {
            let radians = degrees.to_radians();
            embedding(radians.cos(), radians.sin())
        };
        let canons = vec![angle(0.0), angle(78.46)];
        let detected = vec![angle(30.0), angle(10.0)];
        let thresholds = vec![IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx); 2];
        let result =
            assign_faces_to_canons(&detected, &canons, &thresholds, AssignmentConfig::default())
                .unwrap();
        assert_eq!(result.matched_count, 2, "greedy would leave one face unmatched");
        assert_eq!(result.faces[0].character_index, Some(1));
        assert_eq!(result.faces[1].character_index, Some(0));
    }

    #[test]
    fn angle_offsets_restore_the_frontal_scale() {
        let mut offsets = BTreeMap::new();
        offsets.insert(FaceViewAngle::Profile, -0.08);
        assert!((normalize_angle_score(0.82, FaceViewAngle::Profile, &offsets) - 0.90).abs() < 1e-6);
        assert!((normalize_angle_score(0.82, FaceViewAngle::Frontal, &offsets) - 0.82).abs() < 1e-6);
        // No calibration entry → no invented correction.
        assert!((normalize_angle_score(0.82, FaceViewAngle::Unknown, &offsets) - 0.82).abs() < 1e-6);
    }

    #[test]
    fn calibration_flag_matches_the_disabled_fail_gate() {
        // Perfectly separated classes: the pass boundary lands just above the
        // negative maximum, the fail boundary would land above it, and the
        // inverted grey zone must disable auto-fail *and* say so.
        // Distinct values (no repeats): the counts below are per-sample.
        let positives: Vec<_> = (0..400)
            .map(|i| sample(0.900 + i as f32 * 0.000_01, EmbeddingSource::ArcFaceOnnx))
            .collect();
        let negatives: Vec<_> = (0..400)
            .map(|i| sample(0.100 + i as f32 * 0.000_01, EmbeddingSource::ArcFaceOnnx))
            .collect();
        let thresholds =
            calibrate_thresholds(&positives, &negatives, CalibrationConfig::default()).unwrap();
        assert!(thresholds.auto_pass_enabled);
        assert!(!thresholds.auto_fail_enabled);
        assert_eq!(thresholds.fail.0, -1.0);
        assert!(thresholds.fail.0 <= thresholds.pass.0);
        let diagnostics = thresholds.calibration.unwrap();
        assert!(!diagnostics.auto_fail_enabled);
    }
}
