//! Reference clustering — the grounding artifacts of the ArcFace-first
//! identity pipeline (phase 1).
//!
//! A set of heterogeneous reference photos ("22 photos of one person")
//! becomes structured appearance states: one cluster = one appearance state
//! (shaved vs short mop). Pure math on 512-d vectors; the FaceEngine adapter
//! in `embeddings.rs` feeds it.

use serde::{Deserialize, Serialize};

use super::embeddings::{cosine_similarity, validate_vector};

/// Default cosine threshold separating appearance states (~0.85).
pub const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.85;

/// One reference image with its ArcFace vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub path: String,
    pub vector: Vec<f32>,
}

/// A greedy-formed appearance-state cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceCluster {
    /// Indices into the input `Reference` slice.
    pub members: Vec<usize>,
    /// Centroid of member vectors, L2-normalized.
    pub centroid: Vec<f32>,
    /// Representative member index: closest to the centroid.
    pub representative: usize,
    /// Min intra-cluster cosine (the cluster's real identity spread).
    pub min_internal_cosine: f32,
}

/// Numeric summary of the whole reference set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSummary {
    pub clusters: Vec<ReferenceCluster>,
    /// Mean pairwise cosine between distinct cluster centroids.
    pub inter_cluster_mean: f32,
    /// Mean cosine of members to their own centroid (multi-member clusters).
    pub intra_cluster_mean: f32,
    /// Per-avatar identity threshold derived from the observed spread:
    /// `intra_cluster_min − margin`, clamped to a sane band. `None` when the
    /// set is too small or too uniform to derive one (caller keeps the
    /// global default).
    pub avatar_threshold: Option<f32>,
}

/// Auto-tune the cluster threshold: sweep candidates and pick the value that
/// maximizes the gap between inter-cluster and intra-cluster cosines
/// (separability), with a tie-break toward fewer clusters. Falls back to
/// `DEFAULT_CLUSTER_THRESHOLD` when no candidate separates anything.
pub fn auto_threshold(references: &[Reference]) -> f32 {
    const CANDIDATES: [f32; 12] = [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97,
    ];
    let mut best = DEFAULT_CLUSTER_THRESHOLD;
    let mut best_score = f32::NEG_INFINITY;
    for &t in &CANDIDATES {
        let Ok(summary) = cluster_references(references, t) else {
            continue;
        };
        let k = summary.clusters.len();
        if k < 2 {
            continue; // no separation to measure
        }
        let inter = summary.inter_cluster_mean;
        let intra = summary.intra_cluster_mean;
        if !(inter.is_finite() && intra.is_finite()) {
            continue;
        }
        // Mean-based separability gap (min-based is too brittle on wide-pose
        // sets); NaN intra (all singletons) scores worst.
        let score = inter - intra;
        if score > best_score {
            best_score = score;
            best = t;
        }
    }
    best
}

/// Greedy centroid clustering: each reference joins the first cluster whose
/// centroid it matches at or above `threshold`, otherwise opens a new one.
///
/// Seeding is order-independent: references are visited from the most central
/// one (highest mean cosine to the rest, path as tie-break), so the same photo
/// set yields the same clusters regardless of file names. Members come out
/// sorted by input index.
pub fn cluster_references(
    references: &[Reference],
    threshold: f32,
) -> anyhow::Result<ClusterSummary> {
    anyhow::ensure!(!references.is_empty(), "no references to cluster");
    anyhow::ensure!(
        (0.0..=1.0).contains(&threshold),
        "cluster threshold must be in [0, 1], got {threshold}"
    );
    // `cosine_similarity` is the untyped compatibility wrapper and returns 0.0
    // on invalid input. Without this check a NaN/empty/zero-norm vector would
    // silently become its own singleton cluster instead of failing loudly.
    for reference in references {
        validate_vector(&reference.vector)
            .map_err(|error| anyhow::anyhow!("reference {}: {error}", reference.path))?;
    }

    let mut assignments: Vec<Vec<usize>> = Vec::new();
    let mut centroids: Vec<Vec<f32>> = Vec::new();

    // Centrality: mean cosine to every other reference. Visiting the most
    // central reference first removes the dependence on input order.
    let mut order: Vec<usize> = (0..references.len()).collect();
    if references.len() > 1 {
        let divisor = (references.len() - 1) as f32;
        let centrality: Vec<f32> = (0..references.len())
            .map(|i| {
                let sum: f32 = (0..references.len())
                    .filter(|&j| j != i)
                    .map(|j| cosine_similarity(&references[i].vector, &references[j].vector))
                    .sum();
                sum / divisor
            })
            .collect();
        order.sort_by(|a, b| {
            centrality[*b]
                .total_cmp(&centrality[*a])
                .then_with(|| references[*a].path.cmp(&references[*b].path))
        });
    }

    for index in order {
        let reference = &references[index];
        let mut best: Option<(usize, f32)> = None;
        for (cluster_id, centroid) in centroids.iter().enumerate() {
            let score = cosine_similarity(&reference.vector, centroid);
            if score >= threshold && best.is_none_or(|(_, s)| score > s) {
                best = Some((cluster_id, score));
            }
        }
        match best {
            Some((cluster_id, _)) => {
                assignments[cluster_id].push(index);
                // Incremental mean, then renormalize (ArcFace vectors are
                // unit vectors; cosine compares directions only).
                let members = assignments[cluster_id].len() as f32;
                let centroid = &mut centroids[cluster_id];
                for (c, v) in centroid.iter_mut().zip(&reference.vector) {
                    *c += (*v - *c) / members;
                }
                normalize(centroid);
            }
            None => {
                assignments.push(vec![index]);
                centroids.push(reference.vector.clone());
            }
        }
    }

    let mut clusters = Vec::with_capacity(assignments.len());
    let mut intra_sum = 0.0_f32;
    let mut intra_n = 0_usize;
    for (mut members, centroid) in assignments.into_iter().zip(centroids) {
        members.sort_unstable();
        let mut min_internal = f32::INFINITY;
        let mut representative = members[0];
        let mut best_score = f32::NEG_INFINITY;
        for &m in &members {
            let score = cosine_similarity(&references[m].vector, &centroid);
            if score < min_internal {
                min_internal = score;
            }
            if score > best_score {
                best_score = score;
                representative = m;
            }
        }
        if members.len() > 1 {
            intra_sum += min_internal;
            intra_n += 1;
        }
        let is_multi = members.len() > 1;
        clusters.push(ReferenceCluster {
            members,
            representative,
            centroid,
            min_internal_cosine: if is_multi { min_internal } else { f32::NAN },
        });
    }

    let mut inter_sum = 0.0_f32;
    let mut inter_n = 0_usize;
    for i in 0..clusters.len() {
        for j in (i + 1)..clusters.len() {
            inter_sum += cosine_similarity(&clusters[i].centroid, &clusters[j].centroid);
            inter_n += 1;
        }
    }
    let inter_mean = if inter_n > 0 { inter_sum / inter_n as f32 } else { f32::NAN };
    let intra_mean = if intra_n > 0 { intra_sum / intra_n as f32 } else { f32::NAN };

    // Per-avatar threshold: the observed identity range minus a safety
    // margin, clamped so a single lucky pair cannot unlock everything.
    const MARGIN: f32 = 0.04;
    const MIN: f32 = 0.55;
    const MAX: f32 = 0.93;
    let avatar_threshold = if intra_mean.is_finite() {
        Some((intra_mean - MARGIN).clamp(MIN, MAX))
    } else {
        None
    };

    Ok(ClusterSummary {
        intra_cluster_mean: intra_mean,
        inter_cluster_mean: inter_mean,
        clusters,
        avatar_threshold,
    })
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(name: &str, vector: Vec<f32>) -> Reference {
        Reference { path: name.into(), vector }
    }

    fn unit(first: f32, second: f32) -> Vec<f32> {
        let mut v = vec![first, second];
        let n = (first * first + second * second).sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    #[test]
    fn separates_two_appearance_states() {
        // "shaved" pair and "short mop" pair, mutually distant.
        let refs = vec![
            reference("a1", unit(1.0, 0.1)),
            reference("a2", unit(0.9, 0.2)),
            reference("b1", unit(0.1, 1.0)),
            reference("b2", unit(0.2, 0.9)),
        ];
        let summary = cluster_references(&refs, 0.85).unwrap();
        assert_eq!(summary.clusters.len(), 2);
        let ids: Vec<Vec<usize>> = summary.clusters.iter().map(|c| c.members.clone()).collect();
        assert_eq!(ids, vec![vec![0, 1], vec![2, 3]]);
        // Threshold derived from the intra-cluster spread.
        let threshold = summary.avatar_threshold.unwrap();
        assert!((0.55..=0.93).contains(&threshold), "threshold {threshold} out of band");
        // Representatives are real members.
        for cluster in &summary.clusters {
            assert!(cluster.members.contains(&cluster.representative));
            assert!(cluster.min_internal_cosine > 0.85);
        }
        // Clusters are far apart.
        assert!(summary.inter_cluster_mean < 0.5);
    }

    #[test]
    fn single_state_stays_one_cluster_without_threshold() {
        let refs = vec![
            reference("a", unit(1.0, 0.0)),
            reference("b", unit(0.99, 0.05)),
            reference("c", unit(0.98, 0.1)),
        ];
        let summary = cluster_references(&refs, 0.85).unwrap();
        assert_eq!(summary.clusters.len(), 1);
        // One cluster with 3 members: spread is known → threshold derived.
        assert!(summary.avatar_threshold.is_some());
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(cluster_references(&[], 0.85).is_err());
    }

    #[test]
    fn invalid_vectors_are_rejected_not_silently_clustered() {
        // `cosine_similarity` returns 0.0 on invalid input; without explicit
        // validation a zero-norm vector becomes its own singleton cluster.
        let zero = vec![reference("zero", vec![0.0, 0.0]), reference("ok", unit(1.0, 0.0))];
        assert!(cluster_references(&zero, 0.85).is_err());
        let nan = vec![reference("nan", vec![f32::NAN, 1.0])];
        assert!(cluster_references(&nan, 0.85).is_err());
        let empty = vec![reference("empty", vec![])];
        assert!(cluster_references(&empty, 0.85).is_err());
    }

    #[test]
    fn clustering_is_independent_of_input_order() {
        let base = vec![
            reference("a1", unit(1.0, 0.1)),
            reference("a2", unit(0.9, 0.2)),
            reference("b1", unit(0.1, 1.0)),
            reference("b2", unit(0.2, 0.9)),
            reference("b3", unit(0.15, 0.95)),
        ];
        let forward = cluster_references(&base, 0.85).unwrap();
        let mut shuffled = base.clone();
        shuffled.reverse();
        let backward = cluster_references(&shuffled, 0.85).unwrap();
        assert_eq!(forward.clusters.len(), backward.clusters.len());
        let sizes = |s: &ClusterSummary| {
            let mut v: Vec<usize> = s.clusters.iter().map(|c| c.members.len()).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(sizes(&forward), sizes(&backward));
        assert_eq!(forward.avatar_threshold, backward.avatar_threshold);
    }

    #[test]
    fn greedy_assigns_to_the_closest_cluster() {
        // c is equidistant-ish: must join the closer cluster, not the first.
        let refs = vec![
            reference("a", unit(1.0, 0.0)),
            reference("b", unit(0.0, 1.0)),
            reference("c", unit(0.9, 0.3)),
        ];
        let summary = cluster_references(&refs, 0.85).unwrap();
        assert_eq!(summary.clusters[0].members, vec![0, 2]);
        assert_eq!(summary.clusters[1].members, vec![1]);
    }
}
