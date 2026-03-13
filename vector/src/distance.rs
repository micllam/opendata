//! Distance computation functions for vector similarity.
//!
//! This module provides distance and similarity functions used for scoring
//! candidates during similarity search.

use crate::serde::collection_meta::DistanceMetric;
use std::cmp::Ordering;

/// L2-normalize a vector in place.
///
/// Projects the vector onto the unit hypersphere. Zero vectors are left
/// unchanged to avoid division by zero.
pub(crate) fn l2_normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Compute distance/similarity between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// * `metric` - Distance metric to use
///
/// # Returns
/// Distance/similarity score. Higher scores indicate more similar vectors,
/// except for L2 distance where lower scores indicate more similar vectors.
///
/// # Panics
/// Panics if the vectors have different lengths.
pub(crate) fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> VectorDistance {
    assert_eq!(
        a.len(),
        b.len(),
        "Cannot compute distance between vectors of different lengths"
    );

    let v = match metric {
        DistanceMetric::L2 | DistanceMetric::Cosine => l2_distance(a, b),
        DistanceMetric::DotProduct => dot_product(a, b),
    };
    VectorDistance { score: v, metric }
}

/// Compute a uniform distance where lower = closer, suitable for comparing
/// across distance metrics in the boundary replication formula.
pub(crate) fn raw_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 | DistanceMetric::Cosine => compute_distance(a, b, metric).score(),
        DistanceMetric::DotProduct => -compute_distance(a, b, metric).score(),
    }
}

/// Compute L2 (Euclidean) distance between two vectors.
///
/// Formula: sqrt(sum((a[i] - b[i])²))
///
/// Lower scores indicate more similar vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product between two vectors.
///
/// Formula: sum(a[i] * b[i])
///
/// Higher scores indicate more similar vectors (for normalized vectors).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// A distance/similarity score between two vectors, with metric-aware ordering.
///
/// Ordering is defined so that `a < b` means `a` is **more similar** than `b`.
/// This abstracts over the direction of each metric:
/// - L2: lower raw value = more similar (natural order)
/// - DotProduct: higher raw value = more similar (reversed order)
#[derive(Copy, Clone, Debug)]
pub(crate) struct VectorDistance {
    score: f32,
    metric: DistanceMetric,
}

impl VectorDistance {
    /// Returns the raw distance/similarity value.
    pub(crate) fn score(&self) -> f32 {
        self.score
    }
}

impl PartialEq for VectorDistance {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for VectorDistance {}

impl PartialOrd for VectorDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VectorDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.metric {
            // L2/Cosine: lower value = more similar, so natural order
            DistanceMetric::L2 | DistanceMetric::Cosine => self.score.total_cmp(&other.score),
            // DotProduct: higher value = more similar, so reverse order
            DistanceMetric::DotProduct => other.score.total_cmp(&self.score),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // Parameterized tests for distance functions
    #[rstest]
    #[case(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], 5.196, "different vectors")]
    #[case(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0.0, "identical vectors")]
    fn should_compute_l2_distance(
        #[case] a: Vec<f32>,
        #[case] b: Vec<f32>,
        #[case] expected: f32,
        #[case] _desc: &str,
    ) {
        // when
        let distance = l2_distance(&a, &b);

        // then
        assert!((distance - expected).abs() < 0.01);
    }

    #[rstest]
    #[case(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], 32.0, "normal vectors")]
    #[case(vec![1.0, 0.0], vec![0.0, 1.0], 0.0, "orthogonal vectors")]
    fn should_compute_dot_product(
        #[case] a: Vec<f32>,
        #[case] b: Vec<f32>,
        #[case] expected: f32,
        #[case] _desc: &str,
    ) {
        // when
        let dot = dot_product(&a, &b);

        // then
        assert_eq!(dot, expected);
    }

    #[rstest]
    #[case(DistanceMetric::L2, "L2")]
    #[case(DistanceMetric::Cosine, "Cosine")]
    #[case(DistanceMetric::DotProduct, "DotProduct")]
    fn should_use_correct_metric(#[case] metric: DistanceMetric, #[case] _desc: &str) {
        // given
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        // when
        let result = compute_distance(&a, &b, metric);

        // then - verify result matches direct function call
        let expected = match metric {
            DistanceMetric::L2 | DistanceMetric::Cosine => l2_distance(&a, &b),
            DistanceMetric::DotProduct => dot_product(&a, &b),
        };
        assert_eq!(result.score(), expected);
    }

    #[test]
    #[should_panic(expected = "Cannot compute distance between vectors of different lengths")]
    fn should_panic_on_mismatched_dimensions() {
        // given - vectors with different lengths
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        // when - attempt to compute distance
        compute_distance(&a, &b, DistanceMetric::L2);

        // then - should panic
    }

    // ---- VectorDistance ordering ----

    #[test]
    fn should_order_l2_by_lower_is_more_similar() {
        // given
        let closer = compute_distance(&[0.0, 0.0], &[1.0, 0.0], DistanceMetric::L2);
        let farther = compute_distance(&[0.0, 0.0], &[3.0, 0.0], DistanceMetric::L2);

        // then - closer (lower L2) should be "less than" farther
        assert!(closer < farther);
        assert!(farther > closer);
        assert_ne!(closer, farther);
    }

    #[test]
    fn should_order_dot_product_by_higher_is_more_similar() {
        // given
        let more_similar = compute_distance(&[3.0, 0.0], &[2.0, 0.0], DistanceMetric::DotProduct);
        let less_similar = compute_distance(&[3.0, 0.0], &[0.0, 2.0], DistanceMetric::DotProduct);

        // then - higher dot product should be "less than" (more similar)
        assert!(more_similar < less_similar);
    }

    #[test]
    fn should_consider_equal_distances_equal() {
        // given
        let d1 = compute_distance(&[1.0, 0.0], &[0.0, 1.0], DistanceMetric::L2);
        let d2 = compute_distance(&[0.0, 1.0], &[1.0, 0.0], DistanceMetric::L2);

        // then
        assert_eq!(d1, d2);
    }

    #[test]
    fn should_sort_vector_distances_most_similar_first() {
        // given - three L2 distances
        let d_far = compute_distance(&[0.0], &[10.0], DistanceMetric::L2);
        let d_mid = compute_distance(&[0.0], &[5.0], DistanceMetric::L2);
        let d_near = compute_distance(&[0.0], &[1.0], DistanceMetric::L2);
        let mut distances = [d_far, d_mid, d_near];

        // when
        distances.sort();

        // then - most similar (nearest) first
        assert_eq!(distances[0].score(), d_near.score());
        assert_eq!(distances[1].score(), d_mid.score());
        assert_eq!(distances[2].score(), d_far.score());
    }

    #[test]
    fn should_normalize_vector_to_unit_length() {
        // given
        let mut v = vec![3.0, 4.0];

        // when
        l2_normalize_vector(&mut v);

        // then
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn should_leave_zero_vector_unchanged() {
        // given
        let mut v = vec![0.0, 0.0, 0.0];

        // when
        l2_normalize_vector(&mut v);

        // then
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn should_order_cosine_by_lower_is_more_similar() {
        // given - cosine uses L2 on normalized vectors
        let a = [1.0, 0.0]; // 0 degrees
        let b_close = [
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ]; // 45 degrees
        let b_far = [0.0, 1.0]; // 90 degrees

        // when
        let closer = compute_distance(&a, &b_close, DistanceMetric::Cosine);
        let farther = compute_distance(&a, &b_far, DistanceMetric::Cosine);

        // then
        assert!(closer < farther);
    }

    #[rstest]
    #[case(vec![1.0, 0.0], vec![1.0, 0.0], 0.0, "identical unit vectors")]
    #[case(vec![1.0, 0.0], vec![0.0, 1.0], std::f32::consts::SQRT_2, "orthogonal unit vectors")]
    #[case(vec![1.0, 0.0], vec![-1.0, 0.0], 2.0, "opposite unit vectors")]
    fn should_compute_cosine_distance(
        #[case] a: Vec<f32>,
        #[case] b: Vec<f32>,
        #[case] expected: f32,
        #[case] _desc: &str,
    ) {
        // when - cosine distance is L2 distance on pre-normalized vectors
        let distance = compute_distance(&a, &b, DistanceMetric::Cosine);

        // then
        assert!((distance.score() - expected).abs() < 1e-6);
    }

    #[test]
    fn should_consider_equal_cosine_distances_equal() {
        // given - two pairs of unit vectors at the same angle
        let a = [1.0, 0.0];
        let d1 = compute_distance(&a, &[0.0, 1.0], DistanceMetric::Cosine);
        let d2 = compute_distance(&a, &[0.0, 1.0], DistanceMetric::Cosine);

        // then
        assert_eq!(d1, d2);
    }

    #[test]
    fn should_sort_cosine_distances_most_similar_first() {
        // given - unit vectors at increasing angles from [1, 0]
        let origin = [1.0, 0.0];
        let d_far = compute_distance(&origin, &[-1.0, 0.0], DistanceMetric::Cosine); // 180 degrees
        let d_mid = compute_distance(&origin, &[0.0, 1.0], DistanceMetric::Cosine); // 90 degrees
        let d_near = compute_distance(
            &origin,
            &[
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            DistanceMetric::Cosine,
        ); // 45 degrees
        let mut distances = [d_far, d_mid, d_near];

        // when
        distances.sort();

        // then - most similar (smallest angle) first
        assert_eq!(distances[0].score(), d_near.score());
        assert_eq!(distances[1].score(), d_mid.score());
        assert_eq!(distances[2].score(), d_far.score());
    }

    #[test]
    fn should_compute_raw_cosine_distance_as_lower_is_closer() {
        // given - unit vectors at different angles
        let a = [1.0, 0.0];
        let b_close = [
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ]; // 45 degrees
        let b_far = [0.0, 1.0]; // 90 degrees

        // when
        let close_dist = raw_distance(&a, &b_close, DistanceMetric::Cosine);
        let far_dist = raw_distance(&a, &b_far, DistanceMetric::Cosine);

        // then - lower raw distance means closer
        assert!(close_dist < far_dist);
    }

    #[test]
    fn should_normalize_if_needed_for_cosine_metric() {
        // given
        let mut v = vec![3.0, 4.0];

        // when
        DistanceMetric::Cosine.normalize_if_needed(&mut v);

        // then - vector should be unit length
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn should_not_normalize_for_non_cosine_metrics() {
        // given
        let mut v_l2 = vec![3.0, 4.0];
        let mut v_dot = vec![3.0, 4.0];

        // when
        DistanceMetric::L2.normalize_if_needed(&mut v_l2);
        DistanceMetric::DotProduct.normalize_if_needed(&mut v_dot);

        // then - vectors should be unchanged
        assert_eq!(v_l2, vec![3.0, 4.0]);
        assert_eq!(v_dot, vec![3.0, 4.0]);
    }
}
