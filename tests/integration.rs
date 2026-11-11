//! Integration tests over the public library surface. No network, no models,
//! no ffmpeg: every case is deterministic and runs on any machine.

mod cache_key {
    #[test]
    fn same_inputs_same_key() {
        let params = serde_json::json!({"w": 1080, "h": 1920});
        let a = avatar_media_mcp::infra::cache::Cache::key("fal", "m1", &params, &["h1".into()]);
        let b = avatar_media_mcp::infra::cache::Cache::key("fal", "m1", &params, &["h1".into()]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_model_different_key() {
        let params = serde_json::json!({});
        let a = avatar_media_mcp::infra::cache::Cache::key("fal", "m1", &params, &[]);
        let b = avatar_media_mcp::infra::cache::Cache::key("fal", "m2", &params, &[]);
        assert_ne!(a, b);
    }
}

mod identity_scoring {
    use avatar_media_mcp::identity::{embeddings, scoring};

    #[test]
    fn cosine_of_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((embeddings::cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stage1_boundaries() {
        let t = scoring::IdentityThresholds::default();
        assert!(matches!(scoring::stage1(0.9, &t), scoring::Stage1Verdict::Pass(_)));
        assert!(matches!(scoring::stage1(0.65, &t), scoring::Stage1Verdict::Borderline(_)));
        assert!(matches!(scoring::stage1(0.4, &t), scoring::Stage1Verdict::Fail(_)));
    }

    #[test]
    fn each_to_own_canon_detects_swap() {
        let canon_a = vec![1.0, 0.0];
        let canon_b = vec![0.0, 1.0];
        // Face close to A, far from B → ok.
        let ok = vec![vec![0.95, 0.05]];
        assert!(scoring::each_to_own_canon(&ok, &[canon_a.clone(), canon_b.clone()]));
        // Face equidistant → suspicious.
        let bad = vec![vec![0.5, 0.5]];
        assert!(!scoring::each_to_own_canon(&bad, &[canon_a, canon_b]));
    }
}
