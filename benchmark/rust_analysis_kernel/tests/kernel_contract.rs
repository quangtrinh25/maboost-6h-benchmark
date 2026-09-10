use maboost_analysis_kernel::{
    evaluate_draws_safe, KernelError, METRIC_AUPRC, METRIC_AUROC, METRIC_BRIER,
    METRIC_LOS_MAE, METRIC_LOS_RMSE,
};


fn fixture() -> ([f64; 4], [f64; 4], [f64; 4], [u32; 4]) {
    (
        [0.0, 1.0, 0.0, 1.0],
        [0.1, 0.8, 0.3, 0.6],
        [0.2, 0.7, 0.4, 0.5],
        [0, 0, 1, 2],
    )
}


#[test]
fn brier_difference_matches_explicit_cluster_repetition() {
    let (labels, a, b, groups) = fixture();
    let got = evaluate_draws_safe(
        METRIC_BRIER,
        &labels,
        &a,
        &b,
        &groups,
        3,
        &[0, 2, 2],
        1,
        1,
    )
    .unwrap();
    assert!((got[0] - (-0.065_f64)).abs() <= 1e-15);
}


#[test]
fn loss_metrics_preserve_square_root_after_mean() {
    let labels = [1.0, 3.0];
    let a = [0.0, 5.0];
    let b = [2.0, 2.0];
    let groups = [0, 1];
    let draws = [0, 1];
    let mae = evaluate_draws_safe(
        METRIC_LOS_MAE, &labels, &a, &b, &groups, 2, &draws, 1, 1,
    )
    .unwrap()[0];
    let rmse = evaluate_draws_safe(
        METRIC_LOS_RMSE, &labels, &a, &b, &groups, 2, &draws, 1, 1,
    )
    .unwrap()[0];
    assert!((mae - 0.5).abs() <= 1e-15);
    assert!((rmse - (2.5_f64.sqrt() - 1.0)).abs() <= 1e-15);
}


#[test]
fn ranking_ties_match_hand_checked_weighted_definitions() {
    let labels = [1.0, 0.0, 1.0, 0.0];
    let a = [0.8, 0.8, 0.4, 0.2];
    let b = [0.9, 0.7, 0.7, 0.1];
    let groups = [0, 1, 2, 3];
    let draws = [0, 1, 2, 3];
    for metric in [METRIC_AUROC, METRIC_AUPRC] {
        let got = evaluate_draws_safe(
            metric, &labels, &a, &b, &groups, 4, &draws, 1, 1,
        )
        .unwrap()[0];
        assert!((got - (-0.25)).abs() <= 1e-15, "metric={metric}, got={got}");
    }
}


#[test]
fn auprc_single_class_draw_matches_sklearn_difference_semantics() {
    let labels = [0.0, 1.0];
    let a = [0.2, 0.8];
    let b = [0.4, 0.6];
    let groups = [0, 1];
    let no_positive = evaluate_draws_safe(
        METRIC_AUPRC, &labels, &a, &b, &groups, 2, &[0, 0], 1, 1,
    )
    .unwrap();
    let no_negative = evaluate_draws_safe(
        METRIC_AUPRC, &labels, &a, &b, &groups, 2, &[1, 1], 1, 1,
    )
    .unwrap();
    assert_eq!(no_positive, vec![0.0]);
    assert_eq!(no_negative, vec![0.0]);
}


#[test]
fn zero_weight_clusters_are_excluded() {
    let labels = [0.0, 1.0, 0.0, 1.0];
    let a = [0.1, 0.9, 0.8, 0.2];
    let b = [0.4, 0.6, 0.3, 0.7];
    let groups = [0, 1, 2, 3];
    let got = evaluate_draws_safe(
        METRIC_BRIER, &labels, &a, &b, &groups, 4, &[0, 1, 1, 0], 1, 1,
    )
    .unwrap()[0];
    assert!((got - (-0.15)).abs() <= 1e-15);
}


#[test]
fn missing_class_reports_the_replicate_index() {
    let labels = [0.0, 1.0];
    let a = [0.2, 0.8];
    let b = [0.3, 0.7];
    let groups = [0, 1];
    let error = evaluate_draws_safe(
        METRIC_AUROC, &labels, &a, &b, &groups, 2, &[0, 0, 0, 1], 2, 2,
    )
    .unwrap_err();
    assert_eq!(error, KernelError::MissingClass { replicate: 0 });
}


#[test]
fn invalid_group_index_fails_before_evaluation() {
    let error = evaluate_draws_safe(
        METRIC_BRIER,
        &[0.0, 1.0],
        &[0.2, 0.8],
        &[0.3, 0.7],
        &[0, 2],
        2,
        &[0, 1],
        1,
        1,
    )
    .unwrap_err();
    assert_eq!(error, KernelError::InvalidClusterCode);
}


#[test]
fn thread_count_does_not_change_ordered_results() {
    let (labels, a, b, groups) = fixture();
    let draws = [0, 1, 2, 2, 1, 0, 1, 1, 2, 0, 2, 1];
    let one = evaluate_draws_safe(
        METRIC_BRIER, &labels, &a, &b, &groups, 3, &draws, 4, 1,
    )
    .unwrap();
    let four = evaluate_draws_safe(
        METRIC_BRIER, &labels, &a, &b, &groups, 3, &draws, 4, 4,
    )
    .unwrap();
    assert_eq!(one, four);
}
