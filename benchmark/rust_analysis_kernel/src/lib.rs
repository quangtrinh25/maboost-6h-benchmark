use std::cmp::min;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::thread;


pub const METRIC_AUROC: u32 = 1;
pub const METRIC_AUPRC: u32 = 2;
pub const METRIC_BRIER: u32 = 3;
pub const METRIC_LOS_MAE: u32 = 4;
pub const METRIC_LOS_RMSE: u32 = 5;


#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    InvalidMetric,
    LengthMismatch,
    NonFinite,
    InvalidLabel,
    InvalidClusterCode,
    InvalidDraw,
    MissingClass { replicate: usize },
    NullPointer,
    Panic,
}


impl KernelError {
    fn code(self) -> i32 {
        match self {
            Self::InvalidMetric => 1,
            Self::LengthMismatch => 2,
            Self::NonFinite => 3,
            Self::InvalidLabel => 4,
            Self::InvalidClusterCode => 5,
            Self::InvalidDraw => 6,
            Self::MissingClass { .. } => 7,
            Self::NullPointer => 8,
            Self::Panic => 9,
        }
    }
}


#[derive(Clone, Copy)]
struct RankRow {
    score: f64,
    positive: bool,
    cluster: usize,
}


enum PreparedMetric {
    Loss {
        cluster_rows: Vec<u64>,
        error_a: Vec<f64>,
        error_b: Vec<f64>,
        square_root: bool,
    },
    Rank {
        order_a: Vec<RankRow>,
        order_b: Vec<RankRow>,
        positives: Vec<u64>,
        negatives: Vec<u64>,
        average_precision: bool,
    },
}


fn validate_inputs(
    metric: u32,
    labels: &[f64],
    predictions_a: &[f64],
    predictions_b: &[f64],
    cluster_codes: &[u32],
    cluster_count: usize,
    draws: &[u32],
    replicates: usize,
) -> Result<(), KernelError> {
    if !matches!(
        metric,
        METRIC_AUROC | METRIC_AUPRC | METRIC_BRIER | METRIC_LOS_MAE | METRIC_LOS_RMSE
    ) {
        return Err(KernelError::InvalidMetric);
    }
    if labels.is_empty()
        || cluster_count == 0
        || replicates == 0
        || predictions_a.len() != labels.len()
        || predictions_b.len() != labels.len()
        || cluster_codes.len() != labels.len()
        || draws.len() != replicates.saturating_mul(cluster_count)
    {
        return Err(KernelError::LengthMismatch);
    }
    if labels.iter().chain(predictions_a).chain(predictions_b).any(|x| !x.is_finite()) {
        return Err(KernelError::NonFinite);
    }
    if matches!(metric, METRIC_AUROC | METRIC_AUPRC | METRIC_BRIER)
        && labels.iter().any(|&x| x != 0.0 && x != 1.0)
    {
        return Err(KernelError::InvalidLabel);
    }
    let mut seen = vec![false; cluster_count];
    for &code in cluster_codes {
        let index = code as usize;
        if index >= cluster_count {
            return Err(KernelError::InvalidClusterCode);
        }
        seen[index] = true;
    }
    if seen.iter().any(|present| !present) {
        return Err(KernelError::InvalidClusterCode);
    }
    if draws.iter().any(|&code| code as usize >= cluster_count) {
        return Err(KernelError::InvalidDraw);
    }
    Ok(())
}


fn prepare_metric(
    metric: u32,
    labels: &[f64],
    predictions_a: &[f64],
    predictions_b: &[f64],
    cluster_codes: &[u32],
    cluster_count: usize,
) -> PreparedMetric {
    if matches!(metric, METRIC_BRIER | METRIC_LOS_MAE | METRIC_LOS_RMSE) {
        let mut cluster_rows = vec![0_u64; cluster_count];
        let mut error_a = vec![0.0_f64; cluster_count];
        let mut error_b = vec![0.0_f64; cluster_count];
        for row in 0..labels.len() {
            let cluster = cluster_codes[row] as usize;
            cluster_rows[cluster] += 1;
            let delta_a = labels[row] - predictions_a[row];
            let delta_b = labels[row] - predictions_b[row];
            if metric == METRIC_LOS_MAE {
                error_a[cluster] += delta_a.abs();
                error_b[cluster] += delta_b.abs();
            } else {
                error_a[cluster] += delta_a * delta_a;
                error_b[cluster] += delta_b * delta_b;
            }
        }
        return PreparedMetric::Loss {
            cluster_rows,
            error_a,
            error_b,
            square_root: metric == METRIC_LOS_RMSE,
        };
    }

    let mut positives = vec![0_u64; cluster_count];
    let mut negatives = vec![0_u64; cluster_count];
    let mut order_a = Vec::with_capacity(labels.len());
    let mut order_b = Vec::with_capacity(labels.len());
    for row in 0..labels.len() {
        let positive = labels[row] == 1.0;
        let cluster = cluster_codes[row] as usize;
        if positive {
            positives[cluster] += 1;
        } else {
            negatives[cluster] += 1;
        }
        order_a.push(RankRow { score: predictions_a[row], positive, cluster });
        order_b.push(RankRow { score: predictions_b[row], positive, cluster });
    }
    order_a.sort_unstable_by(|left, right| left.score.total_cmp(&right.score));
    order_b.sort_unstable_by(|left, right| left.score.total_cmp(&right.score));
    PreparedMetric::Rank {
        order_a,
        order_b,
        positives,
        negatives,
        average_precision: metric == METRIC_AUPRC,
    }
}


fn cluster_multiplicities(draw: &[u32], counts: &mut [u32]) {
    counts.fill(0);
    for &cluster in draw {
        counts[cluster as usize] += 1;
    }
}


fn loss_value(
    cluster_rows: &[u64],
    errors: &[f64],
    multiplicities: &[u32],
    square_root: bool,
) -> f64 {
    let mut sampled_rows = 0_u64;
    let mut error_sum = 0.0_f64;
    for cluster in 0..multiplicities.len() {
        let weight = multiplicities[cluster] as u64;
        sampled_rows += weight * cluster_rows[cluster];
        error_sum += (weight as f64) * errors[cluster];
    }
    let mean = error_sum / sampled_rows as f64;
    if square_root { mean.sqrt() } else { mean }
}


fn class_totals(
    positives: &[u64],
    negatives: &[u64],
    multiplicities: &[u32],
) -> (u64, u64) {
    let mut positive = 0_u64;
    let mut negative = 0_u64;
    for cluster in 0..multiplicities.len() {
        let weight = multiplicities[cluster] as u64;
        positive += weight * positives[cluster];
        negative += weight * negatives[cluster];
    }
    (positive, negative)
}


fn rank_value(
    rows: &[RankRow],
    multiplicities: &[u32],
    total_positive: u64,
    total_negative: u64,
    average_precision: bool,
) -> f64 {
    if average_precision {
        let mut cursor = rows.len();
        let mut cumulative_positive = 0_u64;
        let mut cumulative_total = 0_u64;
        let mut value = 0.0_f64;
        while cursor > 0 {
            let end = cursor;
            cursor -= 1;
            let score = rows[cursor].score;
            while cursor > 0 && rows[cursor - 1].score == score {
                cursor -= 1;
            }
            let mut positive_weight = 0_u64;
            let mut negative_weight = 0_u64;
            for row in &rows[cursor..end] {
                let weight = multiplicities[row.cluster] as u64;
                if row.positive {
                    positive_weight += weight;
                } else {
                    negative_weight += weight;
                }
            }
            cumulative_positive += positive_weight;
            cumulative_total += positive_weight + negative_weight;
            if positive_weight > 0 {
                value += (positive_weight as f64 / total_positive as f64)
                    * (cumulative_positive as f64 / cumulative_total as f64);
            }
        }
        return value;
    }

    let mut cursor = 0_usize;
    let mut negative_below = 0_u64;
    let mut numerator = 0.0_f64;
    while cursor < rows.len() {
        let start = cursor;
        let score = rows[cursor].score;
        cursor += 1;
        while cursor < rows.len() && rows[cursor].score == score {
            cursor += 1;
        }
        let mut positive_weight = 0_u64;
        let mut negative_weight = 0_u64;
        for row in &rows[start..cursor] {
            let weight = multiplicities[row.cluster] as u64;
            if row.positive {
                positive_weight += weight;
            } else {
                negative_weight += weight;
            }
        }
        numerator += positive_weight as f64
            * (negative_below as f64 + 0.5 * negative_weight as f64);
        negative_below += negative_weight;
    }
    numerator / (total_positive as f64 * total_negative as f64)
}


fn evaluate_one(
    prepared: &PreparedMetric,
    multiplicities: &[u32],
    replicate: usize,
) -> Result<f64, KernelError> {
    match prepared {
        PreparedMetric::Loss { cluster_rows, error_a, error_b, square_root } => Ok(
            loss_value(cluster_rows, error_a, multiplicities, *square_root)
                - loss_value(cluster_rows, error_b, multiplicities, *square_root),
        ),
        PreparedMetric::Rank {
            order_a,
            order_b,
            positives,
            negatives,
            average_precision,
        } => {
            let (total_positive, total_negative) =
                class_totals(positives, negatives, multiplicities);
            // sklearn's average_precision_score is defined for a sampled set
            // containing no positives (both models receive AP=0, hence a zero
            // paired difference) and for an all-positive set (both receive
            // AP=1).  AUROC alone requires both classes.
            if *average_precision && total_positive == 0 {
                return Ok(0.0);
            }
            if !*average_precision && (total_positive == 0 || total_negative == 0) {
                return Err(KernelError::MissingClass { replicate });
            }
            Ok(
                rank_value(
                    order_a,
                    multiplicities,
                    total_positive,
                    total_negative,
                    *average_precision,
                ) - rank_value(
                    order_b,
                    multiplicities,
                    total_positive,
                    total_negative,
                    *average_precision,
                ),
            )
        }
    }
}


fn evaluate_range(
    prepared: &PreparedMetric,
    draws: &[u32],
    cluster_count: usize,
    start: usize,
    output: &mut [f64],
) -> Result<(), KernelError> {
    let mut multiplicities = vec![0_u32; cluster_count];
    for (local, slot) in output.iter_mut().enumerate() {
        let replicate = start + local;
        let offset = replicate * cluster_count;
        cluster_multiplicities(
            &draws[offset..offset + cluster_count],
            &mut multiplicities,
        );
        *slot = evaluate_one(prepared, &multiplicities, replicate)?;
    }
    Ok(())
}


pub fn evaluate_draws_safe(
    metric: u32,
    labels: &[f64],
    predictions_a: &[f64],
    predictions_b: &[f64],
    cluster_codes: &[u32],
    cluster_count: usize,
    draws: &[u32],
    replicates: usize,
    threads: usize,
) -> Result<Vec<f64>, KernelError> {
    validate_inputs(
        metric,
        labels,
        predictions_a,
        predictions_b,
        cluster_codes,
        cluster_count,
        draws,
        replicates,
    )?;
    let prepared = prepare_metric(
        metric,
        labels,
        predictions_a,
        predictions_b,
        cluster_codes,
        cluster_count,
    );
    let worker_count = min(replicates, threads.max(1));
    let chunk_size = (replicates + worker_count - 1) / worker_count;
    let mut output = vec![0.0_f64; replicates];
    let result = thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut remaining = output.as_mut_slice();
        let mut start = 0_usize;
        while start < replicates {
            let stop = min(replicates, start + chunk_size);
            let (chunk, rest) = remaining.split_at_mut(stop - start);
            remaining = rest;
            let prepared_ref = &prepared;
            handles.push(scope.spawn(move || {
                evaluate_range(prepared_ref, draws, cluster_count, start, chunk)
            }));
            start = stop;
        }
        let mut first_error: Option<KernelError> = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let replace = match (first_error, error) {
                        (None, _) => true,
                        (
                            Some(KernelError::MissingClass { replicate: current }),
                            KernelError::MissingClass { replicate: candidate },
                        ) => candidate < current,
                        _ => false,
                    };
                    if replace {
                        first_error = Some(error);
                    }
                }
                Err(_) => first_error = Some(KernelError::Panic),
            }
        }
        first_error.map_or(Ok(()), Err)
    });
    result?;
    Ok(output)
}


#[no_mangle]
pub extern "C" fn maboost_analysis_abi_version_v1() -> u32 {
    1
}


#[no_mangle]
pub unsafe extern "C" fn maboost_bootstrap_evaluate_v1(
    metric: u32,
    labels: *const f64,
    predictions_a: *const f64,
    predictions_b: *const f64,
    cluster_codes: *const u32,
    row_count: usize,
    cluster_count: usize,
    draws: *const u32,
    replicates: usize,
    threads: usize,
    output: *mut f64,
    error_replicate: *mut usize,
) -> i32 {
    let guarded = catch_unwind(AssertUnwindSafe(|| {
        if labels.is_null()
            || predictions_a.is_null()
            || predictions_b.is_null()
            || cluster_codes.is_null()
            || draws.is_null()
            || output.is_null()
            || error_replicate.is_null()
        {
            return Err(KernelError::NullPointer);
        }
        let labels = slice::from_raw_parts(labels, row_count);
        let predictions_a = slice::from_raw_parts(predictions_a, row_count);
        let predictions_b = slice::from_raw_parts(predictions_b, row_count);
        let cluster_codes = slice::from_raw_parts(cluster_codes, row_count);
        let draw_count = match replicates.checked_mul(cluster_count) {
            Some(value) => value,
            None => return Err(KernelError::LengthMismatch),
        };
        let draws = slice::from_raw_parts(draws, draw_count);
        match evaluate_draws_safe(
            metric,
            labels,
            predictions_a,
            predictions_b,
            cluster_codes,
            cluster_count,
            draws,
            replicates,
            threads,
        ) {
            Ok(values) => {
                slice::from_raw_parts_mut(output, replicates).copy_from_slice(&values);
                *error_replicate = usize::MAX;
                Ok(())
            }
            Err(error) => {
                if let KernelError::MissingClass { replicate } = error {
                    *error_replicate = replicate;
                }
                Err(error)
            }
        }
    }));
    match guarded {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => error.code(),
        Err(_) => KernelError::Panic.code(),
    }
}
