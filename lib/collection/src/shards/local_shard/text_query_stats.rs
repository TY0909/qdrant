use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use segment::types::Filter;
use shard::query::payload_query::{
    TextQueryStats, TextQueryStatsRequest, validate_text_query_schema,
};
use shard::segment_holder::locked::LockedSegmentHolder;
use tokio_util::task::AbortOnDropHandle;

use super::LocalShard;
use crate::operations::types::{CollectionError, CollectionResult};

impl LocalShard {
    pub async fn text_query_stats(
        &self,
        request: TextQueryStatsRequest,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
        is_stopped: Arc<AtomicBool>,
    ) -> CollectionResult<TextQueryStats> {
        validate_text_query_schema(
            &request.key,
            self.payload_index_schema.read().schema.get(&request.key),
        )?;
        let segments = self.segments.clone();
        let hw_counter = hw_measurement_acc.get_counter_cell();
        let cpu_utilization = hw_measurement_acc.cpu_utilization();
        let task = AbortOnDropHandle::new(self.search_runtime.spawn_blocking(move || {
            cpu_utilization.measure(|| {
                Self::compute_text_query_stats(
                    segments,
                    &request.key,
                    &request.query_str,
                    request.corpus.as_ref(),
                    &hw_counter,
                    is_stopped.as_ref(),
                )
            })
        }));
        tokio::time::timeout(timeout, task)
            .await
            .map_err(|_| CollectionError::timeout(timeout, "compute_text_query_stats"))??
    }

    fn compute_text_query_stats(
        segments: LockedSegmentHolder,
        key: &segment::json_path::JsonPath,
        query_str: &str,
        corpus: Option<&Filter>,
        hw_counter: &HardwareCounterCell,
        is_stopped: &AtomicBool,
    ) -> CollectionResult<TextQueryStats> {
        let segments_guard = segments.read();
        let mut merged = TextQueryStats::default();
        for segment in segments_guard.non_appendable_then_appendable_segments() {
            if is_stopped.load(Ordering::Relaxed) {
                return Err(CollectionError::cancelled(
                    "Text query statistics collection was cancelled",
                ));
            }
            let stats = segment
                .get()
                .read()
                .payload_text_stats(key, query_str, corpus, is_stopped, hw_counter)?;
            merged.merge(stats.into())?;
        }
        Ok(merged)
    }
}
