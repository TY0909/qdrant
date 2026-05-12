use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use segment::data_types::query_context::PayloadTextSearchContext;
use segment::types::{Filter, ScoredPoint, TokenWeightSet};
use shard::common::stopping_guard::StoppingGuard;
use shard::query::payload_query::{
    PayloadQueryInternal, QueryPayloadRequestInternal, TextQueryInternal,
};
use shard::segment_holder::locked::LockedSegmentHolder;
use tokio_util::task::AbortOnDropHandle;

use super::LocalShard;
use crate::collection_manager::holders::segment_holder::LockedSegment;
use crate::collection_manager::segments_searcher::{PreparedPayloadTextSearch, SegmentsSearcher};
use crate::operations::types::{CollectionError, CollectionResult};

#[derive(Clone)]
struct PreparedTextQuery {
    key: segment::json_path::JsonPath,
    query: TokenWeightSet,
    indexed_points: usize,
}

#[derive(Clone)]
struct PreparedPayloadQueryRequest {
    search: PreparedPayloadTextSearch,
    score_threshold: Option<f32>,
}

impl LocalShard {
    pub(super) async fn query_payload_batch(
        &self,
        batch: Arc<Vec<QueryPayloadRequestInternal>>,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        self.execute_payload_queries(batch.as_ref(), timeout, hw_measurement_acc)
            .await
    }

    pub async fn search_with_payload_query(
        &self,
        payload_query: PayloadQueryInternal,
        filter: Option<Filter>,
        limit: usize,
        score_threshold: Option<f32>,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let request = QueryPayloadRequestInternal {
            payload_query,
            filter,
            score_threshold,
            limit,
        };

        let mut results = self
            .execute_payload_queries(std::slice::from_ref(&request), timeout, hw_measurement_acc)
            .await?;

        Ok(results.pop().unwrap_or_default())
    }

    async fn execute_payload_queries(
        &self,
        requests: &[QueryPayloadRequestInternal],
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let stopping_guard = StoppingGuard::new();
        let start = Instant::now();

        let prepared_requests = self
            .prepare_payload_query_requests(
                requests,
                timeout,
                hw_measurement_acc.clone(),
                &stopping_guard,
            )
            .await?;

        if prepared_requests.is_empty() {
            return Ok(vec![]);
        }

        if prepared_requests
            .iter()
            .all(|request| request.search.ctx.query.tokens.is_empty())
        {
            return Ok(vec![vec![]; prepared_requests.len()]);
        }

        let search_timeout = timeout.saturating_sub(start.elapsed());
        let prepared_searches = Arc::new(
            prepared_requests
                .iter()
                .map(|request| request.search.clone())
                .collect::<Vec<_>>(),
        );

        let future = SegmentsSearcher::search_payload_query_batch(
            self.segments.clone(),
            prepared_searches,
            &self.search_runtime,
            hw_measurement_acc,
            search_timeout,
        );

        let mut results = tokio::time::timeout(search_timeout, future)
            .await
            .map_err(|_| {
                CollectionError::timeout(search_timeout, "search_payload_query_batch")
            })??;

        for (result, prepared_request) in results.iter_mut().zip(&prepared_requests) {
            if let Some(threshold) = prepared_request.score_threshold {
                result.retain(|point| point.score >= threshold);
            }
        }

        Ok(results)
    }

    async fn prepare_payload_query_requests(
        &self,
        requests: &[QueryPayloadRequestInternal],
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
        stopping_guard: &StoppingGuard,
    ) -> CollectionResult<Vec<PreparedPayloadQueryRequest>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let requests_to_prepare = requests.to_vec();
        let segments = self.segments.clone();
        let is_stopped = stopping_guard.get_is_stopped().clone();
        let cpu_utilization = hw_measurement_acc.cpu_utilization();
        let hw_measurement_acc_clone = hw_measurement_acc.clone();

        let task = AbortOnDropHandle::new(self.search_runtime.spawn_blocking(move || {
            let hw_counter = hw_measurement_acc_clone.get_counter_cell();
            cpu_utilization.measure(|| {
                Self::prepare_payload_query_requests_blocking(
                    segments,
                    &requests_to_prepare,
                    timeout,
                    &hw_counter,
                    &is_stopped,
                )
            })
        }));

        let prepared_queries = tokio::time::timeout(timeout, task)
            .await
            .map_err(|_| CollectionError::timeout(timeout, "prepare_payload_query_batch"))???;

        let is_stopped = stopping_guard.get_is_stopped();
        Ok(requests
            .iter()
            .zip(prepared_queries)
            .map(|(request, prepared_query)| PreparedPayloadQueryRequest {
                search: PreparedPayloadTextSearch {
                    ctx: Arc::new(PayloadTextSearchContext {
                        key: prepared_query.key,
                        query: prepared_query.query,
                        filter: request.filter.clone(),
                        top: request.limit,
                        is_stopped: is_stopped.clone(),
                    }),
                    indexed_points: prepared_query.indexed_points,
                },
                score_threshold: request.score_threshold,
            })
            .collect())
    }

    fn prepare_payload_query_requests_blocking(
        segments: LockedSegmentHolder,
        requests: &[QueryPayloadRequestInternal],
        timeout: Duration,
        hw_counter: &HardwareCounterCell,
        is_stopped: &std::sync::atomic::AtomicBool,
    ) -> CollectionResult<Vec<PreparedTextQuery>> {
        let start = Instant::now();

        let segment_readers: Vec<_> = {
            let Some(segments_guard) = segments.try_read_for(timeout) else {
                return Err(CollectionError::timeout(
                    timeout,
                    "prepare_payload_query_batch",
                ));
            };
            segments_guard
                .non_appendable_then_appendable_segments()
                .collect()
        };

        if segment_readers.is_empty() {
            return Ok(requests
                .iter()
                .map(|request| match &request.payload_query {
                    PayloadQueryInternal::Text(text_query) => PreparedTextQuery {
                        key: text_query.key.clone(),
                        query: TokenWeightSet::default(),
                        indexed_points: 0,
                    },
                })
                .collect());
        }

        let mut cache: AHashMap<TextQueryInternal, PreparedTextQuery> = AHashMap::new();
        let mut prepared_requests = Vec::with_capacity(requests.len());

        for request in requests {
            if is_stopped.load(Relaxed) {
                return Err(CollectionError::cancelled(
                    "prepare_payload_query_batch was cancelled",
                ));
            }

            match &request.payload_query {
                PayloadQueryInternal::Text(text_query) => {
                    let prepared = if let Some(prepared) = cache.get(text_query) {
                        prepared.clone()
                    } else {
                        let prepared = Self::compute_text_query_idf_from_segments(
                            &segment_readers,
                            text_query,
                            timeout.saturating_sub(start.elapsed()),
                            hw_counter,
                        )?;
                        cache.insert(text_query.clone(), prepared.clone());
                        prepared
                    };
                    prepared_requests.push(prepared);
                }
            }
        }

        Ok(prepared_requests)
    }

    fn compute_text_query_idf_from_segments(
        segment_readers: &[LockedSegment],
        text_query: &TextQueryInternal,
        timeout: Duration,
        hw_counter: &HardwareCounterCell,
    ) -> CollectionResult<PreparedTextQuery> {
        let start = Instant::now();
        let TextQueryInternal { key, query_str } = text_query;

        let mut tokens = Vec::new();
        for segment in segment_readers {
            let Some(segment_read) = segment
                .get()
                .try_read_for(timeout.saturating_sub(start.elapsed()))
            else {
                return Err(CollectionError::timeout(
                    timeout,
                    "prepare_payload_query_batch",
                ));
            };
            let result = segment_read.text_index_tokenize_query(key, query_str, hw_counter);
            if !result.is_empty() {
                tokens = result;
                break;
            }
        }

        if tokens.is_empty() {
            return Ok(PreparedTextQuery {
                key: key.clone(),
                query: TokenWeightSet::default(),
                indexed_points: 0,
            });
        }

        let mut total_doc_count = 0;
        let mut doc_frequencies = vec![0; tokens.len()];

        for segment in segment_readers {
            let Some(segment_read) = segment
                .get()
                .try_read_for(timeout.saturating_sub(start.elapsed()))
            else {
                return Err(CollectionError::timeout(
                    timeout,
                    "prepare_payload_query_batch",
                ));
            };
            segment_read.fill_text_index_idf(
                key,
                &tokens,
                &mut total_doc_count,
                &mut doc_frequencies,
                hw_counter,
            );
        }

        if total_doc_count == 0 {
            return Ok(PreparedTextQuery {
                key: key.clone(),
                query: TokenWeightSet::default(),
                indexed_points: 0,
            });
        }

        let n = total_doc_count as f32;
        let idfs = doc_frequencies
            .iter()
            .map(|&df| {
                let df = df as f32;
                ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();

        Ok(PreparedTextQuery {
            key: key.clone(),
            query: TokenWeightSet { tokens, idfs },
            indexed_points: total_doc_count,
        })
    }
}
