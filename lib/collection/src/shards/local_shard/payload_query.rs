use std::sync::Arc;
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use futures::future::try_join_all;
use segment::data_types::query_context::PayloadTextSearchContext;
use segment::types::{Filter, ScoredPoint, TokenWeightSet};
use shard::common::stopping_guard::StoppingGuard;
use shard::query::payload_query::{
    PayloadQueryInternal, QueryPayloadRequestInternal, TextQueryInternal,
};
use shard::segment_holder::locked::LockedSegmentHolder;

use super::LocalShard;
use crate::collection_manager::segments_searcher::SegmentsSearcher;
use crate::operations::types::{CollectionError, CollectionResult};

impl LocalShard {
    /// Basic parallel batching for payload queries, used by the universal query API.
    pub(super) async fn query_payload_batch(
        &self,
        batch: Arc<Vec<QueryPayloadRequestInternal>>,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let payload_queries = batch
            .iter()
            .map(|request| self.query_payload(request, timeout, hw_measurement_acc.clone()));

        let all_payload_results = try_join_all(payload_queries);
        tokio::time::timeout(timeout, all_payload_results)
            .await
            .map_err(|_| CollectionError::timeout(timeout, "Query payload"))?
    }

    async fn query_payload(
        &self,
        request: &QueryPayloadRequestInternal,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let QueryPayloadRequestInternal {
            payload_query,
            filter,
            score_threshold,
            limit,
        } = request;

        self.search_with_payload_query(
            payload_query.clone(),
            filter.clone(),
            *limit,
            *score_threshold,
            timeout,
            hw_measurement_acc,
        )
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
        let stopping_guard = StoppingGuard::new();

        match payload_query {
            PayloadQueryInternal::Text(text_query) => {
                self.search_with_text_query(
                    text_query,
                    filter,
                    limit,
                    score_threshold,
                    timeout,
                    hw_measurement_acc,
                    &stopping_guard,
                )
                .await
            }
        }
    }

    async fn search_with_text_query(
        &self,
        text_query: TextQueryInternal,
        filter: Option<Filter>,
        limit: usize,
        score_threshold: Option<f32>,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
        stopping_guard: &StoppingGuard,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let TextQueryInternal { key, query_str } = text_query;

        // Step 1: Tokenize the query and compute global IDF across all segments.
        // This follows the same pattern as sparse vector IDF computation.
        let token_weight_set = Self::compute_text_query_idf(
            self.segments.clone(),
            &key,
            &query_str,
            &hw_measurement_acc.get_counter_cell(),
        )?;

        if token_weight_set.tokens.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Search the text index across all segments with the computed IDF
        let ctx = PayloadTextSearchContext {
            key,
            query: token_weight_set,
            filter,
            top: limit,
            is_stopped: stopping_guard.get_is_stopped(),
        };

        let arc_ctx = Arc::new(ctx);

        let future = SegmentsSearcher::search_payload_query(
            self.segments.clone(),
            arc_ctx,
            &self.search_runtime,
            hw_measurement_acc,
            timeout,
        );

        let mut res = tokio::time::timeout(timeout, future)
            .await
            .map_err(|_elapsed| CollectionError::timeout(timeout, "search_with_payload_query"))??;

        // Apply score threshold (BM25 scores are always positive, higher is better)
        if let Some(threshold) = score_threshold {
            res.retain(|point| point.score >= threshold);
        }

        Ok(res)
    }

    /// Compute IDF weights for query tokens across all segments.
    ///
    /// This tokenizes the query string using the text index's tokenizer,
    /// then gathers document frequency for each token across all segments,
    /// and computes the global IDF weight for each token.
    ///
    /// This mirrors the sparse vector IDF computation pattern.
    fn compute_text_query_idf(
        segments: LockedSegmentHolder,
        key: &segment::json_path::JsonPath,
        query_str: &str,
        hw_counter: &HardwareCounterCell,
    ) -> CollectionResult<TokenWeightSet> {
        let segments_guard = segments.read();

        let segment_readers: Vec<_> = segments_guard
            .non_appendable_then_appendable_segments()
            .collect();

        if segment_readers.is_empty() {
            return Ok(TokenWeightSet {
                tokens: vec![],
                idfs: vec![],
            });
        }

        // Tokenize the query using the first segment's tokenizer.
        // All segments should share the same index configuration.
        // `text_index_tokenize_query` already returns sorted, deduplicated tokens,
        // matching the sparse vector behavior where each dimension appears once.
        let mut tokens: Vec<String> = Vec::new();
        for segment in &segment_readers {
            let segment_read = segment.get().read();
            let result = segment_read.text_index_tokenize_query(key, query_str, hw_counter);
            if !result.is_empty() {
                tokens = result;
                break;
            }
        }

        if tokens.is_empty() {
            return Ok(TokenWeightSet {
                tokens: vec![],
                idfs: vec![],
            });
        }

        // Gather document frequency and total document count across all segments
        let mut total_doc_count: usize = 0;
        let mut doc_frequencies: Vec<usize> = vec![0; tokens.len()];

        for segment in &segment_readers {
            let segment_read = segment.get().read();
            segment_read.fill_text_index_idf(
                key,
                &tokens,
                &mut total_doc_count,
                &mut doc_frequencies,
                hw_counter,
            );
        }

        if total_doc_count == 0 {
            return Ok(TokenWeightSet {
                tokens: vec![],
                idfs: vec![],
            });
        }

        // Compute IDF using the same formula as sparse vectors:
        // idf = ln((N - df + 0.5) / (df + 0.5) + 1)
        let n = total_doc_count as f32;
        let idfs: Vec<f32> = doc_frequencies
            .iter()
            .map(|&df| {
                let df = df as f32;
                ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();

        Ok(TokenWeightSet { tokens, idfs })
    }
}
