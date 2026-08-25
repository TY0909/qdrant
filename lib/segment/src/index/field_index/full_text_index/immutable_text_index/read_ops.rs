use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::{UniversalRead, UserData};

use super::super::full_text_index_read::{FullTextIndexRead, default_check_match_batch};
use super::super::full_text_index_scoring::FullTextIndexScoring;
use super::super::inverted_index::{
    InvertedIndex, InvertedIndexScoring, ParsedQuery, TokenId, bm25_scoring_not_enabled_error,
};
use super::super::tokenizers::Tokenizer;
use super::ImmutableFullTextIndex;
use crate::common::operation_error::OperationResult;
use crate::index::field_index::{CardinalityEstimation, PayloadBlockCondition};
use crate::index::payload_config::StorageType;
use crate::types::{FieldCondition, PayloadKeyType, QueryTokenWeightSet};

impl<S: UniversalRead> FullTextIndexRead for ImmutableFullTextIndex<S> {
    fn tokenizer(&self) -> &Tokenizer {
        &self.storage.tokenizer
    }

    fn telemetry_index_type(&self) -> &'static str {
        "immutable_full_text"
    }

    fn points_count(&self) -> usize {
        self.inverted_index.points_count()
    }

    fn document_length(
        &self,
        point_id: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<u32>> {
        self.inverted_index.document_length(point_id, hw_counter)
    }

    fn values_count(&self, point_id: PointOffsetType) -> usize {
        self.inverted_index.values_count(point_id)
    }

    fn values_is_empty(&self, point_id: PointOffsetType) -> bool {
        self.inverted_index.values_is_empty(point_id)
    }

    fn for_each_token_id<'a, U: UserData>(
        &self,
        iter: impl Iterator<Item = (U, &'a str)>,
        hw_counter: &HardwareCounterCell,
        f: impl FnMut(U, Option<TokenId>),
    ) -> OperationResult<()> {
        self.inverted_index.for_each_token_id(iter, hw_counter, f)
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<usize>> {
        self.inverted_index.get_posting_len(token_id, hw_counter)
    }

    fn filter_query<'a>(
        &'a self,
        query: ParsedQuery,
        hw_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + 'a>> {
        self.inverted_index.filter(query, hw_counter)
    }

    fn estimate_query_cardinality(
        &self,
        query: &ParsedQuery,
        condition: &FieldCondition,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<CardinalityEstimation> {
        self.inverted_index
            .estimate_cardinality(query, condition, hw_counter)
    }

    fn check_match(&self, query: &ParsedQuery, point_id: PointOffsetType) -> OperationResult<bool> {
        self.inverted_index.check_match(query, point_id)
    }

    fn check_match_batch<U: UserData>(
        &self,
        query: &ParsedQuery,
        items: impl Iterator<Item = (U, PointOffsetType)>,
        on_match: impl FnMut(U, bool),
    ) -> OperationResult<()> {
        default_check_match_batch(self, query, items, on_match)
    }

    fn for_each_payload_block_inner(
        &self,
        threshold: usize,
        key: PayloadKeyType,
        f: &mut dyn FnMut(PayloadBlockCondition) -> OperationResult<()>,
    ) -> OperationResult<()> {
        self.inverted_index
            .for_each_payload_block(threshold, key, f)
    }

    fn get_storage_type(&self) -> StorageType {
        StorageType::Mmap { is_on_disk: false }
    }

    fn ram_usage_bytes(&self) -> usize {
        self.cached_ram_usage_bytes
    }

    fn is_on_disk(&self) -> bool {
        false
    }
}

impl<S: UniversalRead> FullTextIndexScoring for ImmutableFullTextIndex<S> {
    fn search_text_index<F>(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        is_stopped: &AtomicBool,
        filter: F,
    ) -> OperationResult<Vec<ScoredPointOffset>>
    where
        F: Fn(PointOffsetType) -> bool,
    {
        if top == 0 {
            return Ok(Vec::new());
        }
        self.inverted_index.search_text_index(
            query,
            self.storage
                .bm25_params
                .ok_or_else(bm25_scoring_not_enabled_error)?,
            top,
            is_stopped,
            filter,
        )
    }

    fn search_text_index_plain(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        ordered_prefiltered_points: &[PointOffsetType],
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if top == 0 {
            return Ok(Vec::new());
        }
        self.inverted_index.search_text_index_plain(
            query,
            self.storage
                .bm25_params
                .ok_or_else(bm25_scoring_not_enabled_error)?,
            top,
            ordered_prefiltered_points,
            is_stopped,
        )
    }
}
