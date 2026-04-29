use std::collections::HashMap;
use std::fmt::Debug;

use ahash::AHashMap;
use common::counter::hardware_counter::HardwareCounterCell;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};
use itertools::Either;
use posting_list::{PostingBuilder, PostingIterator, PostingList, PostingListView, PostingValue};

use super::immutable_postings_enum::ImmutablePostings;
use super::mmap_inverted_index::MmapInvertedIndex;
use super::mmap_inverted_index::mmap_postings_enum::MmapPostingsEnum;
use super::mutable_inverted_index::MutableInvertedIndex;
use super::positions::Positions;
use super::postings_iterator::{
    intersect_compressed_postings_iterator, merge_compressed_postings_iterator,
};
use super::{Document, InvertedIndex, ParsedQuery, TokenId, TokenSet};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::inverted_index::positions::{
    WeightInfo, WeightInfoAndPositions,
};
use crate::index::field_index::full_text_index::inverted_index::postings_iterator::{
    check_compressed_postings_phrase, intersect_compressed_postings_phrase_iterator,
};
use crate::types::TokenWeightSet;

/// Collect posting-list views for every token in `token_ids`.
/// Returns `None` as soon as any token id is out of range.
fn get_all_or_none<'a, V: PostingValue>(
    postings: &'a [PostingList<V>],
    token_ids: &[TokenId],
) -> Option<Vec<(TokenId, PostingListView<'a, V>)>> {
    token_ids
        .iter()
        .map(|&token_id| {
            postings
                .get(token_id as usize)
                .map(|list| (token_id, list.view()))
        })
        .collect()
}

#[cfg_attr(test, derive(Clone))]
#[derive(Debug)]
pub struct ImmutableInvertedIndex {
    pub(in crate::index::field_index::full_text_index) postings: ImmutablePostings,
    pub(in crate::index::field_index::full_text_index) vocab: HashMap<String, TokenId>,
    pub(in crate::index::field_index::full_text_index) has_weight: bool,

    pub(in crate::index::field_index::full_text_index) point_to_tokens_count: Vec<usize>,
    pub(in crate::index::field_index::full_text_index) points_count: usize,
}

impl ImmutableInvertedIndex {
    /// Iterate over point ids whose documents contain all given tokens
    fn filter_has_all<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> impl Iterator<Item = PointOffsetType> + 'a {
        // in case of immutable index, deleted documents are still in the postings
        let filter = move |idx| {
            self.point_to_tokens_count
                .get(idx as usize)
                .is_some_and(|x| *x > 0)
        };

        fn intersection<'a, V: PostingValue>(
            postings: &'a [PostingList<V>],
            tokens: TokenSet,
            filter: impl Fn(PointOffsetType) -> bool + 'a,
        ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
            let postings_opt: Option<Vec<_>> = tokens
                .tokens()
                .iter()
                .map(|&token_id| postings.get(token_id as usize).map(PostingList::view))
                .collect();

            // All tokens must have postings
            let Some(postings) = postings_opt else {
                return Box::new(std::iter::empty());
            };

            // Query must not be empty
            if postings.is_empty() {
                return Box::new(std::iter::empty());
            };

            Box::new(intersect_compressed_postings_iterator(postings, filter))
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => {
                Either::Left(intersection(postings, tokens, filter))
            }
            ImmutablePostings::WithPositions(postings) => {
                Either::Right(intersection(postings, tokens, filter))
            }
            ImmutablePostings::WithWeight(postings) => {
                Either::Right(intersection(postings, tokens, filter))
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                Either::Right(intersection(postings, tokens, filter))
            }
        }
    }

    /// Iterate over point ids whose documents contain at least one of the given tokens
    fn filter_has_any<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> impl Iterator<Item = PointOffsetType> + 'a {
        // in case of immutable index, deleted documents are still in the postings
        let is_active = move |idx| {
            self.point_to_tokens_count
                .get(idx as usize)
                .is_some_and(|x| *x > 0)
        };

        fn merge<'a, V: PostingValue>(
            postings: &'a [PostingList<V>],
            tokens: TokenSet,
            is_active: impl Fn(PointOffsetType) -> bool + 'a,
        ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
            let postings: Vec<_> = tokens
                .tokens()
                .iter()
                .filter_map(|&token_id| postings.get(token_id as usize).map(PostingList::view))
                .collect();

            // Query must not be empty
            if postings.is_empty() {
                return Box::new(std::iter::empty());
            };

            Box::new(merge_compressed_postings_iterator(postings, is_active))
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => Either::Left(merge(postings, tokens, is_active)),
            ImmutablePostings::WithPositions(postings) => {
                Either::Right(merge(postings, tokens, is_active))
            }
            ImmutablePostings::WithWeight(postings) => {
                Either::Right(merge(postings, tokens, is_active))
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                Either::Right(merge(postings, tokens, is_active))
            }
        }
    }

    fn check_has_subset(&self, tokens: &TokenSet, point_id: PointOffsetType) -> bool {
        if tokens.is_empty() {
            return false;
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return false;
        }

        fn check_intersection<V: PostingValue>(
            postings: &[PostingList<V>],
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> bool {
            // Check that all tokens are in document
            tokens.tokens().iter().all(|token_id| {
                let posting_list = &postings[*token_id as usize];
                posting_list.visitor().contains(point_id)
            })
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => check_intersection(postings, tokens, point_id),
            ImmutablePostings::WithPositions(postings) => {
                check_intersection(postings, tokens, point_id)
            }
            ImmutablePostings::WithWeight(postings) => {
                check_intersection(postings, tokens, point_id)
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                check_intersection(postings, tokens, point_id)
            }
        }
    }

    fn check_has_any(&self, tokens: &TokenSet, point_id: PointOffsetType) -> bool {
        if tokens.is_empty() {
            return false;
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return false;
        }

        fn check_any<V: PostingValue>(
            postings: &[PostingList<V>],
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> bool {
            // Check that at least one token is in document
            tokens.tokens().iter().any(|token_id| {
                let posting_list = &postings[*token_id as usize];
                posting_list.visitor().contains(point_id)
            })
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => check_any(postings, tokens, point_id),
            ImmutablePostings::WithPositions(postings) => check_any(postings, tokens, point_id),
            ImmutablePostings::WithWeight(postings) => check_any(postings, tokens, point_id),
            ImmutablePostings::WithWeightAndPositions(postings) => {
                check_any(postings, tokens, point_id)
            }
        }
    }

    /// Iterate over point ids whose documents contain all given tokens in the same order they are provided
    pub fn filter_has_phrase<'a>(
        &'a self,
        phrase: Document,
    ) -> impl Iterator<Item = PointOffsetType> + 'a {
        // in case of mmap immutable index, deleted points are still in the postings
        let is_active = move |idx| {
            self.point_to_tokens_count
                .get(idx as usize)
                .is_some_and(|x| *x > 0)
        };

        match &self.postings {
            ImmutablePostings::WithPositions(postings) => {
                // Deduplicate phrase tokens: repeated tokens (e.g. "zn zn") must
                // not fetch the same posting list twice, otherwise positions get
                // added twice in `phrase_in_all_postings`.
                let unique_tokens = phrase.to_token_set();
                if let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens()) {
                    Either::Right(intersect_compressed_postings_phrase_iterator(
                        phrase,
                        selected_postings,
                        is_active,
                    ))
                } else {
                    Either::Left(std::iter::empty())
                }
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                // Deduplicate phrase tokens: repeated tokens (e.g. "zn zn") must
                // not fetch the same posting list twice, otherwise positions get
                // added twice in `phrase_in_all_postings`.
                let unique_tokens = phrase.to_token_set();
                if let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens()) {
                    Either::Right(intersect_compressed_postings_phrase_iterator(
                        phrase,
                        selected_postings,
                        is_active,
                    ))
                } else {
                    Either::Left(std::iter::empty())
                }
            }
            // cannot do phrase matching if there's no positional information
            ImmutablePostings::Ids(_postings) => Either::Left(std::iter::empty()),
            ImmutablePostings::WithWeight(_postings) => Either::Left(std::iter::empty()),
        }
    }

    /// Checks if the point document contains all given tokens in the same order they are provided
    pub fn check_has_phrase(&self, phrase: &Document, point_id: PointOffsetType) -> bool {
        // in case of mmap immutable index, deleted points are still in the postings
        if self
            .point_to_tokens_count
            .get(point_id as usize)
            .is_none_or(|x| *x == 0)
        {
            return false;
        }

        match &self.postings {
            ImmutablePostings::WithPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens())
                else {
                    return false;
                };

                check_compressed_postings_phrase(phrase, point_id, selected_postings)
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens())
                else {
                    return false;
                };

                check_compressed_postings_phrase(phrase, point_id, selected_postings)
            }
            // cannot do phrase matching if there's no positional information
            ImmutablePostings::Ids(_postings) => false,
            ImmutablePostings::WithWeight(_postings) => false,
        }
    }

    pub fn search_text_index_plain(
        &self,
        query: &TokenWeightSet,
        top: usize,
        ordered_prefiltered_points: &[PointOffsetType],
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if !self.has_weight {
            return Ok(vec![]);
        }

        /// Generic helper: collect posting iterators for query tokens, then
        /// iterate through sorted prefiltered points, advancing each iterator
        /// in tandem (just like the sparse-vector `plain_search` pipeline).
        fn score_with_iterators<V: PostingValue>(
            postings: &[PostingList<V>],
            vocab: &HashMap<String, TokenId>,
            query: &TokenWeightSet,
            top: usize,
            ordered_prefiltered_points: &[PointOffsetType],
            extract_weight: fn(&V) -> f32,
        ) -> Vec<ScoredPointOffset> {
            let mut iterators: Vec<(PostingIterator<'_, V>, f32)> = query
                .tokens
                .iter()
                .zip(query.idfs.iter())
                .filter_map(|(token, &idf)| {
                    let tid = *vocab.get(token.as_str())?;
                    let iter = postings.get(tid as usize)?.iter();
                    Some((iter, idf))
                })
                .collect();

            if iterators.is_empty() {
                return vec![];
            }

            let mut top_k = TopK::new(top);

            for &point_id in ordered_prefiltered_points {
                let mut score = 0.0f32;
                for (iter, idf) in iterators.iter_mut() {
                    if let Some(elem) = iter.advance_until_greater_or_equal(point_id) {
                        if elem.id == point_id {
                            score += extract_weight(&elem.value) * *idf;
                        }
                    }
                }
                if score > 0.0 {
                    top_k.push(ScoredPointOffset {
                        idx: point_id,
                        score,
                    });
                }
            }

            top_k.into_vec()
        }

        let result = match &self.postings {
            ImmutablePostings::WithWeight(postings) => score_with_iterators(
                postings,
                &self.vocab,
                query,
                top,
                ordered_prefiltered_points,
                |w| w.token_weight(),
            ),
            ImmutablePostings::WithWeightAndPositions(postings) => score_with_iterators(
                postings,
                &self.vocab,
                query,
                top,
                ordered_prefiltered_points,
                |w| w.token_weight(),
            ),
            // ID-only or position-only postings have no weights
            ImmutablePostings::Ids(_) | ImmutablePostings::WithPositions(_) => vec![],
        };

        Ok(result)
    }

    /// BM25 scoring with filter and pruning for immutable compressed posting lists.
    ///
    /// Follows the sparse vector batched search pipeline: iterates posting
    /// lists in batches, accumulates TF·IDF scores, applies the filter, and
    /// uses `max_next_weight` to prune the longest posting list when the
    /// top-k heap is full.
    pub fn search_text_index<F>(
        &self,
        query: &TokenWeightSet,
        top: usize,
        filter: F,
    ) -> OperationResult<Vec<ScoredPointOffset>>
    where
        F: Fn(PointOffsetType) -> bool,
    {
        if !self.has_weight {
            return Ok(vec![]);
        }

        /// Trait to abstract over `WeightInfo` and `WeightInfoAndPositions`.
        trait HasWeights {
            fn token_weight(&self) -> f32;
            fn max_next_weight(&self) -> f32;
        }

        impl HasWeights for WeightInfo {
            fn token_weight(&self) -> f32 {
                WeightInfo::token_weight(self)
            }
            fn max_next_weight(&self) -> f32 {
                WeightInfo::max_next_weight(self)
            }
        }

        impl HasWeights for WeightInfoAndPositions {
            fn token_weight(&self) -> f32 {
                WeightInfoAndPositions::token_weight(self)
            }
            fn max_next_weight(&self) -> f32 {
                WeightInfoAndPositions::max_next_weight(self)
            }
        }

        fn search_inner<V: PostingValue + HasWeights>(
            postings: &[PostingList<V>],
            vocab: &HashMap<String, TokenId>,
            query: &TokenWeightSet,
            top: usize,
            filter: impl Fn(PointOffsetType) -> bool,
        ) -> Vec<ScoredPointOffset> {
            struct IndexedIterator<'a, V: PostingValue> {
                iter: PostingIterator<'a, V>,
                idf: f32,
            }

            let mut iterators: Vec<IndexedIterator<'_, V>> = query
                .tokens
                .iter()
                .zip(query.idfs.iter())
                .filter_map(|(token, &idf)| {
                    let tid = *vocab.get(token.as_str())?;
                    let iter = postings.get(tid as usize)?.iter();
                    Some(IndexedIterator { iter, idf })
                })
                .collect();

            if iterators.is_empty() {
                return vec![];
            }

            let mut top_k = TopK::new(top);

            // Find global min/max point IDs.
            let mut min_id = PointOffsetType::MAX;
            let mut max_id: PointOffsetType = 0;
            // Get max_id from posting list views before creating iterators
            for (token, _) in query.tokens.iter().zip(query.idfs.iter()) {
                if let Some(tid) = vocab.get(token.as_str()) {
                    if let Some(posting) = postings.get(*tid as usize) {
                        if let Some(last_id) = posting.view().get_last_id() {
                            max_id = max_id.max(last_id);
                        }
                    }
                }
            }
            for ii in &mut iterators {
                if let Some(elem) = ii.iter.advance_until_greater_or_equal(0) {
                    min_id = min_id.min(elem.id);
                }
            }

            if min_id > max_id {
                return vec![];
            }

            const BATCH_SIZE: u32 = 10_000;
            let mut batch_start = min_id;
            let mut best_min_score = f32::MIN;

            loop {
                if batch_start > max_id {
                    break;
                }
                let batch_last = batch_start.saturating_add(BATCH_SIZE).min(max_id);

                // ── batch accumulation ──
                let batch_len = (batch_last - batch_start + 1) as usize;
                let mut scores = vec![0.0f32; batch_len];

                for ii in iterators.iter_mut() {
                    // Advance through elements up to batch_last
                    loop {
                        let Some(elem) = ii.iter.advance_until_greater_or_equal(batch_start) else {
                            break;
                        };
                        if elem.id > batch_last {
                            break;
                        }
                        let local = (elem.id - batch_start) as usize;
                        scores[local] += elem.value.token_weight() * ii.idf;
                        // Advance past current element
                        ii.iter.next();
                    }
                }

                // ── publish scored points ──
                for (local, &score) in scores.iter().enumerate() {
                    if score > 0.0 && score > top_k.threshold() {
                        let point_id = batch_start + local as PointOffsetType;
                        if filter(point_id) {
                            top_k.push(ScoredPointOffset {
                                idx: point_id,
                                score,
                            });
                        }
                    }
                }

                // ── remove exhausted iterators ──
                iterators.retain(|ii| ii.iter.len() > 0);

                if iterators.is_empty() {
                    break;
                }

                // Fast-path: single iterator left
                if iterators.len() == 1 {
                    let ii = &mut iterators[0];
                    for elem in ii.iter.by_ref() {
                        if filter(elem.id) {
                            let score = elem.value.token_weight() * ii.idf;
                            top_k.push(ScoredPointOffset {
                                idx: elem.id,
                                score,
                            });
                        }
                    }
                    break;
                }

                // ── pruning with max_next_weight ──
                if top_k.len() >= top {
                    let new_min_score = top_k.threshold();
                    if new_min_score > best_min_score {
                        best_min_score = new_min_score;

                        // Promote longest iterator to front
                        let longest_idx = iterators
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, ii)| ii.iter.len())
                            .map(|(i, _)| i)
                            .unwrap();
                        if longest_idx != 0 {
                            iterators.swap(0, longest_idx);
                        }

                        // Try pruning the longest iterator
                        if let Some(elem) = iterators[0]
                            .iter
                            .advance_until_greater_or_equal(batch_start)
                        {
                            let next_min_in_others = iterators[1..]
                                .iter_mut()
                                .filter_map(|ii| {
                                    ii.iter
                                        .advance_until_greater_or_equal(batch_start)
                                        .map(|e| e.id)
                                })
                                .min();

                            let can_prune = match next_min_in_others {
                                Some(next_min) if next_min > elem.id => {
                                    let max_weight =
                                        elem.value.token_weight().max(elem.value.max_next_weight());
                                    max_weight * iterators[0].idf <= new_min_score
                                }
                                None => {
                                    let max_weight =
                                        elem.value.token_weight().max(elem.value.max_next_weight());
                                    max_weight * iterators[0].idf <= new_min_score
                                }
                                _ => false,
                            };

                            if can_prune {
                                match next_min_in_others {
                                    Some(next_min) => {
                                        iterators[0].iter.advance_until_greater_or_equal(next_min);
                                    }
                                    None => {
                                        // Exhaust the iterator
                                        for _ in iterators[0].iter.by_ref() {}
                                    }
                                }
                            }
                        }
                    }
                }

                // Update batch_start from remaining iterators
                batch_start = iterators
                    .iter_mut()
                    .filter_map(|ii| {
                        ii.iter
                            .advance_until_greater_or_equal(batch_start)
                            .map(|e| e.id)
                    })
                    .min()
                    .unwrap_or(max_id + 1);
            }

            top_k.into_vec()
        }

        let result = match &self.postings {
            ImmutablePostings::WithWeight(postings) => {
                search_inner(postings, &self.vocab, query, top, filter)
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                search_inner(postings, &self.vocab, query, top, filter)
            }
            ImmutablePostings::Ids(_) | ImmutablePostings::WithPositions(_) => vec![],
        };

        Ok(result)
    }
}

impl InvertedIndex for ImmutableInvertedIndex {
    fn get_vocab_mut(&mut self) -> &mut HashMap<String, TokenId> {
        &mut self.vocab
    }

    fn index_tokens(
        &mut self,
        _idx: PointOffsetType,
        _tokens: super::TokenSet,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to immutable text index",
        ))
    }

    fn index_token_weight_map(
        &mut self,
        _idx: PointOffsetType,
        _token_weight_map: super::TokenWeightMap,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to immutable text index",
        ))
    }

    fn index_document(
        &mut self,
        _idx: PointOffsetType,
        _document: super::Document,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to immutable text index",
        ))
    }

    fn remove(&mut self, idx: PointOffsetType) -> bool {
        if self.values_is_empty(idx) {
            return false; // Already removed or never actually existed
        }
        self.point_to_tokens_count[idx as usize] = 0;
        self.points_count -= 1;
        true
    }

    fn filter<'a>(
        &'a self,
        query: ParsedQuery,
        _hw_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + 'a>> {
        match query {
            ParsedQuery::AllTokens(tokens) => Ok(Box::new(self.filter_has_all(tokens))),
            ParsedQuery::Phrase(tokens) => Ok(Box::new(self.filter_has_phrase(tokens))),
            ParsedQuery::AnyTokens(tokens) => Ok(Box::new(self.filter_has_any(tokens))),
        }
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        _: &HardwareCounterCell,
    ) -> OperationResult<Option<usize>> {
        Ok(self.postings.posting_len(token_id))
    }

    fn vocab_with_postings_len_iter(
        &self,
    ) -> impl Iterator<Item = OperationResult<(&str, usize)>> + '_ {
        self.vocab.iter().filter_map(|(token, &token_id)| {
            self.postings
                .posting_len(token_id)
                .map(|len| Ok((token.as_str(), len)))
        })
    }

    fn check_match(
        &self,
        parsed_query: &ParsedQuery,
        point_id: PointOffsetType,
    ) -> OperationResult<bool> {
        let matched = match parsed_query {
            ParsedQuery::AllTokens(tokens) => self.check_has_subset(tokens, point_id),
            ParsedQuery::Phrase(phrase) => self.check_has_phrase(phrase, point_id),
            ParsedQuery::AnyTokens(tokens) => self.check_has_any(tokens, point_id),
        };
        Ok(matched)
    }

    fn values_is_empty(&self, point_id: PointOffsetType) -> bool {
        self.point_to_tokens_count
            .get(point_id as usize)
            .is_none_or(|count| *count == 0)
    }

    fn values_count(&self, point_id: PointOffsetType) -> usize {
        self.point_to_tokens_count
            .get(point_id as usize)
            .copied()
            .unwrap_or(0)
    }

    fn points_count(&self) -> usize {
        self.points_count
    }

    fn get_token_id(&self, token: &str, _: &HardwareCounterCell) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

impl From<MutableInvertedIndex> for ImmutableInvertedIndex {
    fn from(index: MutableInvertedIndex) -> Self {
        let MutableInvertedIndex {
            postings,
            vocab,
            point_to_tokens,
            has_weight,
            point_to_doc,
            points_count,
        } = index;

        let (postings, vocab, orig_to_new_token) = optimized_postings_and_vocab(postings, vocab);

        let postings = match (has_weight, point_to_doc) {
            (false, None) => ImmutablePostings::Ids(create_compressed_postings(postings)),
            (false, Some(point_to_doc)) => {
                ImmutablePostings::WithPositions(create_compressed_postings_with_positions(
                    postings,
                    point_to_doc,
                    &orig_to_new_token,
                ))
            }
            (true, None) => {
                ImmutablePostings::WithWeight(create_compressed_postings_with_weight(postings))
            }
            (true, Some(point_to_doc)) => ImmutablePostings::WithWeightAndPositions(
                create_compressed_postings_with_weight_and_positions(
                    postings,
                    point_to_doc,
                    &orig_to_new_token,
                ),
            ),
        };

        ImmutableInvertedIndex {
            postings,
            vocab,
            has_weight,
            point_to_tokens_count: point_to_tokens
                .iter()
                .map(|tokenset| {
                    tokenset
                        .as_ref()
                        .map(|tokenset| tokenset.len())
                        .unwrap_or(0)
                })
                .collect(),
            points_count,
        }
    }
}

fn optimized_postings_and_vocab(
    postings: Vec<super::posting_list::PostingList>,
    vocab: HashMap<String, u32>,
) -> (
    Vec<super::posting_list::PostingList>,
    HashMap<String, u32>,
    AHashMap<u32, u32>,
) {
    // Keep only tokens that have non-empty postings
    let (postings, orig_to_new_token): (Vec<_>, AHashMap<_, _>) = postings
        .into_iter()
        .enumerate()
        .filter_map(|(orig_token, posting)| (!posting.is_empty()).then_some((orig_token, posting)))
        .enumerate()
        .map(|(new_token, (orig_token, posting))| {
            (posting, (orig_token as TokenId, new_token as TokenId))
        })
        .unzip();

    // Update vocab entries
    let mut vocab: HashMap<String, TokenId> = vocab
        .into_iter()
        .filter_map(|(key, orig_token)| {
            orig_to_new_token
                .get(&orig_token)
                .map(|new_token| (key, *new_token))
        })
        .collect();

    vocab.shrink_to_fit();

    (postings, vocab, orig_to_new_token)
}

fn create_compressed_postings(
    postings: Vec<super::posting_list::PostingList>,
) -> Vec<PostingList<()>> {
    postings
        .into_iter()
        .map(|posting| {
            let mut builder = PostingBuilder::new();
            for id in posting.iter() {
                builder.add_id(id);
            }
            builder.build()
        })
        .collect()
}

fn create_compressed_postings_with_weight(
    postings: Vec<super::posting_list::PostingList>,
) -> Vec<PostingList<WeightInfo>> {
    postings
        .into_iter()
        .map(|posting| {
            let mut builder = PostingBuilder::new();
            for element in posting.iter_element() {
                builder.add(
                    element.point_id(),
                    WeightInfo::new(element.weight(), element.max_next_weight()),
                );
            }
            builder.build()
        })
        .collect()
}

fn create_compressed_postings_with_weight_and_positions(
    postings: Vec<super::posting_list::PostingList>,
    point_to_doc: Vec<Option<Document>>,
    orig_to_new_token: &AHashMap<TokenId, TokenId>,
) -> Vec<PostingList<WeightInfoAndPositions>> {
    // precalculate positions for each token in each document
    let mut point_to_tokens_positions: Vec<AHashMap<TokenId, Vec<u32>>> = point_to_doc
        .into_iter()
        .map(|doc_opt| {
            let Some(doc) = doc_opt else {
                return AHashMap::new();
            };

            // get positions for each token in the document
            let doc_len = doc.len();
            (0u32..).zip(doc).fold(
                AHashMap::with_capacity(doc_len),
                |mut map: AHashMap<u32, Vec<u32>>, (position, token)| {
                    // use translation of original token to new token from postings optimization
                    let new_token = orig_to_new_token[&token];
                    map.entry(new_token).or_default().push(position);
                    map
                },
            )
        })
        .collect::<Vec<_>>();

    (0u32..)
            .zip(postings)
            .map(|(token, posting)| {
                posting
                    .iter_element()
                    .map(|element| {
                        let id = element.point_id();
                        let positions = point_to_tokens_positions[id as usize]
                            .remove(&token)
                            .expect(
                                "If id is this token's posting list, it should have at least one position",
                            );
                        let weight_and_positions = WeightInfoAndPositions::new(element.weight(), element.max_next_weight(), positions);
                        (id, weight_and_positions)
                    })
                    .collect()
            })
            .collect()
}

fn create_compressed_postings_with_positions(
    postings: Vec<super::posting_list::PostingList>,
    point_to_doc: Vec<Option<Document>>,
    orig_to_new_token: &AHashMap<TokenId, TokenId>,
) -> Vec<PostingList<Positions>> {
    // precalculate positions for each token in each document
    let mut point_to_tokens_positions: Vec<AHashMap<TokenId, Positions>> = point_to_doc
        .into_iter()
        .map(|doc_opt| {
            let Some(doc) = doc_opt else {
                return AHashMap::new();
            };

            // get positions for each token in the document
            let doc_len = doc.len();
            (0u32..).zip(doc).fold(
                AHashMap::with_capacity(doc_len),
                |mut map: AHashMap<u32, Positions>, (position, token)| {
                    // use translation of original token to new token from postings optimization
                    let new_token = orig_to_new_token[&token];
                    map.entry(new_token).or_default().push(position);
                    map
                },
            )
        })
        .collect::<Vec<_>>();

    (0u32..)
            .zip(postings)
            .map(|(token, posting)| {
                posting
                    .iter()
                    .map(|id| {
                        let positions = point_to_tokens_positions[id as usize]
                            .remove(&token)
                            .expect(
                                "If id is this token's posting list, it should have at least one position",
                            );
                        (id, positions)
                    })
                    .collect()
            })
            .collect()
}

impl TryFrom<&MmapInvertedIndex> for ImmutableInvertedIndex {
    type Error = OperationError;

    fn try_from(index: &MmapInvertedIndex) -> OperationResult<Self> {
        let postings = match &index.storage.postings {
            MmapPostingsEnum::Ids(postings) => ImmutablePostings::Ids(postings.all_postings()?),
            MmapPostingsEnum::WithPositions(postings) => {
                ImmutablePostings::WithPositions(postings.all_postings()?)
            }
            MmapPostingsEnum::WithWeight(postings) => {
                ImmutablePostings::WithWeight(postings.all_postings()?)
            }
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                ImmutablePostings::WithWeightAndPositions(postings.all_postings()?)
            }
        };

        let vocab: HashMap<String, TokenId> = index
            .storage
            .vocab
            .iter()
            .map(|(token_str, token_id)| (token_str.to_owned(), token_id[0]))
            .collect();

        debug_assert!(
            postings.len() == vocab.len(),
            "postings and vocab must be the same size",
        );

        Ok(ImmutableInvertedIndex {
            postings,
            vocab,
            has_weight: index.has_weight,
            point_to_tokens_count: index.storage.point_to_tokens_count.to_vec(),
            points_count: index.points_count(),
        })
    }
}

impl ImmutableInvertedIndex {
    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        let Self {
            postings,
            vocab,
            has_weight: _,
            point_to_tokens_count,
            points_count: _,
        } = self;

        let postings_bytes = postings.ram_usage_bytes();
        // HashMap per-slot overhead: hash (u64) + metadata pointer
        let hashmap_entry_overhead = std::mem::size_of::<u64>() + std::mem::size_of::<usize>();
        let vocab_base_bytes = vocab.capacity()
            * (std::mem::size_of::<String>()
                + std::mem::size_of::<TokenId>()
                + hashmap_entry_overhead);
        // Account for actual heap-allocated string data
        let vocab_heap_bytes: usize = vocab.keys().map(|s| s.capacity()).sum();
        let pttc_bytes = point_to_tokens_count.capacity() * std::mem::size_of::<usize>();
        postings_bytes + vocab_base_bytes + vocab_heap_bytes + pttc_bytes
    }
}
