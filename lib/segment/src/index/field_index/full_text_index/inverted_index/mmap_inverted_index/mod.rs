use std::collections::HashMap;
use std::path::PathBuf;

use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::mmap::{self, Advice, AdviceSetting, MmapSlice, create_and_ensure_length};
use common::mmap_hashmap::{MmapHashMap, READ_ENTRY_OVERHEAD};
use common::stored_bitslice::MmapBitSlice;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::{MmapFile, OpenOptions};
use posting_list::{PostingIterator, PostingValue};
use types::ZerocopyPostingValue;
use uio_postings::UniversalPostings;

use self::create_postings::create_postings_file;
use super::immutable_inverted_index::ImmutableInvertedIndex;
use super::immutable_postings_enum::ImmutablePostings;
use super::mmap_inverted_index::mmap_postings_enum::MmapPostingsEnum;
use super::positions::Positions;
use super::postings_iterator::{
    intersect_compressed_postings_iterator, merge_compressed_postings_iterator,
};
use super::{InvertedIndex, ParsedQuery, TokenId, TokenSet};
use crate::common::Flusher;
use crate::common::buffered_update_bitslice::BufferedUpdateBitSlice;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::inverted_index::Document;
use crate::index::field_index::full_text_index::inverted_index::positions::{
    WeightInfo, WeightInfoAndPositions,
};
use crate::index::field_index::full_text_index::inverted_index::postings_iterator::{
    check_compressed_postings_phrase, intersect_compressed_postings_phrase_iterator,
};
use crate::types::TokenWeightSet;

mod create_postings;
pub mod mmap_postings_enum;
mod raw_posting_list;
pub(in crate::index::field_index::full_text_index) mod types;
mod uio_postings;

const POSTINGS_FILE: &str = "postings.dat";
const VOCAB_FILE: &str = "vocab.dat";
const POINT_TO_TOKENS_COUNT_FILE: &str = "point_to_tokens_count.dat";
const DELETED_POINTS_FILE: &str = "deleted_points.dat";

pub struct MmapInvertedIndex {
    pub(in crate::index::field_index::full_text_index) path: PathBuf,
    pub(in crate::index::field_index::full_text_index) storage: Storage,
    pub(in crate::index::field_index::full_text_index) has_weight: bool,
    /// Number of points which are not deleted
    pub(in crate::index::field_index::full_text_index) active_points_count: usize,
    is_on_disk: bool,
}

pub(in crate::index::field_index::full_text_index) struct Storage {
    pub(in crate::index::field_index::full_text_index) postings: MmapPostingsEnum,
    pub(in crate::index::field_index::full_text_index) vocab: MmapHashMap<str, TokenId>,
    pub(in crate::index::field_index::full_text_index) point_to_tokens_count: MmapSlice<usize>,
    pub(in crate::index::field_index::full_text_index) deleted_points:
        BufferedUpdateBitSlice<MmapFile>,
}

impl MmapInvertedIndex {
    pub fn create(path: PathBuf, inverted_index: &ImmutableInvertedIndex) -> OperationResult<()> {
        let ImmutableInvertedIndex {
            postings,
            vocab,
            has_weight: _,
            point_to_tokens_count,
            points_count: _,
        } = inverted_index;

        debug_assert_eq!(vocab.len(), postings.len());

        let postings_path = path.join(POSTINGS_FILE);
        let vocab_path = path.join(VOCAB_FILE);
        let point_to_tokens_count_path = path.join(POINT_TO_TOKENS_COUNT_FILE);
        let deleted_points_path = path.join(DELETED_POINTS_FILE);

        match postings {
            ImmutablePostings::Ids(postings) => create_postings_file(postings_path, postings)?,
            ImmutablePostings::WithPositions(postings) => {
                create_postings_file(postings_path, postings)?
            }
            ImmutablePostings::WithWeight(postings) => {
                create_postings_file(postings_path, postings)?
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                create_postings_file(postings_path, postings)?
            }
        }

        // Currently MmapHashMap maps str -> [u32], but we only need to map str -> u32.
        // TODO: Consider making another mmap structure for this case.
        MmapHashMap::<str, TokenId>::create(
            &vocab_path,
            vocab.iter().map(|(k, v)| (k.as_str(), std::iter::once(*v))),
        )?;

        // Save point_to_tokens_count, separated into a bitslice for None values and a slice for actual values
        //
        // None values are represented as deleted in the bitslice
        let deleted_bitslice: BitVec = point_to_tokens_count
            .iter()
            .map(|count| *count == 0)
            .collect();
        {
            let deleted_flags_count = deleted_bitslice.len();
            let _ = create_and_ensure_length(
                &deleted_points_path,
                deleted_flags_count
                    .div_ceil(u8::BITS as usize)
                    .next_multiple_of(size_of::<u64>()),
            )?;

            let mut deleted_storage =
                MmapBitSlice::open(&deleted_points_path, OpenOptions::default())?;
            deleted_storage.write_bitslice(&deleted_bitslice)?;
            deleted_storage.flusher()()?;
        }

        // The actual values go in the slice
        let point_to_tokens_count_iter = point_to_tokens_count.iter().copied();

        MmapSlice::create(&point_to_tokens_count_path, point_to_tokens_count_iter)?;

        Ok(())
    }

    pub fn open(
        path: PathBuf,
        populate: bool,
        has_positions: bool,
        has_weight: bool,
    ) -> OperationResult<Option<Self>> {
        let postings_path = path.join(POSTINGS_FILE);
        let vocab_path = path.join(VOCAB_FILE);
        let point_to_tokens_count_path = path.join(POINT_TO_TOKENS_COUNT_FILE);
        let deleted_points_path = path.join(DELETED_POINTS_FILE);

        // If postings don't exist, assume the index doesn't exist on disk
        if !postings_path.is_file() {
            return Ok(None);
        }

        let postings_open_options = OpenOptions {
            writeable: false,
            need_sequential: false,
            disk_parallel: None,
            populate: Some(populate),
            advice: Some(AdviceSetting::Advice(Advice::Normal)),
            prevent_caching: None,
        };
        let postings = match (has_weight, has_positions) {
            (false, false) => MmapPostingsEnum::Ids(UniversalPostings::<(), MmapFile>::open(
                &postings_path,
                postings_open_options,
            )?),
            (false, true) => {
                MmapPostingsEnum::WithPositions(UniversalPostings::<Positions, MmapFile>::open(
                    &postings_path,
                    postings_open_options,
                )?)
            }
            (true, false) => {
                MmapPostingsEnum::WithWeight(UniversalPostings::<WeightInfo, MmapFile>::open(
                    &postings_path,
                    postings_open_options,
                )?)
            }
            (true, true) => MmapPostingsEnum::WithWeightAndPositions(UniversalPostings::<
                WeightInfoAndPositions,
                MmapFile,
            >::open(
                &postings_path,
                postings_open_options,
            )?),
        };
        let vocab = MmapHashMap::<str, TokenId>::open(&vocab_path, false)?;

        let point_to_tokens_count = unsafe {
            MmapSlice::try_from(mmap::open_write_mmap(
                &point_to_tokens_count_path,
                AdviceSetting::Global,
                populate,
            )?)?
        };

        let deleted = MmapBitSlice::open(
            &deleted_points_path,
            OpenOptions {
                populate: Some(populate),
                ..OpenOptions::default()
            },
        )?;
        let num_deleted_points = deleted.count_ones()?;
        let deleted_points = BufferedUpdateBitSlice::new(deleted);
        let points_count = point_to_tokens_count.len() - num_deleted_points;

        Ok(Some(Self {
            path,
            storage: Storage {
                postings,
                vocab,
                point_to_tokens_count,
                deleted_points,
            },
            has_weight,
            active_points_count: points_count,
            is_on_disk: !populate,
        }))
    }

    pub(super) fn iter_vocab(&self) -> impl Iterator<Item = (&str, &TokenId)> + '_ {
        // unwrap safety: we know that each token points to a token id.
        self.storage
            .vocab
            .iter()
            .map(|(k, v)| (k, v.first().unwrap()))
    }

    /// Returns whether the point id is valid and active.
    pub fn is_active(&self, point_id: PointOffsetType) -> bool {
        let is_deleted = self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true);
        !is_deleted
    }

    /// Iterate over point ids whose documents contain all given tokens.
    ///
    /// Pre-collected upfront because [`UniversalPostings`] exposes posting
    /// views via a `FnOnce` callback. Acceptable since this index lives
    /// on disk.
    pub fn filter_has_all(&self, tokens: TokenSet) -> OperationResult<Vec<PointOffsetType>> {
        // in case of mmap immutable index, deleted points are still in the postings
        let filter = move |idx| self.is_active(idx);

        fn intersection<V: ZerocopyPostingValue>(
            postings: &UniversalPostings<V, MmapFile>,
            tokens: TokenSet,
            filter: impl Fn(PointOffsetType) -> bool,
        ) -> OperationResult<Vec<PointOffsetType>> {
            let result =
                postings.with_all_or_none_postings(tokens.tokens(), |posting_readers| {
                    if posting_readers.is_empty() {
                        return Ok(Vec::new());
                    }
                    let posting_readers = posting_readers
                        .into_iter()
                        .map(|(_token_id, posting_list_view)| posting_list_view)
                        .collect();
                    Ok(intersect_compressed_postings_iterator(posting_readers, filter).collect())
                })?;
            // Some token has no posting list -> no matches
            Ok(result.unwrap_or_default())
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => intersection(postings, tokens, filter),
            MmapPostingsEnum::WithPositions(postings) => intersection(postings, tokens, filter),
            MmapPostingsEnum::WithWeight(postings) => intersection(postings, tokens, filter),
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                intersection(postings, tokens, filter)
            }
        }
    }

    /// Iterate over point ids whose documents contain at least one of the given tokens
    fn filter_has_any(&self, tokens: TokenSet) -> OperationResult<Vec<PointOffsetType>> {
        // in case of immutable index, deleted documents are still in the postings
        let is_active = move |idx| self.is_active(idx);

        fn merge<V: ZerocopyPostingValue>(
            postings: &UniversalPostings<V, MmapFile>,
            tokens: TokenSet,
            is_active: impl Fn(PointOffsetType) -> bool,
        ) -> OperationResult<Vec<PointOffsetType>> {
            postings.with_existing_postings(tokens.tokens(), |posting_readers| {
                if posting_readers.is_empty() {
                    return Ok(Vec::new());
                }
                let posting_readers = posting_readers
                    .into_iter()
                    .map(|(_token_id, posting_list_view)| posting_list_view)
                    .collect();
                Ok(merge_compressed_postings_iterator(posting_readers, is_active).collect())
            })
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => merge(postings, tokens, is_active),
            MmapPostingsEnum::WithPositions(postings) => merge(postings, tokens, is_active),
            MmapPostingsEnum::WithWeight(postings) => merge(postings, tokens, is_active),
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                merge(postings, tokens, is_active)
            }
        }
    }

    fn check_has_subset(
        &self,
        tokens: &TokenSet,
        point_id: PointOffsetType,
    ) -> OperationResult<bool> {
        // check non-empty query
        if tokens.is_empty() {
            return Ok(false);
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return Ok(false);
        }

        fn check_intersection<V: ZerocopyPostingValue>(
            postings: &UniversalPostings<V, MmapFile>,
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> OperationResult<bool> {
            let result = postings.with_all_or_none_postings(tokens.tokens(), |all_postings| {
                Ok(all_postings
                    .into_iter()
                    .all(|(_token_id, posting)| posting.visitor().contains(point_id)))
            })?;
            // Some token has no posting list -> no match
            Ok(result.unwrap_or(false))
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => check_intersection(postings, tokens, point_id),
            MmapPostingsEnum::WithPositions(postings) => {
                check_intersection(postings, tokens, point_id)
            }
            MmapPostingsEnum::WithWeight(postings) => {
                check_intersection(postings, tokens, point_id)
            }
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                check_intersection(postings, tokens, point_id)
            }
        }
    }

    fn check_has_any(&self, tokens: &TokenSet, point_id: PointOffsetType) -> OperationResult<bool> {
        if tokens.is_empty() {
            return Ok(false);
        }

        // check presence of the document
        if self.values_is_empty(point_id) {
            return Ok(false);
        }

        fn check_any<V: ZerocopyPostingValue>(
            postings: &UniversalPostings<V, MmapFile>,
            tokens: &TokenSet,
            point_id: PointOffsetType,
        ) -> OperationResult<bool> {
            postings.with_existing_postings(tokens.tokens(), |all_postings| {
                Ok(all_postings
                    .into_iter()
                    .any(|(_token_id, posting)| posting.visitor().contains(point_id)))
            })
        }

        match &self.storage.postings {
            MmapPostingsEnum::Ids(postings) => check_any(postings, tokens, point_id),
            MmapPostingsEnum::WithPositions(postings) => check_any(postings, tokens, point_id),
            MmapPostingsEnum::WithWeight(postings) => check_any(postings, tokens, point_id),
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                check_any(postings, tokens, point_id)
            }
        }
    }

    /// Iterate over point ids whose documents contain all given tokens in the same order they are provided
    pub fn filter_has_phrase(&self, phrase: Document) -> OperationResult<Vec<PointOffsetType>> {
        // in case of mmap immutable index, deleted points are still in the postings
        let is_active = move |idx| self.is_active(idx);

        match &self.storage.postings {
            MmapPostingsEnum::WithPositions(postings) => {
                // Deduplicate phrase tokens: repeated tokens (e.g. "zn zn") must
                // not fetch the same posting list twice, otherwise positions get
                // added twice in `phrase_in_all_postings`.
                let unique_tokens = phrase.to_token_set();
                let result = postings.with_all_or_none_postings(
                    unique_tokens.tokens(),
                    |selected_postings| {
                        Ok(intersect_compressed_postings_phrase_iterator(
                            phrase,
                            selected_postings,
                            is_active,
                        )
                        .collect())
                    },
                )?;
                // Some token has no posting list -> no matches
                Ok(result.unwrap_or_default())
            }
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                // Deduplicate phrase tokens: repeated tokens (e.g. "zn zn") must
                // not fetch the same posting list twice, otherwise positions get
                // added twice in `phrase_in_all_postings`.
                let unique_tokens = phrase.to_token_set();
                let result = postings.with_all_or_none_postings(
                    unique_tokens.tokens(),
                    |selected_postings| {
                        Ok(intersect_compressed_postings_phrase_iterator(
                            phrase,
                            selected_postings,
                            is_active,
                        )
                        .collect())
                    },
                )?;
                // Some token has no posting list -> no matches
                Ok(result.unwrap_or_default())
            }
            // cannot do phrase matching if there's no positional information
            MmapPostingsEnum::Ids(_postings) => Ok(Vec::new()),
            MmapPostingsEnum::WithWeight(_postings) => Ok(Vec::new()),
        }
    }

    pub fn check_has_phrase(
        &self,
        phrase: &Document,
        point_id: PointOffsetType,
    ) -> OperationResult<bool> {
        // in case of mmap immutable index, deleted points are still in the postings
        if !self.is_active(point_id) {
            return Ok(false);
        }

        match &self.storage.postings {
            MmapPostingsEnum::WithPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                let result = postings.with_all_or_none_postings(
                    unique_tokens.tokens(),
                    |selected_postings| {
                        Ok(check_compressed_postings_phrase(
                            phrase,
                            point_id,
                            selected_postings,
                        ))
                    },
                )?;
                // Some token has no posting list -> no match
                Ok(result.unwrap_or(false))
            }
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                let result = postings.with_all_or_none_postings(
                    unique_tokens.tokens(),
                    |selected_postings| {
                        Ok(check_compressed_postings_phrase(
                            phrase,
                            point_id,
                            selected_postings,
                        ))
                    },
                )?;
                // Some token has no posting list -> no match
                Ok(result.unwrap_or(false))
            }
            // cannot do phrase matching if there's no positional information
            MmapPostingsEnum::Ids(_postings) => Ok(false),
            MmapPostingsEnum::WithWeight(_postings) => Ok(false),
        }
    }

    pub fn files(&self) -> Vec<PathBuf> {
        vec![
            self.path.join(POSTINGS_FILE),
            self.path.join(VOCAB_FILE),
            self.path.join(POINT_TO_TOKENS_COUNT_FILE),
            self.path.join(DELETED_POINTS_FILE),
        ]
    }

    pub fn immutable_files(&self) -> Vec<PathBuf> {
        vec![self.path.join(POSTINGS_FILE), self.path.join(VOCAB_FILE)]
    }

    pub fn flusher(&self) -> Flusher {
        self.storage.deleted_points.flusher()
    }

    pub fn is_on_disk(&self) -> bool {
        self.is_on_disk
    }

    /// Populate all pages in the mmap.
    /// Block until all pages are populated.
    pub fn populate(&self) -> OperationResult<()> {
        self.storage.postings.populate()?;
        self.storage.vocab.populate()?;
        self.storage.point_to_tokens_count.populate()?;
        Ok(())
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> OperationResult<()> {
        let Self {
            path: _,
            storage,
            has_weight: _,
            active_points_count: _,
            is_on_disk: _,
        } = self;
        let Storage {
            postings,
            vocab,
            point_to_tokens_count,
            deleted_points,
        } = storage;
        postings.clear_cache()?;
        vocab.clear_cache()?;
        point_to_tokens_count.clear_cache()?;
        deleted_points.clear_cache()?;
        Ok(())
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

        // Resolve query tokens to (token_id, idf) pairs via the mmap vocab.
        let token_ids_and_idfs: Vec<(TokenId, f32)> = query
            .tokens
            .iter()
            .zip(query.idfs.iter())
            .filter_map(|(token, &idf)| {
                let tid = self
                    .storage
                    .vocab
                    .get(token.as_str())
                    .ok()
                    .flatten()
                    .and_then(<[TokenId]>::first)
                    .copied()?;
                Some((tid, idf))
            })
            .collect();

        if token_ids_and_idfs.is_empty() {
            return Ok(vec![]);
        }

        let token_ids: Vec<TokenId> = token_ids_and_idfs.iter().map(|(tid, _)| *tid).collect();

        /// Generic helper: score prefiltered points using mmap-backed posting
        /// list views. The callback pattern is required by `UniversalPostings`.
        fn score_with_views<V: ZerocopyPostingValue>(
            postings: &UniversalPostings<V, MmapFile>,
            token_ids: &[TokenId],
            idfs: &[f32],
            top: usize,
            ordered_prefiltered_points: &[PointOffsetType],
            extract_weight: fn(&V) -> f32,
        ) -> OperationResult<Vec<ScoredPointOffset>> {
            postings.with_existing_postings(token_ids, |views| {
                // Build a map from token_id -> idf for the views we actually got.
                // `views` may be a subset of requested token_ids if some were missing.
                let mut iterators: Vec<(PostingIterator<'_, V>, f32)> =
                    Vec::with_capacity(views.len());
                for (token_id, view) in views {
                    // Find the idf for this token_id from the original parallel arrays.
                    let idf = token_ids
                        .iter()
                        .zip(idfs.iter())
                        .find(|&(&tid, _)| tid == token_id)
                        .map(|(_, &idf)| idf)
                        .unwrap_or(0.0);
                    iterators.push((view.into_iter(), idf));
                }

                if iterators.is_empty() {
                    return Ok(vec![]);
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

                Ok(top_k.into_vec())
            })
        }

        let idfs: Vec<f32> = token_ids_and_idfs.iter().map(|(_, idf)| *idf).collect();

        match &self.storage.postings {
            MmapPostingsEnum::WithWeight(postings) => score_with_views(
                postings,
                &token_ids,
                &idfs,
                top,
                ordered_prefiltered_points,
                |w| w.token_weight(),
            ),
            MmapPostingsEnum::WithWeightAndPositions(postings) => score_with_views(
                postings,
                &token_ids,
                &idfs,
                top,
                ordered_prefiltered_points,
                |w| w.token_weight(),
            ),
            // ID-only or position-only postings have no weights
            MmapPostingsEnum::Ids(_) | MmapPostingsEnum::WithPositions(_) => Ok(vec![]),
        }
    }

    /// BM25 scoring with filter and pruning for mmap-backed posting lists.
    ///
    /// Follows the sparse vector batched search pipeline: iterates posting
    /// lists in batches, accumulates TF·IDF scores, applies the filter, and
    /// uses `max_next_weight` to prune the longest posting list.
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

        let token_ids_and_idfs: Vec<(TokenId, f32)> = query
            .tokens
            .iter()
            .zip(query.idfs.iter())
            .filter_map(|(token, &idf)| {
                let tid = self
                    .storage
                    .vocab
                    .get(token.as_str())
                    .ok()
                    .flatten()
                    .and_then(<[TokenId]>::first)
                    .copied()?;
                Some((tid, idf))
            })
            .collect();

        if token_ids_and_idfs.is_empty() {
            return Ok(vec![]);
        }

        let token_ids: Vec<TokenId> = token_ids_and_idfs.iter().map(|(tid, _)| *tid).collect();
        let idfs: Vec<f32> = token_ids_and_idfs.iter().map(|(_, idf)| *idf).collect();

        /// Trait to abstract weight extraction for pruning.
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

        fn search_inner<V: ZerocopyPostingValue + HasWeights>(
            postings: &UniversalPostings<V, MmapFile>,
            token_ids: &[TokenId],
            idfs: &[f32],
            top: usize,
            filter: impl Fn(PointOffsetType) -> bool,
        ) -> OperationResult<Vec<ScoredPointOffset>> {
            postings.with_existing_postings(token_ids, |views| {
                struct IndexedIterator<'a, V: PostingValue> {
                    iter: PostingIterator<'a, V>,
                    idf: f32,
                }

                let mut iterators: Vec<IndexedIterator<'_, V>> = Vec::with_capacity(views.len());
                let mut max_id: PointOffsetType = 0;
                let mut min_id = PointOffsetType::MAX;

                for (token_id, view) in views {
                    let idf = token_ids
                        .iter()
                        .zip(idfs.iter())
                        .find(|&(&tid, _)| tid == token_id)
                        .map(|(_, &idf)| idf)
                        .unwrap_or(0.0);

                    if let Some(last_id) = view.get_last_id() {
                        max_id = max_id.max(last_id);
                    }

                    let mut iter = view.into_iter();
                    if let Some(first) = iter.advance_until_greater_or_equal(0) {
                        min_id = min_id.min(first.id);
                    }
                    iterators.push(IndexedIterator { iter, idf });
                }

                if iterators.is_empty() || min_id > max_id {
                    return Ok(vec![]);
                }

                let mut top_k = TopK::new(top);
                const BATCH_SIZE: u32 = 10_000;
                let mut batch_start = min_id;
                let mut best_min_score = f32::MIN;

                loop {
                    if batch_start > max_id {
                        break;
                    }
                    let batch_last = batch_start.saturating_add(BATCH_SIZE).min(max_id);

                    let batch_len = (batch_last - batch_start + 1) as usize;
                    let mut scores = vec![0.0f32; batch_len];

                    for ii in iterators.iter_mut() {
                        loop {
                            let Some(elem) = ii.iter.advance_until_greater_or_equal(batch_start)
                            else {
                                break;
                            };
                            if elem.id > batch_last {
                                break;
                            }
                            let local = (elem.id - batch_start) as usize;
                            scores[local] += elem.value.token_weight() * ii.idf;
                            ii.iter.next();
                        }
                    }

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

                    iterators.retain(|ii| ii.iter.len() > 0);

                    if iterators.is_empty() {
                        break;
                    }

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

                    if top_k.len() >= top {
                        let new_min_score = top_k.threshold();
                        if new_min_score > best_min_score {
                            best_min_score = new_min_score;

                            let longest_idx = iterators
                                .iter()
                                .enumerate()
                                .max_by_key(|(_, ii)| ii.iter.len())
                                .map(|(i, _)| i)
                                .unwrap();
                            if longest_idx != 0 {
                                iterators.swap(0, longest_idx);
                            }

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
                                        let max_weight = elem
                                            .value
                                            .token_weight()
                                            .max(elem.value.max_next_weight());
                                        max_weight * iterators[0].idf <= new_min_score
                                    }
                                    None => {
                                        let max_weight = elem
                                            .value
                                            .token_weight()
                                            .max(elem.value.max_next_weight());
                                        max_weight * iterators[0].idf <= new_min_score
                                    }
                                    _ => false,
                                };

                                if can_prune {
                                    match next_min_in_others {
                                        Some(next_min) => {
                                            iterators[0]
                                                .iter
                                                .advance_until_greater_or_equal(next_min);
                                        }
                                        None => for _ in iterators[0].iter.by_ref() {},
                                    }
                                }
                            }
                        }
                    }

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

                Ok(top_k.into_vec())
            })
        }

        match &self.storage.postings {
            MmapPostingsEnum::WithWeight(postings) => {
                search_inner(postings, &token_ids, &idfs, top, filter)
            }
            MmapPostingsEnum::WithWeightAndPositions(postings) => {
                search_inner(postings, &token_ids, &idfs, top, filter)
            }
            MmapPostingsEnum::Ids(_) | MmapPostingsEnum::WithPositions(_) => Ok(vec![]),
        }
    }
}

impl InvertedIndex for MmapInvertedIndex {
    fn get_vocab_mut(&mut self) -> &mut HashMap<String, TokenId> {
        unreachable!("MmapInvertedIndex does not support mutable operations")
    }

    fn index_tokens(
        &mut self,
        _idx: PointOffsetType,
        _tokens: super::TokenSet,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to mmap immutable text index",
        ))
    }

    fn index_token_weight_map(
        &mut self,
        _idx: PointOffsetType,
        _token_weight_map: super::TokenWeightMap,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to mmap immutable text index",
        ))
    }

    fn index_document(
        &mut self,
        _idx: PointOffsetType,
        _document: Document,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        Err(OperationError::service_error(
            "Can't add values to mmap immutable text index",
        ))
    }

    fn remove(&mut self, idx: PointOffsetType) -> bool {
        let Some(is_deleted) = self.storage.deleted_points.get(idx as usize) else {
            return false; // Never existed
        };

        if is_deleted {
            return false; // Already removed
        }

        self.storage.deleted_points.set(idx as usize, true);
        if let Some(count) = self.storage.point_to_tokens_count.get_mut(idx as usize) {
            *count = 0;

            // `deleted_points`'s length can be larger than `point_to_tokens_count`'s length.
            // Only if the index is within bounds of `point_to_tokens_count`, we decrement the active points count.
            self.active_points_count -= 1;
        }

        true
    }

    fn filter<'a>(
        &'a self,
        query: ParsedQuery,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + 'a>> {
        let ids = match query {
            ParsedQuery::AllTokens(tokens) => self.filter_has_all(tokens)?,
            ParsedQuery::Phrase(phrase) => self.filter_has_phrase(phrase)?,
            ParsedQuery::AnyTokens(tokens) => self.filter_has_any(tokens)?,
        };
        Ok(Box::new(ids.into_iter()))
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Option<usize>> {
        self.storage.postings.posting_len(token_id)
    }

    fn vocab_with_postings_len_iter(
        &self,
    ) -> impl Iterator<Item = OperationResult<(&str, usize)>> + '_ {
        self.iter_vocab().filter_map(move |(token, &token_id)| {
            // Surface read errors as iterator items; drop tokens with no
            // posting list silently (same as the in-memory variants).
            match self.storage.postings.posting_len(token_id) {
                Ok(Some(posting_len)) => Some(Ok((token, posting_len))),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
    }

    fn check_match(
        &self,
        parsed_query: &ParsedQuery,
        point_id: PointOffsetType,
    ) -> OperationResult<bool> {
        match parsed_query {
            ParsedQuery::AllTokens(tokens) => self.check_has_subset(tokens, point_id),
            ParsedQuery::Phrase(phrase) => self.check_has_phrase(phrase, point_id),
            ParsedQuery::AnyTokens(tokens) => self.check_has_any(tokens, point_id),
        }
    }

    fn values_is_empty(&self, point_id: PointOffsetType) -> bool {
        if self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true)
        {
            return true;
        }
        self.storage
            .point_to_tokens_count
            .get(point_id as usize)
            .map(|count| *count == 0)
            // if the point does not exist, it is considered empty
            .unwrap_or(true)
    }

    fn values_count(&self, point_id: PointOffsetType) -> usize {
        if self
            .storage
            .deleted_points
            .get(point_id as usize)
            .unwrap_or(true)
        {
            return 0;
        }

        self.storage
            .point_to_tokens_count
            .get(point_id as usize)
            .copied()
            // if the point does not exist, it is considered empty
            .unwrap_or(0)
    }

    fn points_count(&self) -> usize {
        self.active_points_count
    }

    fn get_token_id(&self, token: &str, hw_counter: &HardwareCounterCell) -> Option<TokenId> {
        if self.is_on_disk {
            hw_counter.payload_index_io_read_counter().incr_delta(
                READ_ENTRY_OVERHEAD + size_of::<TokenId>(), // Avoid check overhead and assume token is always read
            );
        }

        self.storage
            .vocab
            .get(token)
            .ok()
            .flatten()
            .and_then(<[TokenId]>::first)
            .copied()
    }
}
