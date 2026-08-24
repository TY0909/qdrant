use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicBool;

use ahash::AHashMap;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::UserData;
use itertools::Either;
use posting_list::{PostingBuilder, PostingList, PostingListView, PostingValue};

use super::immutable_postings_enum::ImmutablePostings;
use super::length_norm::EncodedDocumentLength;
use super::mutable_inverted_index::MutableInvertedIndex;
use super::on_disk_inverted_index::OnDiskInvertedIndex;
use super::on_disk_inverted_index::on_disk_postings_enum::OnDiskPostingsEnum;
use super::positions::Positions;
use super::postings_iterator::{
    intersect_compressed_postings_iterator, merge_compressed_postings_iterator,
};
use super::scoring::{
    Bm25SearchContext, Bm25SearchOptions, CompressedBm25PostingListIter,
    InMemoryEncodedDocumentLengths, IndexedBm25Posting,
};
use super::term_frequency::TermFrequency;
use super::term_frequency_and_positions::TermFrequencyAndPositions;
use super::{
    Bm25Params, Document, InvertedIndex, InvertedIndexScoring, ParsedQuery,
    TermFrequencyPostingValue, TokenId, TokenSet, bm25_scoring_not_enabled_error,
};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::full_text_index_scoring::FullTextSearchScratchPool;
use crate::index::field_index::full_text_index::inverted_index::postings_iterator::{
    check_compressed_postings_phrase, intersect_compressed_postings_phrase_iterator,
};
use crate::types::QueryTokenWeightSet;

fn scoring_postings<'a, V: TermFrequencyPostingValue>(
    postings: &'a [PostingList<V>],
    vocab: &HashMap<String, TokenId>,
    query: &QueryTokenWeightSet,
) -> Vec<IndexedBm25Posting<CompressedBm25PostingListIter<'a, V>>> {
    let mut resolved = query
        .query_tokens()
        .iter()
        .filter_map(|query_token| {
            let token_id = *vocab.get(query_token.token())?;
            let posting = postings.get(token_id as usize)?;
            let view = posting.view();
            let last_id = view.components().last_id.map(|id| id.get());
            Some((token_id, view, last_id, query_token.idf()))
        })
        .collect::<Vec<_>>();
    resolved.sort_unstable_by_key(|(token_id, _, _, _)| *token_id);
    resolved
        .into_iter()
        .map(|(_, view, last_id, idf)| {
            IndexedBm25Posting::new(
                CompressedBm25PostingListIter::new(view.into_iter(), last_id),
                idf,
            )
        })
        .collect()
}

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
    pub(in crate::index::field_index::full_text_index) point_to_tokens_count: Vec<usize>,
    pub(in crate::index::field_index::full_text_index) points_count: usize,
    pub(in crate::index::field_index::full_text_index) bm25: Option<super::ImmutableBm25State>,
    pub(super) search_scratch_pool: FullTextSearchScratchPool,
}

impl ImmutableInvertedIndex {
    /// Iterate over point ids whose documents contain all given tokens
    fn filter_has_all<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
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
        ) -> impl Iterator<Item = PointOffsetType> + 'a {
            let postings_opt: Option<Vec<_>> = tokens
                .tokens()
                .iter()
                .map(|&token_id| postings.get(token_id as usize).map(PostingList::view))
                .collect();

            // All tokens must have postings
            let Some(postings) = postings_opt else {
                return Either::Left(std::iter::empty());
            };

            // Query must not be empty
            if postings.is_empty() {
                return Either::Left(std::iter::empty());
            };

            Either::Right(intersect_compressed_postings_iterator(postings, filter))
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => Box::new(intersection(postings, tokens, filter)),
            ImmutablePostings::WithPositions(postings) => {
                Box::new(intersection(postings, tokens, filter))
            }
            ImmutablePostings::WithFrequencies(postings) => {
                Box::new(intersection(postings, tokens, filter))
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                Box::new(intersection(postings, tokens, filter))
            }
        }
    }

    /// Iterate over point ids whose documents contain at least one of the given tokens
    fn filter_has_any<'a>(
        &'a self,
        tokens: TokenSet,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
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
        ) -> impl Iterator<Item = PointOffsetType> + 'a {
            let postings: Vec<_> = tokens
                .tokens()
                .iter()
                .filter_map(|&token_id| postings.get(token_id as usize).map(PostingList::view))
                .collect();

            // Query must not be empty
            if postings.is_empty() {
                return Either::Left(std::iter::empty());
            };

            Either::Right(merge_compressed_postings_iterator(postings, is_active))
        }

        match &self.postings {
            ImmutablePostings::Ids(postings) => Box::new(merge(postings, tokens, is_active)),
            ImmutablePostings::WithPositions(postings) => {
                Box::new(merge(postings, tokens, is_active))
            }
            ImmutablePostings::WithFrequencies(postings) => {
                Box::new(merge(postings, tokens, is_active))
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                Box::new(merge(postings, tokens, is_active))
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
            ImmutablePostings::WithFrequencies(postings) => {
                check_intersection(postings, tokens, point_id)
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
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
            ImmutablePostings::WithFrequencies(postings) => check_any(postings, tokens, point_id),
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                check_any(postings, tokens, point_id)
            }
        }
    }

    /// Iterate over point ids whose documents contain all given tokens in the same order they are provided
    pub fn filter_has_phrase<'a>(
        &'a self,
        phrase: Document,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
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
                    Box::new(intersect_compressed_postings_phrase_iterator(
                        phrase,
                        selected_postings,
                        is_active,
                    ))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                if let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens()) {
                    Box::new(intersect_compressed_postings_phrase_iterator(
                        phrase,
                        selected_postings,
                        is_active,
                    ))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            // cannot do phrase matching if there's no positional information
            ImmutablePostings::Ids(_) | ImmutablePostings::WithFrequencies(_) => {
                Box::new(std::iter::empty())
            }
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

                check_compressed_postings_phrase(phrase, point_id, &selected_postings)
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                let unique_tokens = phrase.to_token_set();
                let Some(selected_postings) = get_all_or_none(postings, unique_tokens.tokens())
                else {
                    return false;
                };

                check_compressed_postings_phrase(phrase, point_id, &selected_postings)
            }
            // cannot do phrase matching if there's no positional information
            ImmutablePostings::Ids(_) | ImmutablePostings::WithFrequencies(_) => false,
        }
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
        self.points_count = self.points_count.saturating_sub(1);
        true
    }

    fn filter<'a>(
        &'a self,
        query: ParsedQuery,
        _hw_counter: &'a HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + 'a>> {
        match query {
            ParsedQuery::AllTokens(tokens) => Ok(self.filter_has_all(tokens)),
            ParsedQuery::Phrase(tokens) => Ok(self.filter_has_phrase(tokens)),
            ParsedQuery::AnyTokens(tokens) => Ok(self.filter_has_any(tokens)),
        }
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        _: &HardwareCounterCell,
    ) -> OperationResult<Option<usize>> {
        Ok(self.postings.posting_len(token_id))
    }

    fn for_each_vocab_with_postings_len(
        &self,
        mut f: impl FnMut(&str, usize) -> OperationResult<()>,
    ) -> OperationResult<()> {
        self.vocab.iter().try_for_each(|(token, &token_id)| {
            if let Some(len) = self.postings.posting_len(token_id) {
                f(token.as_str(), len)?;
            }
            Ok(())
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

    fn for_each_token_id<'a, U: UserData>(
        &self,
        tokens: impl Iterator<Item = (U, &'a str)>,
        _: &HardwareCounterCell,
        mut f: impl FnMut(U, Option<TokenId>),
    ) -> OperationResult<()> {
        tokens.for_each(|(user_data, token)| f(user_data, self.vocab.get(token).copied()));
        Ok(())
    }
}

impl InvertedIndexScoring for ImmutableInvertedIndex {
    fn search_text_index_plain(
        &self,
        query: &QueryTokenWeightSet,
        params: Bm25Params,
        top: usize,
        ordered_prefiltered_points: &[PointOffsetType],
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if top == 0 {
            return Ok(Vec::new());
        }
        let Some(bm25) = &self.bm25 else {
            return Err(bm25_scoring_not_enabled_error());
        };
        let options = Bm25SearchOptions {
            params,
            top,
            is_stopped,
        };

        fn search<V: TermFrequencyPostingValue>(
            index: &ImmutableInvertedIndex,
            postings: &[PostingList<V>],
            bm25: &super::ImmutableBm25State,
            query: &QueryTokenWeightSet,
            options: Bm25SearchOptions<'_>,
            ordered_prefiltered_points: &[PointOffsetType],
        ) -> OperationResult<Vec<ScoredPointOffset>> {
            let Some(context) = Bm25SearchContext::new(
                scoring_postings(postings, &index.vocab, query),
                InMemoryEncodedDocumentLengths(&bm25.document_lengths),
                options.params,
                bm25.stats,
                options.top,
                options.is_stopped,
            ) else {
                return Ok(Vec::new());
            };
            context.plain_search(
                &index.search_scratch_pool,
                ordered_prefiltered_points,
                |point_id| !index.values_is_empty(point_id),
            )
        }

        match &self.postings {
            ImmutablePostings::WithFrequencies(postings) => search(
                self,
                postings,
                bm25,
                query,
                options,
                ordered_prefiltered_points,
            ),
            ImmutablePostings::WithFrequenciesAndPositions(postings) => search(
                self,
                postings,
                bm25,
                query,
                options,
                ordered_prefiltered_points,
            ),
            ImmutablePostings::Ids(_) | ImmutablePostings::WithPositions(_) => {
                Err(bm25_scoring_not_enabled_error())
            }
        }
    }

    fn search_text_index<F>(
        &self,
        query: &QueryTokenWeightSet,
        params: Bm25Params,
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
        let Some(bm25) = &self.bm25 else {
            return Err(bm25_scoring_not_enabled_error());
        };
        let options = Bm25SearchOptions {
            params,
            top,
            is_stopped,
        };

        fn search<V: TermFrequencyPostingValue>(
            index: &ImmutableInvertedIndex,
            postings: &[PostingList<V>],
            bm25: &super::ImmutableBm25State,
            query: &QueryTokenWeightSet,
            options: Bm25SearchOptions<'_>,
            filter: impl Fn(PointOffsetType) -> bool,
        ) -> OperationResult<Vec<ScoredPointOffset>> {
            let Some(context) = Bm25SearchContext::new(
                scoring_postings(postings, &index.vocab, query),
                InMemoryEncodedDocumentLengths(&bm25.document_lengths),
                options.params,
                bm25.stats,
                options.top,
                options.is_stopped,
            ) else {
                return Ok(Vec::new());
            };
            context.search(&index.search_scratch_pool, |point_id| {
                !index.values_is_empty(point_id) && filter(point_id)
            })
        }

        match &self.postings {
            ImmutablePostings::WithFrequencies(postings) => {
                search(self, postings, bm25, query, options, filter)
            }
            ImmutablePostings::WithFrequenciesAndPositions(postings) => {
                search(self, postings, bm25, query, options, filter)
            }
            ImmutablePostings::Ids(_) | ImmutablePostings::WithPositions(_) => {
                Err(bm25_scoring_not_enabled_error())
            }
        }
    }
}

impl TryFrom<MutableInvertedIndex> for ImmutableInvertedIndex {
    type Error = OperationError;

    fn try_from(index: MutableInvertedIndex) -> OperationResult<Self> {
        let MutableInvertedIndex {
            postings,
            vocab,
            point_to_tokens,
            bm25,
            point_to_doc,
            points_count,
            search_scratch_pool,
        } = index;

        let with_frequencies = bm25.is_some();

        let (postings, vocab, orig_to_new_token) = optimized_postings_and_vocab(postings, vocab);

        let postings = match (with_frequencies, point_to_doc) {
            (false, None) => ImmutablePostings::Ids(create_compressed_postings(postings)),
            (false, Some(point_to_doc)) => {
                ImmutablePostings::WithPositions(create_compressed_postings_with_positions(
                    postings,
                    point_to_doc,
                    &orig_to_new_token,
                )?)
            }
            (true, None) => ImmutablePostings::WithFrequencies(
                create_compressed_postings_with_frequencies(postings),
            ),
            (true, Some(point_to_doc)) => ImmutablePostings::WithFrequenciesAndPositions(
                create_compressed_postings_with_frequencies_and_positions(
                    postings,
                    point_to_doc,
                    &orig_to_new_token,
                )?,
            ),
        };

        let bm25 = bm25.map(|bm25| {
            let document_lengths = bm25
                .document_lengths
                .into_iter()
                .map(|length| EncodedDocumentLength::new(length).encoded())
                .collect();
            super::Bm25State {
                document_lengths,
                stats: bm25.stats,
            }
        });

        Ok(ImmutableInvertedIndex {
            postings,
            vocab,
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
            bm25,
            search_scratch_pool,
        })
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

fn create_compressed_postings_with_positions(
    postings: Vec<super::posting_list::PostingList>,
    point_to_doc: Vec<Option<Document>>,
    orig_to_new_token: &AHashMap<TokenId, TokenId>,
) -> OperationResult<Vec<PostingList<Positions>>> {
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
                    let Some(&new_token) = orig_to_new_token.get(&token) else {
                        return map;
                    };
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
                .map(|point_id| {
                    let positions = point_to_tokens_positions
                        .get_mut(point_id as usize)
                        .and_then(|positions| positions.remove(&token))
                        .ok_or_else(|| {
                            OperationError::service_error(format!(
                                "missing positions for token {token} at point {point_id}",
                            ))
                        })?;
                    Ok((point_id, positions))
                })
                .collect::<OperationResult<_>>()
        })
        .collect()
}

fn create_compressed_postings_with_frequencies(
    postings: Vec<super::posting_list::PostingList>,
) -> Vec<PostingList<TermFrequency>> {
    postings
        .into_iter()
        .map(|posting| {
            let mut builder = PostingBuilder::new();
            for element in posting.iter_frequencies() {
                builder.add(
                    element.point_id(),
                    TermFrequency::new(element.term_frequency()),
                );
            }
            builder.build()
        })
        .collect()
}

fn create_compressed_postings_with_frequencies_and_positions(
    postings: Vec<super::posting_list::PostingList>,
    point_to_doc: Vec<Option<Document>>,
    orig_to_new_token: &AHashMap<TokenId, TokenId>,
) -> OperationResult<Vec<PostingList<TermFrequencyAndPositions>>> {
    let mut point_to_token_positions: Vec<AHashMap<TokenId, Vec<u32>>> = point_to_doc
        .into_iter()
        .map(|document| {
            let Some(document) = document else {
                return AHashMap::new();
            };

            let mut positions = AHashMap::with_capacity(document.len());
            for (position, token) in (0u32..).zip(document) {
                let Some(&new_token) = orig_to_new_token.get(&token) else {
                    // Array boundary sentinels create a positional gap but are
                    // deliberately excluded from BM25 statistics and postings.
                    continue;
                };
                positions
                    .entry(new_token)
                    .or_insert_with(Vec::new)
                    .push(position);
            }
            positions
        })
        .collect();

    (0u32..)
        .zip(postings)
        .map(|(token, posting)| {
            posting
                .iter_frequencies()
                .map(|element| {
                    let point_id = element.point_id();
                    let positions = point_to_token_positions
                        .get_mut(point_id as usize)
                        .and_then(|positions| positions.remove(&token))
                        .ok_or_else(|| {
                            OperationError::service_error(format!(
                                "missing positions for token {token} at point {point_id}",
                            ))
                        })?;
                    Ok((
                        point_id,
                        TermFrequencyAndPositions::new(element.term_frequency(), positions),
                    ))
                })
                .collect::<OperationResult<_>>()
        })
        .collect()
}

impl<S: common::universal_io::UniversalRead> TryFrom<&OnDiskInvertedIndex<S>>
    for ImmutableInvertedIndex
{
    type Error = OperationError;

    fn try_from(index: &OnDiskInvertedIndex<S>) -> OperationResult<Self> {
        let postings = match &index.storage.postings {
            OnDiskPostingsEnum::Ids(postings) => ImmutablePostings::Ids(postings.all_postings()?),
            OnDiskPostingsEnum::WithPositions(postings) => {
                ImmutablePostings::WithPositions(postings.all_postings()?)
            }
            OnDiskPostingsEnum::WithFrequencies(postings) => {
                ImmutablePostings::WithFrequencies(postings.all_postings()?)
            }
            OnDiskPostingsEnum::WithFrequenciesAndPositions(postings) => {
                ImmutablePostings::WithFrequenciesAndPositions(postings.all_postings()?)
            }
        };

        let mut vocab = HashMap::with_capacity(index.storage.vocab.keys_count());
        index.storage.vocab.for_each_entry(|token_str, token_id| {
            vocab.insert(token_str.to_owned(), token_id[0]);
            OperationResult::Ok(())
        })?;

        debug_assert!(
            postings.len() == vocab.len(),
            "postings and vocab must be the same size",
        );

        // The in-RAM index uses `count == 0` as its deletion marker. The mmap
        // variant tracks deletions in a separate in-memory bitmask and leaves
        // `point_to_tokens_count` untouched on disk, so we apply the bitmask
        // here when materializing the count vector.
        let mut point_to_tokens_count = index
            .storage
            .point_to_tokens_count
            .read_whole()?
            .into_owned();
        for (idx, count) in point_to_tokens_count.iter_mut().enumerate() {
            if !index
                .storage
                .deleted_points
                .is_active(idx as PointOffsetType)
            {
                *count = 0;
            }
        }

        Ok(ImmutableInvertedIndex {
            postings,
            vocab,
            point_to_tokens_count,
            points_count: index.points_count(),
            bm25: index
                .storage
                .bm25
                .as_ref()
                .map(|bm25| -> OperationResult<_> {
                    Ok(super::Bm25State {
                        document_lengths: bm25.document_lengths.read_whole()?.into_owned(),
                        stats: bm25.stats,
                    })
                })
                .transpose()?,
            search_scratch_pool: FullTextSearchScratchPool::new(),
        })
    }
}

impl ImmutableInvertedIndex {
    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        let Self {
            postings,
            vocab,
            point_to_tokens_count,
            points_count: _,
            bm25,
            search_scratch_pool: _,
        } = self;

        let postings_bytes = postings.ram_usage_bytes();
        // HashMap per-slot overhead: hash (u64) + metadata pointer
        let hashmap_entry_overhead = size_of::<u64>() + size_of::<usize>();
        let vocab_base_bytes = vocab.capacity()
            * (size_of::<String>() + size_of::<TokenId>() + hashmap_entry_overhead);
        // Account for actual heap-allocated string data
        let vocab_heap_bytes: usize = vocab.keys().map(|s| s.capacity()).sum();
        let pttc_bytes = point_to_tokens_count.capacity() * size_of::<usize>();
        let document_lengths_bytes = bm25
            .as_ref()
            .map_or(0, |bm25| bm25.document_lengths.capacity() * size_of::<u8>());
        postings_bytes + vocab_base_bytes + vocab_heap_bytes + pttc_bytes + document_lengths_bytes
    }
}
