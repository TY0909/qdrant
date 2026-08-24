use std::collections::HashMap;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use common::universal_io::UserData;
use itertools::Either;

use super::posting_list::{FrequencyPostingElement, PostingList};
use super::postings_iterator::{intersect_postings_iterator, merge_postings_iterator};
use super::{
    Bm25State, Document, InvertedIndex, MutableBm25State, ParsedQuery, TokenFrequencyMap, TokenId,
    TokenSet,
};
use crate::common::operation_error::{OperationError, OperationResult};

#[cfg_attr(test, derive(Clone))]
pub struct MutableInvertedIndex {
    pub(super) postings: Vec<PostingList>,
    pub vocab: HashMap<String, TokenId>,
    pub(super) point_to_tokens: Vec<Option<TokenSet>>,
    pub(super) bm25: Option<MutableBm25State>,

    /// Optional additional structure to store positional information of tokens in the documents.
    ///
    /// Must be enabled explicitly.
    pub point_to_doc: Option<Vec<Option<Document>>>,
    pub(super) points_count: usize,
}

impl MutableInvertedIndex {
    /// Create a new inverted index with or without positional information.
    pub fn new(with_positions: bool, with_frequencies: bool) -> Self {
        Self {
            postings: Vec::new(),
            vocab: HashMap::new(),
            point_to_tokens: Vec::new(),
            bm25: with_frequencies.then(Bm25State::default),
            point_to_doc: with_positions.then_some(Vec::new()),
            points_count: 0,
        }
    }

    pub(in crate::index::field_index::full_text_index) fn has_frequencies(&self) -> bool {
        self.bm25.is_some()
    }

    fn get_tokens(&self, idx: PointOffsetType) -> Option<&TokenSet> {
        self.point_to_tokens.get(idx as usize)?.as_ref()
    }

    fn get_document(&self, idx: PointOffsetType) -> Option<&Document> {
        self.point_to_doc.as_ref()?.get(idx as usize)?.as_ref()
    }

    /// Iterate over point ids whose documents contain all given tokens
    fn filter_has_all(&self, tokens: TokenSet) -> impl Iterator<Item = PointOffsetType> + '_ {
        let postings_opt: Option<Vec<_>> = tokens
            .tokens()
            .iter()
            .map(|&token_id| {
                // if a ParsedQuery token was given an index, then it must exist in the vocabulary
                // dictionary. Posting list entry can be None but it exists.

                self.postings.get(token_id as usize)
            })
            .collect();

        let Some(postings) = postings_opt else {
            // There are unseen tokens -> no matches
            return Either::Left(std::iter::empty());
        };
        if postings.is_empty() {
            // Empty request -> no matches
            return Either::Left(std::iter::empty());
        }

        Either::Right(intersect_postings_iterator(postings))
    }

    fn filter_has_any(&self, tokens: TokenSet) -> impl Iterator<Item = PointOffsetType> + '_ {
        let postings_opt: Vec<_> = tokens
            .tokens()
            .iter()
            .filter_map(|&token_id| {
                // if a ParsedQuery token was given an index, then it must exist in the vocabulary
                // dictionary. Posting list entry can be None but it exists.
                self.postings.get(token_id as usize)
            })
            .collect();

        if postings_opt.is_empty() {
            // Empty request -> no matches
            return Either::Left(std::iter::empty());
        }

        Either::Right(merge_postings_iterator(postings_opt))
    }

    pub fn filter_has_phrase(
        &self,
        phrase: Document,
    ) -> Box<dyn Iterator<Item = PointOffsetType> + '_> {
        let Some(point_to_doc) = self.point_to_doc.as_ref() else {
            // Return empty iterator when not enabled
            return Box::new(std::iter::empty());
        };

        let iter = self
            .filter_has_all(phrase.to_token_set())
            .filter(move |id| {
                let doc = point_to_doc[*id as usize]
                    .as_ref()
                    .expect("if it passed the intersection filter, it must exist");

                doc.has_phrase(&phrase)
            });

        Box::new(iter)
    }

    pub(in crate::index::field_index::full_text_index) fn index_token_frequencies(
        &mut self,
        point_id: PointOffsetType,
        token_frequencies: TokenFrequencyMap,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        let Some(bm25) = self.bm25.as_mut() else {
            return Err(OperationError::service_error(
                "cannot add term frequencies to an ID-only inverted index",
            ));
        };

        self.points_count += 1;
        let mut hw_cell_wb = hw_counter
            .payload_index_io_write_counter()
            .write_back_counter();

        if self.point_to_tokens.len() <= point_id as usize {
            let new_len = point_id as usize + 1;
            hw_cell_wb
                .incr_delta((new_len - self.point_to_tokens.len()) * size_of::<Option<TokenSet>>());
            self.point_to_tokens.resize_with(new_len, Default::default);

            let document_lengths = &mut bm25.document_lengths;
            hw_cell_wb.incr_delta((new_len - document_lengths.len()) * size_of::<u32>());
            document_lengths.resize(new_len, 0);
        }

        for (token_id, term_frequency) in token_frequencies.iter() {
            let token_idx = token_id as usize;
            if self.postings.len() <= token_idx {
                let new_len = token_idx + 1;
                hw_cell_wb.incr_delta((new_len - self.postings.len()) * size_of::<PostingList>());
                self.postings
                    .resize_with(new_len, || PostingList::new(true));
            }

            hw_cell_wb.incr_delta(size_of::<FrequencyPostingElement>());
            self.postings[token_idx].insert_frequency(point_id, term_frequency);
        }

        let document_length = token_frequencies.document_length();
        bm25.document_lengths[point_id as usize] = document_length;
        bm25.stats.add_document(document_length);
        self.point_to_tokens[point_id as usize] = Some(token_frequencies.tokens_set());
        Ok(())
    }
}

impl InvertedIndex for MutableInvertedIndex {
    fn get_vocab_mut(&mut self) -> &mut HashMap<String, TokenId> {
        &mut self.vocab
    }

    fn index_tokens(
        &mut self,
        point_id: PointOffsetType,
        tokens: TokenSet,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        if self.has_frequencies() {
            return Err(OperationError::service_error(
                "cannot add ID-only tokens to a frequency inverted index",
            ));
        }
        self.points_count += 1;

        let mut hw_cell_wb = hw_counter
            .payload_index_io_write_counter()
            .write_back_counter();

        if self.point_to_tokens.len() <= point_id as usize {
            let new_len = point_id as usize + 1;

            // Only measure the overhead of `TokenSet` here since we account for the tokens a few lines below.
            hw_cell_wb
                .incr_delta((new_len - self.point_to_tokens.len()) * size_of::<Option<TokenSet>>());

            self.point_to_tokens.resize_with(new_len, Default::default);
        }

        for token_id in tokens.tokens() {
            let token_idx_usize = *token_id as usize;

            if self.postings.len() <= token_idx_usize {
                let new_len = token_idx_usize + 1;
                hw_cell_wb.incr_delta((new_len - self.postings.len()) * size_of::<PostingList>());
                self.postings.resize_with(new_len, Default::default);
            }

            hw_cell_wb.incr_delta(size_of_val(&point_id));
            self.postings
                .get_mut(token_idx_usize)
                .expect("posting must exist")
                .insert(point_id);
        }
        self.point_to_tokens[point_id as usize] = Some(tokens);

        Ok(())
    }

    fn index_document(
        &mut self,
        point_id: PointOffsetType,
        ordered_document: Document,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        let Some(point_to_doc) = &mut self.point_to_doc else {
            // Phrase matching is not enabled
            return Ok(());
        };

        let mut hw_cell_wb = hw_counter
            .payload_index_io_write_counter()
            .write_back_counter();

        // Ensure container has enough capacity
        if point_id as usize >= point_to_doc.len() {
            let new_len = point_id as usize + 1;

            hw_cell_wb.incr_delta((new_len - point_to_doc.len()) * size_of::<Option<Document>>());

            point_to_doc.resize_with(new_len, Default::default);
        }

        // Store the ordered document
        point_to_doc[point_id as usize] = Some(ordered_document);

        Ok(())
    }

    fn remove(&mut self, point_id: PointOffsetType) -> bool {
        if point_id as usize >= self.point_to_tokens.len() {
            return false; // Already removed or never actually existed
        }

        let Some(removed_token_set) = self.point_to_tokens[point_id as usize].take() else {
            return false;
        };

        if let Some(point_to_doc) = &mut self.point_to_doc {
            point_to_doc[point_id as usize] = None;
        }

        if let Some(bm25) = &mut self.bm25 {
            let document_length = std::mem::take(&mut bm25.document_lengths[point_id as usize]);
            bm25.stats.remove_document(document_length);
        }

        self.points_count -= 1;

        for removed_token in removed_token_set.tokens() {
            // unwrap safety: posting list exists and contains the point idx
            let posting = self.postings.get_mut(*removed_token as usize).unwrap();
            posting.remove(point_id);
        }

        true
    }

    fn filter(
        &self,
        query: ParsedQuery,
        _hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Box<dyn Iterator<Item = PointOffsetType> + '_>> {
        match query {
            ParsedQuery::AllTokens(tokens) => Ok(Box::new(self.filter_has_all(tokens))),
            ParsedQuery::Phrase(phrase) => Ok(Box::new(self.filter_has_phrase(phrase))),
            ParsedQuery::AnyTokens(tokens) => Ok(Box::new(self.filter_has_any(tokens))),
        }
    }

    fn get_posting_len(
        &self,
        token_id: TokenId,
        _: &HardwareCounterCell,
    ) -> OperationResult<Option<usize>> {
        Ok(self.postings.get(token_id as usize).map(|x| x.len()))
    }

    fn for_each_vocab_with_postings_len(
        &self,
        mut f: impl FnMut(&str, usize) -> OperationResult<()>,
    ) -> OperationResult<()> {
        self.vocab.iter().try_for_each(|(token, &posting_idx)| {
            if let Some(postings) = self.postings.get(posting_idx as usize) {
                f(token.as_str(), postings.len())?;
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
            ParsedQuery::AllTokens(query) => {
                let Some(doc) = self.get_tokens(point_id) else {
                    return Ok(false);
                };

                // Check that all tokens are in document
                doc.has_subset(query)
            }
            ParsedQuery::Phrase(document) => {
                let Some(doc) = self.get_document(point_id) else {
                    return Ok(false);
                };

                // Check that all tokens are in document, in order
                doc.has_phrase(document)
            }
            ParsedQuery::AnyTokens(query) => {
                let Some(doc) = self.get_tokens(point_id) else {
                    return Ok(false);
                };

                // Check that at least one token is in document
                doc.has_any(query)
            }
        };
        Ok(matched)
    }

    fn values_is_empty(&self, point_id: PointOffsetType) -> bool {
        self.get_tokens(point_id).is_none_or(|x| x.is_empty())
    }

    fn values_count(&self, point_id: PointOffsetType) -> usize {
        // Maybe we want number of documents in the future?
        self.get_tokens(point_id).map(|x| x.len()).unwrap_or(0)
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

impl MutableInvertedIndex {
    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        let Self {
            postings,
            vocab,
            point_to_tokens,
            bm25,
            point_to_doc,
            points_count: _,
        } = self;

        let postings_bytes: usize = postings.capacity() * std::mem::size_of::<PostingList>()
            + postings.iter().map(|p| p.heap_bytes()).sum::<usize>();
        let hashmap_entry_overhead = std::mem::size_of::<u64>() + std::mem::size_of::<usize>();
        let vocab_base_bytes = vocab.capacity()
            * (std::mem::size_of::<String>()
                + std::mem::size_of::<TokenId>()
                + hashmap_entry_overhead);
        // String heap data
        let vocab_heap_bytes: usize = vocab.keys().map(|s| s.capacity()).sum();
        // TokenSet wraps Vec<TokenId> — account for heap allocation
        let ptt_bytes: usize = point_to_tokens.capacity() * std::mem::size_of::<Option<TokenSet>>()
            + point_to_tokens
                .iter()
                .filter_map(|opt| opt.as_ref())
                .map(|ts| ts.heap_bytes())
                .sum::<usize>();
        // Document wraps Vec<TokenId> — account for heap allocation
        let ptd_bytes: usize = point_to_doc
            .as_ref()
            .map(|v| {
                v.capacity() * std::mem::size_of::<Option<Document>>()
                    + v.iter()
                        .filter_map(|opt| opt.as_ref())
                        .map(|doc| doc.heap_bytes())
                        .sum::<usize>()
            })
            .unwrap_or(0);
        let document_lengths_bytes = bm25.as_ref().map_or(0, |bm25| {
            bm25.document_lengths.capacity() * size_of::<u32>()
        });
        postings_bytes
            + vocab_base_bytes
            + vocab_heap_bytes
            + ptt_bytes
            + ptd_bytes
            + document_lengths_bytes
    }
}
