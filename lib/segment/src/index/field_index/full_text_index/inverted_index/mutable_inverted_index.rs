use std::collections::HashMap;

use common::counter::hardware_counter::HardwareCounterCell;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};
use itertools::Either;

use super::posting_list::{PostingList, PostingWeightCursor};
use super::postings_iterator::{intersect_postings_iterator, merge_postings_iterator};
use super::{Document, InvertedIndex, ParsedQuery, TokenId, TokenSet};
use crate::common::operation_error::OperationResult;
use crate::index::field_index::full_text_index::inverted_index::TokenWeightMap;
use crate::types::TokenWeightSet;

#[cfg_attr(test, derive(Clone))]
pub struct MutableInvertedIndex {
    pub(super) postings: Vec<PostingList>,
    pub vocab: HashMap<String, TokenId>,
    pub(super) point_to_tokens: Vec<Option<TokenSet>>,

    /// This option decide if the current inverted index store weight info
    /// If true, the posting will store the weight info
    pub has_weight: bool,
    /// Optional additional structure to store positional information of tokens in the documents.
    ///
    /// Must be enabled explicitly.
    pub point_to_doc: Option<Vec<Option<Document>>>,
    pub(super) points_count: usize,
}

impl MutableInvertedIndex {
    /// Create a new inverted index with or without positional information.
    pub fn new(with_positions: bool, with_weight: bool) -> Self {
        Self {
            postings: Vec::new(),
            vocab: HashMap::new(),
            point_to_tokens: Vec::new(),
            has_weight: with_weight,
            point_to_doc: with_positions.then_some(Vec::new()),
            points_count: 0,
        }
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

    pub fn search_text_index_plain(
        &self,
        query: &TokenWeightSet,
        top: usize,
        ordered_prefilterd_points: &[PointOffsetType],
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if !self.has_weight {
            return Ok(vec![]);
        }

        // Collect (cursor, idf) pairs for each query token that exists in the vocab.
        // Each cursor maintains its position so that advancing through sorted point IDs
        // only searches the remaining (unconsumed) portion of the posting list.
        let mut cursors: Vec<(PostingWeightCursor<'_>, f32)> = query
            .tokens
            .iter()
            .zip(query.idfs.iter())
            .filter_map(|(token, &idf)| {
                let tid = *self.vocab.get(token.as_str())?;
                let cursor = self.postings.get(tid as usize)?.weight_cursor()?;
                Some((cursor, idf))
            })
            .collect();

        if cursors.is_empty() {
            return Ok(vec![]);
        }

        let mut top_k = TopK::new(top);

        // Both `ordered_prefilterd_points` and posting lists are sorted by point ID.
        // We advance each cursor forward in tandem, so every posting element is visited
        // at most once across the entire loop.
        for &point_id in ordered_prefilterd_points {
            let mut score = 0.0f32;
            for (cursor, idf) in cursors.iter_mut() {
                if let Some(tf) = cursor.skip_to(point_id) {
                    score += tf * (*idf);
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
    }

    /// BM25 scoring with filter and pruning, following the sparse vector
    /// batched search pipeline.
    ///
    /// Iterates all posting lists in batches of contiguous point IDs,
    /// accumulates TF·IDF scores per point, applies the filter, and uses
    /// `max_next_weight` to prune the longest posting list when the top-k
    /// heap is full.
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

        struct IndexedCursor<'a> {
            cursor: PostingWeightCursor<'a>,
            idf: f32,
        }

        let mut cursors: Vec<IndexedCursor<'_>> = query
            .tokens
            .iter()
            .zip(query.idfs.iter())
            .filter_map(|(token, &idf)| {
                let tid = *self.vocab.get(token.as_str())?;
                let cursor = self.postings.get(tid as usize)?.weight_cursor()?;
                Some(IndexedCursor { cursor, idf })
            })
            .collect();

        if cursors.is_empty() {
            return Ok(vec![]);
        }

        let mut top_k = TopK::new(top);

        // Find global min / max point IDs across all posting lists.
        let mut min_id = PointOffsetType::MAX;
        let mut max_id: PointOffsetType = 0;
        for ic in &cursors {
            if let Some(first) = ic.cursor.peek() {
                min_id = min_id.min(first.point_id());
            }
            if let Some(last_id) = ic.cursor.last_point_id() {
                max_id = max_id.max(last_id);
            }
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
            // Reusable score buffer indexed by (point_id - batch_start).
            let mut scores = vec![0.0f32; batch_len];

            for ic in cursors.iter_mut() {
                ic.cursor.for_each_till_id(batch_last, |id, weight| {
                    let local = (id - batch_start) as usize;
                    scores[local] += weight * ic.idf;
                });
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

            // ── remove exhausted cursors ──
            cursors.retain(|ic| ic.cursor.len_to_end() > 0);

            if cursors.is_empty() {
                break;
            }

            // Fast-path: single cursor left
            if cursors.len() == 1 {
                let ic = &mut cursors[0];
                ic.cursor
                    .for_each_till_id(PointOffsetType::MAX, |id, weight| {
                        if filter(id) {
                            let score = weight * ic.idf;
                            top_k.push(ScoredPointOffset { idx: id, score });
                        }
                    });
                break;
            }

            // ── pruning with max_next_weight ──
            if top_k.len() >= top {
                let new_min_score = top_k.threshold();
                if new_min_score > best_min_score {
                    best_min_score = new_min_score;

                    // Promote longest cursor to front
                    let longest_idx = cursors
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, ic)| ic.cursor.len_to_end())
                        .map(|(i, _)| i)
                        .unwrap();
                    if longest_idx != 0 {
                        cursors.swap(0, longest_idx);
                    }

                    // Try pruning the longest cursor
                    if let Some(elem) = cursors[0].cursor.peek() {
                        let next_min_in_others = cursors[1..]
                            .iter()
                            .filter_map(|ic| ic.cursor.peek().map(|e| e.point_id()))
                            .min();

                        let can_prune = match next_min_in_others {
                            Some(next_min) if next_min > elem.point_id() => {
                                let max_weight = elem.weight().max(elem.max_next_weight());
                                max_weight * cursors[0].idf <= new_min_score
                            }
                            None => {
                                let max_weight = elem.weight().max(elem.max_next_weight());
                                max_weight * cursors[0].idf <= new_min_score
                            }
                            _ => false,
                        };

                        if can_prune {
                            match next_min_in_others {
                                Some(next_min) => {
                                    cursors[0].cursor.skip_to(next_min);
                                }
                                None => {
                                    cursors[0].cursor.skip_to_end();
                                }
                            }
                        }
                    }
                }
            }

            // Update batch_start from remaining cursors
            batch_start = cursors
                .iter()
                .filter_map(|ic| ic.cursor.peek().map(|e| e.point_id()))
                .min()
                .unwrap_or(max_id + 1);
        }

        Ok(top_k.into_vec())
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
                .insert(point_id, None);
        }
        self.point_to_tokens[point_id as usize] = Some(tokens);

        Ok(())
    }

    fn index_token_weight_map(
        &mut self,
        point_id: PointOffsetType,
        token_weight_map: TokenWeightMap,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
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

        for (token_id, token_weight) in token_weight_map.token_weight_iter() {
            let token_idx_usize = *token_id as usize;

            if self.postings.len() <= token_idx_usize {
                let new_len = token_idx_usize + 1;
                hw_cell_wb.incr_delta((new_len - self.postings.len()) * size_of::<PostingList>());
                self.postings
                    .resize_with(new_len, || PostingList::WithWeight {
                        list: Default::default(),
                    });
            }

            hw_cell_wb.incr_delta(size_of_val(&point_id));
            self.postings
                .get_mut(token_idx_usize)
                .expect("posting must exist")
                .insert(point_id, Some(*token_weight));
        }
        self.point_to_tokens[point_id as usize] = Some(token_weight_map.tokens_set());
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

    fn vocab_with_postings_len_iter(
        &self,
    ) -> impl Iterator<Item = OperationResult<(&str, usize)>> + '_ {
        self.vocab.iter().filter_map(|(token, &posting_idx)| {
            self.postings
                .get(posting_idx as usize)
                .map(|postings| Ok((token.as_str(), postings.len())))
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

    fn get_token_id(&self, token: &str, _hw_counter: &HardwareCounterCell) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

impl MutableInvertedIndex {
    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        let Self {
            postings,
            vocab,
            has_weight: _,
            point_to_tokens,
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
        postings_bytes + vocab_base_bytes + vocab_heap_bytes + ptt_bytes + ptd_bytes
    }
}
