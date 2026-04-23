use common::types::PointOffsetType;

use super::InvertedIndex;
use super::mutable_inverted_index::MutableInvertedIndex;
use crate::index::field_index::full_text_index::inverted_index::posting_list::PostingList;
use crate::index::field_index::full_text_index::inverted_index::{
    Document, TokenSet, TokenWeightMap,
};

pub struct MutableInvertedIndexBuilder {
    index: MutableInvertedIndex,
    keep_document: bool,
}

impl MutableInvertedIndexBuilder {
    pub fn new(phrase_matching: bool, enable_score: bool) -> Self {
        // Temporarily save doc info if enable_score is true.
        // If phrase_matching is false, we will release the doc info after the build process over
        let index = MutableInvertedIndex::new(phrase_matching, enable_score);
        Self {
            index,
            keep_document: phrase_matching,
        }
    }

    /// Add a vector to the inverted index builder
    pub fn add(&mut self, idx: PointOffsetType, str_tokens: impl IntoIterator<Item = String>) {
        self.index.points_count += 1;

        // resize point_to_* structures if needed
        if self.index.point_to_tokens.len() <= idx as usize {
            self.index
                .point_to_tokens
                .resize_with(idx as usize + 1, Default::default);

            if let Some(point_to_doc) = self.index.point_to_doc.as_mut() {
                point_to_doc.resize_with(idx as usize + 1, Default::default);
            }
        }

        let tokens = self.index.register_tokens(str_tokens);

        // insert as whole document
        if let Some(point_to_doc) = self.index.point_to_doc.as_mut() {
            point_to_doc[idx as usize] = Some(Document::new(tokens.clone()));
        }

        // insert as tokenset
        let tokens_set = TokenSet::from_iter(tokens);
        self.index.point_to_tokens[idx as usize] = Some(tokens_set);
    }

    /// Consumes the builder and returns a MutableInvertedIndex
    pub fn build(mut self) -> MutableInvertedIndex {
        // If enable_score is true, we will use point_to_doc instead of point_to_tokens
        // Because we need to calculate the weight info
        if self.index.has_weight
            && let Some(point_to_doc) = self.index.point_to_doc.as_ref()
        {
            for (idx, tokens) in point_to_doc.iter().enumerate() {
                if let Some(tokens) = tokens {
                    let token_weight_map = TokenWeightMap::from_tokens(tokens.tokens());
                    for (token_idx, token_weight) in token_weight_map.token_weight_iter() {
                        if self.index.postings.len() <= *token_idx as usize {
                            self.index
                                .postings
                                .resize_with(*token_idx as usize + 1, || PostingList::WithWeight {
                                    list: Default::default(),
                                });
                        }
                        self.index
                            .postings
                            .get_mut(*token_idx as usize)
                            .expect("posting must exist")
                            .insert(idx as PointOffsetType, Some(*token_weight));
                    }
                }
            }
            // If phrase_matching is false, we will not preserve the doc info
            if !self.keep_document {
                self.index.point_to_doc = None;
            }
        } else {
            // build postings from point_to_tokens
            // build in order to increase point id
            for (idx, tokenset) in self.index.point_to_tokens.iter().enumerate() {
                if let Some(tokenset) = tokenset {
                    for token_idx in tokenset.tokens() {
                        if self.index.postings.len() <= *token_idx as usize {
                            self.index
                                .postings
                                .resize_with(*token_idx as usize + 1, Default::default);
                        }
                        self.index
                            .postings
                            .get_mut(*token_idx as usize)
                            .expect("posting must exist")
                            .insert(idx as PointOffsetType, None);
                    }
                }
            }
        }

        self.index
    }
}
