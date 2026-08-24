use common::types::PointOffsetType;

use super::InvertedIndex;
use super::mutable_inverted_index::MutableInvertedIndex;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::inverted_index::posting_list::PostingList;
use crate::index::field_index::full_text_index::inverted_index::{
    ARRAY_BOUNDARY_SENTINEL, Bm25State, Bm25Stats, Document, TokenFrequencyMap, TokenSet,
};

pub struct MutableInvertedIndexBuilder {
    index: MutableInvertedIndex,
    frequency_documents: Option<Vec<Option<Document>>>,
}

impl MutableInvertedIndexBuilder {
    pub fn new(phrase_matching: bool, with_frequencies: bool) -> Self {
        let index = MutableInvertedIndex::new(phrase_matching, with_frequencies);
        Self {
            index,
            // Phrase indexes already retain the ordered token stream.
            frequency_documents: (with_frequencies && !phrase_matching).then_some(Vec::new()),
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
            if let Some(frequency_documents) = self.frequency_documents.as_mut() {
                frequency_documents.resize_with(idx as usize + 1, Default::default);
            }
        }

        let tokens = self.index.register_tokens(str_tokens);

        // insert as whole document
        if let Some(point_to_doc) = self.index.point_to_doc.as_mut() {
            point_to_doc[idx as usize] = Some(Document::new(tokens.clone()));
        }
        if let Some(frequency_documents) = self.frequency_documents.as_mut() {
            frequency_documents[idx as usize] = Some(Document::new(tokens.clone()));
        }

        // insert as tokenset
        let tokens_set = TokenSet::from_iter(tokens);
        self.index.point_to_tokens[idx as usize] = Some(tokens_set);
    }

    /// Consumes the builder and returns a MutableInvertedIndex
    pub fn build(mut self) -> OperationResult<MutableInvertedIndex> {
        if self.index.has_frequencies() {
            let boundary_token_id = self.index.vocab.get(ARRAY_BOUNDARY_SENTINEL).copied();
            let Some(frequency_documents) = self
                .index
                .point_to_doc
                .as_ref()
                .or(self.frequency_documents.as_ref())
            else {
                return Err(OperationError::service_error(
                    "BM25 index builder must retain token frequencies",
                ));
            };

            let mut document_lengths = vec![0; self.index.point_to_tokens.len()];
            let mut stats = Bm25Stats::default();

            for (idx, document) in frequency_documents.iter().enumerate() {
                let Some(document) = document else {
                    continue;
                };
                let token_frequencies =
                    TokenFrequencyMap::from_tokens(document.tokens(), boundary_token_id);
                for (token_id, term_frequency) in token_frequencies.iter() {
                    let token_idx = token_id as usize;
                    if self.index.postings.len() <= token_idx {
                        self.index
                            .postings
                            .resize_with(token_idx + 1, || PostingList::new(true));
                    }
                    self.index.postings[token_idx]
                        .insert_frequency(idx as PointOffsetType, term_frequency);
                }

                let document_length = token_frequencies.document_length();
                document_lengths[idx] = document_length;
                stats.add_document(document_length);
                self.index.point_to_tokens[idx] = Some(token_frequencies.tokens_set());
            }

            self.index.bm25 = Some(Bm25State {
                document_lengths,
                stats,
            });
            return Ok(self.index);
        }

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
                        .insert(idx as PointOffsetType);
                }
            }
        }

        Ok(self.index)
    }
}
