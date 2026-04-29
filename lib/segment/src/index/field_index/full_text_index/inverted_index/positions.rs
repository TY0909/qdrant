use posting_list::{PostingValue, SizedHandler, SizedValue, UnsizedHandler, UnsizedValue};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::index::field_index::full_text_index::inverted_index::{Document, TokenId};

/// Represents a list of positions of a token in a document.
#[derive(Default, Clone, Debug)]
pub struct Positions(Vec<u32>);

impl Positions {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, position: u32) {
        self.0.push(position);
    }

    pub fn to_token_positions(&self, token_id: TokenId) -> Vec<TokenPosition> {
        self.0
            .iter()
            .map(|pos| TokenPosition {
                token_id,
                position: *pos,
            })
            .collect()
    }
}

impl PostingValue for Positions {
    type Handler = UnsizedHandler<Self>;
}

impl UnsizedValue for Positions {
    fn write_len(&self) -> usize {
        self.0.as_bytes().len()
    }

    fn write_to(&self, dst: &mut [u8]) {
        self.0
            .as_slice()
            .write_to(dst)
            .expect("write_len should provide correct length");
    }

    fn from_bytes(data: &[u8]) -> Self {
        let positions =
            <[u32]>::ref_from_bytes(data).expect("write_len should provide correct length");
        Positions(positions.to_vec())
    }
}

#[derive(Default, Clone, Debug, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct WeightInfo {
    token_weight: zerocopy::little_endian::F32,
    max_next_weight: zerocopy::little_endian::F32,
}

impl WeightInfo {
    pub(super) fn new(token_weight: f32, max_next_weight: f32) -> Self {
        Self {
            token_weight: zerocopy::little_endian::F32::new(token_weight),
            max_next_weight: zerocopy::little_endian::F32::new(max_next_weight),
        }
    }

    pub(super) fn token_weight(&self) -> f32 {
        self.token_weight.get()
    }

    pub(super) fn max_next_weight(&self) -> f32 {
        self.max_next_weight.get()
    }
}

impl PostingValue for WeightInfo {
    type Handler = SizedHandler<Self>;
}

impl SizedValue for WeightInfo {}

#[derive(Default, Clone, Debug)]
pub struct WeightInfoAndPositions {
    token_weight: f32,
    max_next_weight: f32,
    positions: Vec<u32>,
}

impl WeightInfoAndPositions {
    pub(super) fn new(token_weight: f32, max_next_weight: f32, positions: Vec<u32>) -> Self {
        Self {
            token_weight,
            max_next_weight,
            positions,
        }
    }

    pub(super) fn token_weight(&self) -> f32 {
        self.token_weight
    }

    pub(super) fn max_next_weight(&self) -> f32 {
        self.max_next_weight
    }
}

impl PostingValue for WeightInfoAndPositions {
    type Handler = UnsizedHandler<Self>;
}

impl UnsizedValue for WeightInfoAndPositions {
    fn write_len(&self) -> usize {
        self.token_weight.as_bytes().len()
            + self.max_next_weight.as_bytes().len()
            + self.positions.as_slice().as_bytes().len()
    }

    fn write_to(&self, dst: &mut [u8]) {
        let f32_size = std::mem::size_of::<f32>();
        let header_size = 2 * f32_size;

        assert_eq!(
            dst.len(),
            self.write_len(),
            "write_len should provide correct length"
        );

        dst[..f32_size].copy_from_slice(self.token_weight.as_bytes());
        dst[f32_size..header_size].copy_from_slice(self.max_next_weight.as_bytes());
        dst[header_size..].copy_from_slice(self.positions.as_slice().as_bytes());
    }

    fn from_bytes(data: &[u8]) -> Self {
        let f32_size = std::mem::size_of::<f32>();
        let header_size = 2 * f32_size;

        let token_weight = *f32::ref_from_bytes(&data[..f32_size])
            .expect("write_len should provide correct length");
        let max_next_weight = *f32::ref_from_bytes(&data[f32_size..header_size])
            .expect("write_len should provide correct length");
        let positions = <[u32]>::ref_from_bytes(&data[header_size..])
            .expect("write_len should provide correct length");

        Self {
            token_weight,
            max_next_weight,
            positions: positions.to_vec(),
        }
    }
}

pub trait PositionalPostingValue {
    fn is_empty(&self) -> bool;
    fn to_token_positions(&self, token_id: TokenId) -> Vec<TokenPosition>;
}

impl PositionalPostingValue for Positions {
    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn to_token_positions(&self, token_id: TokenId) -> Vec<TokenPosition> {
        self.to_token_positions(token_id)
    }
}

impl PositionalPostingValue for WeightInfoAndPositions {
    fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    fn to_token_positions(&self, token_id: TokenId) -> Vec<TokenPosition> {
        self.positions
            .iter()
            .map(|pos| TokenPosition {
                token_id,
                position: *pos,
            })
            .collect()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct TokenPosition {
    token_id: TokenId,
    position: u32,
}

/// A reconstructed partial document which stores [`TokenPosition`]s, ordered by positions
#[derive(Debug)]
pub struct PartialDocument(Vec<TokenPosition>);

impl PartialDocument {
    pub fn new(mut tokens_positions: Vec<TokenPosition>) -> Self {
        tokens_positions.sort_by_key(|tok_pos| tok_pos.position);

        // There should be no duplicate token with same position
        debug_assert!(tokens_positions.array_windows().all(|[a, b]| a != b));

        Self(tokens_positions)
    }

    /// Returns true if any sequential window of tokens match the given phrase.
    pub fn has_phrase(&self, phrase: &Document) -> bool {
        match phrase.tokens() {
            // no tokens in query -> no match
            [] => false,

            // single token -> match if any token matches
            [token] => self.0.iter().any(|tok_pos| tok_pos.token_id == *token),

            // multiple tokens -> match if any sequential window matches
            phrase => self.sequential_windows(phrase.len()).any(|seq_window| {
                seq_window
                    .zip(phrase)
                    .all(|(doc_token, query_token)| &doc_token == query_token)
            }),
        }
    }

    /// Returns an iterator over windows which have sequential sequence of tokens.
    ///
    /// Will only return a window if:
    /// - the window is as large as the window size
    /// - all positions in the window are sequential
    fn sequential_windows(
        &self,
        window_size: usize,
    ) -> impl Iterator<Item = impl Iterator<Item = TokenId>> {
        debug_assert!(window_size >= 2, "Window size must be at least 2");
        self.0.windows(window_size).filter_map(|window| {
            // make sure the positions are sequential
            window
                .array_windows()
                .all(|[a, b]| a.position + 1 == b.position)
                .then_some(window.iter().map(|tok_pos| tok_pos.token_id))
        })
    }
}
