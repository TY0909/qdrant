use std::mem::size_of;

use posting_list::{PostingValue, UnsizedHandler, UnsizedValue};
use zerocopy::{FromBytes, IntoBytes};

use super::positions::{Positions, TokenPosition};
use super::term_frequency::TermFrequency;
use super::{PositionalPostingValue, TokenId};

#[derive(Default, Clone, Debug)]
pub(in crate::index::field_index::full_text_index) struct TermFrequencyAndPositions {
    term_frequency: TermFrequency,
    positions: Positions,
}

impl TermFrequencyAndPositions {
    pub(in crate::index::field_index::full_text_index) fn new(
        term_frequency: u32,
        positions: Vec<u32>,
    ) -> Self {
        Self {
            term_frequency: TermFrequency::new(term_frequency),
            positions: Positions::new(positions),
        }
    }
}

impl PostingValue for TermFrequencyAndPositions {
    type Handler = UnsizedHandler<Self>;
}

impl UnsizedValue for TermFrequencyAndPositions {
    fn write_len(&self) -> usize {
        size_of::<TermFrequency>() + self.positions.write_len()
    }

    fn write_to(&self, dst: &mut [u8]) {
        let header_size = size_of::<TermFrequency>();
        assert_eq!(dst.len(), self.write_len(), "destination length must match");
        dst[..header_size].copy_from_slice(self.term_frequency.as_bytes());
        self.positions.write_to(&mut dst[header_size..]);
    }

    fn from_bytes(data: &[u8]) -> Self {
        let header_size = size_of::<TermFrequency>();
        let term_frequency = *TermFrequency::ref_from_bytes(&data[..header_size])
            .expect("serialized term frequency must have the expected size");
        let positions = Positions::from_bytes(&data[header_size..]);
        Self {
            term_frequency,
            positions,
        }
    }
}

impl PositionalPostingValue for TermFrequencyAndPositions {
    fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    fn to_token_positions(&self, token_id: TokenId) -> Vec<TokenPosition> {
        self.positions.to_token_positions(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialization_roundtrip() {
        let value = TermFrequencyAndPositions::new(3, vec![1, 3, 5]);
        let mut bytes = vec![0; value.write_len()];
        value.write_to(&mut bytes);

        let restored = TermFrequencyAndPositions::from_bytes(&bytes);
        assert_eq!(
            restored.to_token_positions(42),
            value.to_token_positions(42),
        );
    }
}
