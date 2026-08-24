/// One-byte encoding of a non-negative document length.
///
/// Small lengths are exact. Larger lengths retain four significant bits and
/// are rounded down, which preserves ordering for BM25 normalization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub(super) struct EncodedDocumentLength(u8);

impl EncodedDocumentLength {
    pub(super) fn new(document_length: u32) -> Self {
        // Values above the supported range use the maximum representable norm.
        Self(int_to_byte4(document_length.min(i32::MAX as u32)))
    }

    pub(super) fn encoded(self) -> u8 {
        self.0
    }

    pub(super) fn from_encoded(encoded: u8) -> Self {
        Self(encoded)
    }

    pub(super) fn decoded(self) -> u32 {
        byte4_to_int(self.0)
    }
}

const fn long_to_int4(value: u32) -> u32 {
    let num_bits = u32::BITS - value.leading_zeros();
    if num_bits < 4 {
        value
    } else {
        let shift = num_bits - 4;
        let encoded = (value >> shift) & 0x07;
        encoded | ((shift + 1) << 3)
    }
}

const MAX_INT4: u32 = long_to_int4(i32::MAX as u32);
const NUM_FREE_VALUES: u32 = u8::MAX as u32 - MAX_INT4;

const fn int_to_byte4(value: u32) -> u8 {
    if value < NUM_FREE_VALUES {
        value as u8
    } else {
        (NUM_FREE_VALUES + long_to_int4(value - NUM_FREE_VALUES)) as u8
    }
}

const fn int4_to_long(encoded: u32) -> u32 {
    let bits = encoded & 0x07;
    let shift = encoded >> 3;
    if shift == 0 {
        bits
    } else {
        (bits | 0x08) << (shift - 1)
    }
}

const fn byte4_to_int(encoded: u8) -> u32 {
    let encoded = encoded as u32;
    if encoded < NUM_FREE_VALUES {
        encoded
    } else {
        NUM_FREE_VALUES + int4_to_long(encoded - NUM_FREE_VALUES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_norm_matches_known_boundaries() {
        assert_eq!(NUM_FREE_VALUES, 24);
        assert_eq!(EncodedDocumentLength::new(23).encoded(), 23);
        assert_eq!(EncodedDocumentLength::new(24).encoded(), 24);
        assert_eq!(EncodedDocumentLength::new(25).encoded(), 25);
        assert_eq!(EncodedDocumentLength::new(40).encoded(), 40);
        assert_eq!(EncodedDocumentLength::new(41).encoded(), 40);
        assert_eq!(EncodedDocumentLength::new(i32::MAX as u32).encoded(), 255);
        assert_eq!(EncodedDocumentLength::new(u32::MAX).encoded(), 255);
    }

    #[test]
    fn length_norm_decodes_encoded_values() {
        for document_length in 0..=40 {
            let encoded = EncodedDocumentLength::new(document_length);
            assert_eq!(encoded.decoded(), document_length);
        }

        for encoded in 0..=u8::MAX {
            let decoded = EncodedDocumentLength(encoded).decoded();
            assert_eq!(EncodedDocumentLength::new(decoded).encoded(), encoded);
        }

        let maximum_decoded = EncodedDocumentLength(255).decoded();
        assert!(maximum_decoded <= i32::MAX as u32);
        assert_eq!(EncodedDocumentLength::new(maximum_decoded).encoded(), 255);
        assert_eq!(EncodedDocumentLength::new(41).decoded(), 40);
    }
}
