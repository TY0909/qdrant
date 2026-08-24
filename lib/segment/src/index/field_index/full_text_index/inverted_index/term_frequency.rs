use posting_list::{PostingValue, SizedHandler, SizedValue};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

#[derive(Default, Clone, Copy, Debug, FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub(in crate::index::field_index::full_text_index) struct TermFrequency {
    value: zerocopy::little_endian::U32,
}

impl TermFrequency {
    pub(in crate::index::field_index::full_text_index) fn new(value: u32) -> Self {
        Self {
            value: zerocopy::little_endian::U32::new(value),
        }
    }

    pub(in crate::index::field_index::full_text_index) fn get(self) -> u32 {
        self.value.get()
    }
}

impl PostingValue for TermFrequency {
    type Handler = SizedHandler<Self>;
}

impl SizedValue for TermFrequency {}
