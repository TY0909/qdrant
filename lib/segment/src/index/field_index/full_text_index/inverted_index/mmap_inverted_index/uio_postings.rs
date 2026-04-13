use std::borrow::Cow;
use std::marker::PhantomData;
use std::path::PathBuf;

use common::generic_consts::Random;
use common::types::PointOffsetType;
use common::universal_io::{ReadRange, UniversalRead};
use posting_list::{PostingChunk, PostingListView, RemainderPosting, SizedTypeFor};
use zerocopy::FromBytes;

use crate::common::operation_error::OperationResult;
use crate::index::field_index::full_text_index::inverted_index::TokenId;
use crate::index::field_index::full_text_index::inverted_index::mmap_inverted_index::types::{
    PostingListHeader, PostingsHeader, ZerocopyPostingValue,
};

#[allow(dead_code)]
pub struct UniversalPostings<V: ZerocopyPostingValue, S: UniversalRead<u8>> {
    _path: PathBuf,
    storage: S,
    header: PostingsHeader,
    _value_type: PhantomData<V>,
}

#[allow(dead_code)]
impl<V: ZerocopyPostingValue, S: UniversalRead<u8>> UniversalPostings<V, S> {
    fn get_header(&self, token_id: TokenId) -> OperationResult<Option<Cow<'_, PostingListHeader>>> {
        if self.header.posting_count <= token_id as usize {
            return Ok(None);
        }

        let header_length = size_of::<PostingListHeader>() as u64;
        let header_offset =
            size_of::<PostingsHeader>() as u64 + u64::from(token_id) * header_length;

        let header_bytes = self.storage.read::<Random>(ReadRange {
            byte_offset: header_offset,
            length: header_length,
        })?;

        let header = match header_bytes {
            Cow::Borrowed(bytes) => {
                let (h, _) = PostingListHeader::ref_from_prefix(bytes)?;
                Cow::Borrowed(h)
            }
            Cow::Owned(bytes) => {
                let (h, _) = PostingListHeader::read_from_prefix(bytes.as_slice())?;
                Cow::Owned(h)
            }
        };

        Ok(Some(header))
    }

    /// Create PostingListView<V> from the given header
    ///
    /// Assume the following layout:
    ///
    /// ```ignore
    /// last_doc_id: &'a PointOffsetType,
    /// chunks_index: &'a [PostingChunk<()>],
    /// data: &'a [u8],
    /// var_size_data: &'a [u8], // might be empty in case of only ids
    /// _alignment: &'a [u8], // 0-3 extra bytes to align the data
    /// remainder_postings: &'a [PointOffsetType],
    /// ```
    fn with_view<T>(
        &self,
        header: &PostingListHeader,
        callback: impl Fn(PostingListView<'_, V>) -> T,
    ) -> OperationResult<Option<T>> {
        let read_range = ReadRange {
            byte_offset: header.offset,
            length: header.posting_size::<V>() as u64,
        };

        let raw_bytes = self.storage.read::<Random>(read_range)?;

        let (last_doc_id, bytes) = PointOffsetType::read_from_prefix(raw_bytes.as_ref())?;

        let (chunks, bytes) = <[PostingChunk<SizedTypeFor<V>>]>::ref_from_prefix_with_elems(
            bytes,
            header.chunks_count as usize,
        )?;

        let (id_data, bytes) = bytes.split_at(header.ids_data_bytes_count as usize);

        let (var_size_data, bytes) = bytes.split_at(header.var_size_data_bytes_count as usize);

        // skip padding
        let bytes = &bytes[header.alignment_bytes_count as usize..];

        let (remainder_postings, _) =
            <[RemainderPosting<SizedTypeFor<V>>]>::ref_from_prefix_with_elems(
                bytes,
                header.remainder_count as usize,
            )?;

        let posting_list_view = PostingListView::from_components(
            id_data,
            chunks,
            var_size_data,
            remainder_postings,
            Some(last_doc_id),
        );

        let result = callback(posting_list_view);

        Ok(Some(result))
    }
}
