use std::borrow::Cow;
use std::marker::PhantomData;
use std::path::PathBuf;
use crate::common::operation_error::OperationResult;
use crate::index::field_index::full_text_index::inverted_index::TokenId;
use crate::index::field_index::full_text_index::inverted_index::mmap_inverted_index::types::{
    PostingListHeader, PostingsHeader, ZerocopyPostingValue,
};
use common::generic_consts::{Random, Sequential};
use common::universal_io::{ReadRange, UniversalIoError, UniversalRead};
use zerocopy::FromBytes;
use common::ext::OptionExt;
use posting_list::PostingListView;
use crate::index::field_index::full_text_index::inverted_index::mmap_inverted_index::raw_posting_list::RawPostingList;

#[allow(dead_code)]
pub struct UniversalPostings<V: ZerocopyPostingValue, S: UniversalRead<u8>> {
    _path: PathBuf,
    storage: S,
    header: PostingsHeader,
    _value_type: PhantomData<V>,
}

#[allow(dead_code)]
impl<V: ZerocopyPostingValue, S: UniversalRead<u8>> UniversalPostings<V, S> {
    /// Fetch posting list headers for a batch of token ids.
    ///
    /// Returns a vector of the same length as `token_ids`; entry `i` is `None`
    /// for token ids outside the valid range, otherwise `Some(header)`.
    fn with_all_headers(
        &self,
        token_ids: &[TokenId],
        mut callback: impl FnMut(TokenId, Option<PostingListHeader>) -> Result<(), UniversalIoError>,
    ) -> OperationResult<()> {
        let header_length = size_of::<PostingListHeader>() as u64;
        let posting_count = self.header.posting_count;

        let mut filtered_out = Vec::new();

        let ranges = token_ids.iter().filter_map(|&token_id| {
            if posting_count <= token_id as usize {
                filtered_out.push(token_id);
                return None;
            }
            let header_offset =
                size_of::<PostingsHeader>() as u64 + u64::from(token_id) * header_length;

            Some((
                token_id,
                ReadRange {
                    byte_offset: header_offset,
                    length: header_length,
                },
            ))
        });

        self.storage
            .read_batch::<Random, _>(ranges, |token_id, bytes| {
                let (header, _) = PostingListHeader::read_from_prefix(bytes)?;
                callback(token_id, Some(header))
            })?;

        // Explicitly report missing posting lists
        // after the `ranges` iterator is consumed.
        for token_id in filtered_out {
            callback(token_id, None)?;
        }

        Ok(())
    }

    fn get_header(&self, token_id: TokenId) -> OperationResult<Option<Cow<'_, PostingListHeader>>> {
        let mut result = None;
        self.with_all_headers(&[token_id], |_token_id, header| {
            result.replace_if_some(header.map(Cow::Owned));
            Ok(())
        })?;
        Ok(result)
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
    fn raw_posting<'a>(
        &'a self,
        header: Cow<'a, PostingListHeader>,
    ) -> Result<RawPostingList<'a>, UniversalIoError> {
        let read_range = ReadRange {
            byte_offset: header.offset,
            length: header.posting_size::<V>() as u64,
        };
        let bytes = self.storage.read::<Sequential>(read_range)?;
        let result = RawPostingList::new(bytes, header);
        Ok(result)
    }

    pub fn get(&self, token_id: TokenId) -> OperationResult<Option<RawPostingList<'_>>> {
        let header = self.get_header(token_id)?;
        if let Some(header) = header {
            let posting = self.raw_posting(header).map(Some)?;
            return Ok(posting);
        }
        Ok(None)
    }

    /// Retrieves all-or-nothing posting lists.
    /// If at least one token is not present, returns `None`.
    /// All posting list at once are propagated into callback
    pub fn with_all_postings<T>(
        &self,
        _token_ids: &[TokenId],
        _callback: impl FnOnce(Vec<(TokenId, PostingListView<'_, V>)>) -> OperationResult<T>,
    ) -> OperationResult<Option<T>> {
        todo!()
    }
}
