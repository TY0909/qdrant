use crate::index::field_index::full_text_index::inverted_index::mmap_inverted_index::types::{
    ZerocopyPostingValue, PostingsHeader,
};
use common::universal_io::UniversalRead;
use std::marker::PhantomData;
use std::path::PathBuf;

#[allow(dead_code)]
pub struct MmapPostings<V: ZerocopyPostingValue, S: UniversalRead<u8>> {
    _path: PathBuf,
    storage: S,
    header: PostingsHeader,
    _value_type: PhantomData<V>,
}
