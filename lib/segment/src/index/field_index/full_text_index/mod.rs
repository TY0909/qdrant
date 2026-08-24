use std::path::PathBuf;

use common::universal_io::MmapFile;

use self::immutable_text_index::ImmutableFullTextIndex;
use self::inverted_index::{Bm25Params, bm25_scoring_not_enabled_error};
use self::mutable_text_index::MutableFullTextIndex;
use self::on_disk_text_index::OnDiskFullTextIndex;
use crate::common::operation_error::OperationResult;
use crate::data_types::index::TextIndexParams;

pub mod full_text_index_read;
pub mod full_text_index_scoring;
mod immutable_text_index;
mod inverted_index;
mod lifecycle;
mod mutable_text_index;
pub use mutable_text_index::update_only::UpdateOnlyTextKind;
pub mod on_disk_text_index;
pub mod read_only;
mod read_ops;
pub mod stop_words;
pub mod tokenizers;

pub use read_only::ReadOnlyFullTextIndex;
pub use read_ops::FullTextConditionChecker;

pub(super) fn is_bm25_enabled(config: &TextIndexParams) -> bool {
    config
        .bm25_config
        .as_ref()
        .is_some_and(|bm25| bm25.is_enabled())
}

fn bm25_params(config: &TextIndexParams) -> OperationResult<Bm25Params> {
    configured_bm25_params(config).ok_or_else(bm25_scoring_not_enabled_error)
}

fn configured_bm25_params(config: &TextIndexParams) -> Option<Bm25Params> {
    config
        .bm25_config
        .as_ref()
        .filter(|bm25| bm25.is_enabled())
        .map(Bm25Params::from)
}

#[cfg(test)]
mod tests;

pub enum FullTextIndex {
    Mutable(MutableFullTextIndex),
    Immutable(ImmutableFullTextIndex),
    OnDisk(OnDiskFullTextIndex<MmapFile>),
}

pub struct FullTextGridstoreIndexBuilder {
    dir: PathBuf,
    config: TextIndexParams,
    index: Option<FullTextIndex>,
}
