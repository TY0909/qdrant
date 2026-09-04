use std::fmt::Debug;
use std::sync::atomic::AtomicBool;

use common::defaults::POOL_KEEP_LIMIT;
use common::types::{PointOffsetType, ScoredPointOffset};
use parking_lot::Mutex;

use crate::common::operation_error::OperationResult;
use crate::types::QueryTokenWeightSet;

#[derive(Default)]
pub(super) struct FullTextSearchScratch {
    pub(super) scores: Vec<f32>,
    pub(super) document_lengths: Vec<u8>,
    pub(super) selected_document_lengths: Vec<(PointOffsetType, u8)>,
}

/// Reusable buffers for full-text scoring.
pub(super) struct FullTextSearchScratchPool {
    pool: Mutex<Vec<FullTextSearchScratch>>,
}

impl FullTextSearchScratchPool {
    pub(super) fn new() -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(*POOL_KEEP_LIMIT)),
        }
    }

    pub(super) fn get(&self) -> FullTextSearchScratchGuard<'_> {
        let scratch = self.pool.lock().pop().unwrap_or_default();
        FullTextSearchScratchGuard {
            pool: self,
            scratch: Some(scratch),
        }
    }
}

impl Default for FullTextSearchScratchPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for FullTextSearchScratchPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FullTextSearchScratchPool")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl Clone for FullTextSearchScratchPool {
    fn clone(&self) -> Self {
        Self::new()
    }
}

pub(super) struct FullTextSearchScratchGuard<'a> {
    pool: &'a FullTextSearchScratchPool,
    scratch: Option<FullTextSearchScratch>,
}

impl FullTextSearchScratchGuard<'_> {
    pub(super) fn scratch(&mut self) -> &mut FullTextSearchScratch {
        self.scratch
            .get_or_insert_with(FullTextSearchScratch::default)
    }
}

impl Drop for FullTextSearchScratchGuard<'_> {
    fn drop(&mut self) {
        let Some(mut scratch) = self.scratch.take() else {
            return;
        };
        scratch.scores.clear();
        scratch.document_lengths.clear();
        scratch.selected_document_lengths.clear();

        let mut pool = self.pool.pool.lock();
        if pool.len() < *POOL_KEEP_LIMIT {
            pool.push(scratch);
        }
    }
}

/// Scores tokenized full-text queries whose token weights are resolved by the caller.
pub trait FullTextIndexScoring {
    fn search_text_index<F>(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        is_stopped: &AtomicBool,
        filter: F,
    ) -> OperationResult<Vec<ScoredPointOffset>>
    where
        F: Fn(PointOffsetType) -> bool;

    fn search_text_index_plain(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        ordered_prefiltered_points: &[PointOffsetType],
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>>;
}

impl<T: FullTextIndexScoring + ?Sized> FullTextIndexScoring for &T {
    fn search_text_index<F>(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        is_stopped: &AtomicBool,
        filter: F,
    ) -> OperationResult<Vec<ScoredPointOffset>>
    where
        F: Fn(PointOffsetType) -> bool,
    {
        (*self).search_text_index(query, top, is_stopped, filter)
    }

    fn search_text_index_plain(
        &self,
        query: &QueryTokenWeightSet,
        top: usize,
        ordered_prefiltered_points: &[PointOffsetType],
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        (*self).search_text_index_plain(query, top, ordered_prefiltered_points, is_stopped)
    }
}

#[cfg(test)]
mod tests {
    use super::FullTextSearchScratchPool;

    #[test]
    fn scratch_buffers_are_cleared_and_reused() {
        let pool = FullTextSearchScratchPool::new();
        {
            let mut guard = pool.get();
            let scratch = guard.scratch();
            scratch.scores.push(1.0);
            scratch.document_lengths.push(2);
            scratch.selected_document_lengths.push((3, 4));
        }

        assert_eq!(pool.pool.lock().len(), 1);
        let mut guard = pool.get();
        let scratch = guard.scratch();
        assert!(scratch.scores.is_empty());
        assert!(scratch.document_lengths.is_empty());
        assert!(scratch.selected_document_lengths.is_empty());
    }
}
