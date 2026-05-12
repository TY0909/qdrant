use std::sync::Arc;

use common::counter::hardware_counter::HardwareCounterCell;

use crate::common::check_stopped;
use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::PayloadTextSearchContext;
use crate::id_tracker::IdTrackerRead;
use crate::index::PayloadIndexRead;
use crate::payload_storage::PayloadStorageRead;
use crate::segment::read_view::SegmentReadView;
use crate::segment::vector_data_read::VectorDataRead;
use crate::types::{DEFAULT_PAYLOAD_TEXT_FULL_SCAN_THRESHOLD, ScoredPoint};

impl<'s, TIdT, TPI, TPS, TVD> SegmentReadView<'s, TIdT, TPI, TPS, TVD>
where
    TIdT: IdTrackerRead,
    TPI: PayloadIndexRead,
    TPS: PayloadStorageRead,
    TVD: VectorDataRead,
{
    pub fn search_payload_text(
        &self,
        ctx: Arc<PayloadTextSearchContext>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Vec<ScoredPoint>> {
        let PayloadTextSearchContext {
            key,
            query,
            filter,
            top,
            is_stopped,
        } = &*ctx;

        if *top == 0 || query.tokens.is_empty() {
            return Ok(vec![]);
        }

        check_stopped(is_stopped)?;

        let use_plain_filtered_search = match filter {
            Some(filter) => {
                let query_cardinality = self.estimate_point_count(Some(filter), hw_counter)?;
                query_cardinality.max < DEFAULT_PAYLOAD_TEXT_FULL_SCAN_THRESHOLD
            }
            None => false,
        };

        // Find the full text index for the given key.
        let Some(text_index) = self.payload_index.full_text_index_for(key) else {
            return Ok(vec![]);
        };

        // Mirror sparse vector search strategy:
        // for selective filters, materialize matching point ids once and run the
        // iterator-based plain scorer over the sorted subset instead of checking
        // the filter for every candidate from the posting lists.
        let internal_results = if let Some(filter) = filter {
            if use_plain_filtered_search {
                let mut prefiltered_points = self.payload_index.query_points(
                    filter,
                    hw_counter,
                    is_stopped.as_ref(),
                    self.deferred_internal_id(),
                )?;
                prefiltered_points.sort_unstable();
                text_index.search_text_index_plain(query, *top, &prefiltered_points)?
            } else {
                let filter_context = self.payload_index.filter_context(filter, hw_counter)?;
                text_index.search_text_index(query, *top, |point_id| {
                    !self.id_tracker.is_deleted_point(point_id) && filter_context.check(point_id)
                })?
            }
        } else {
            text_index.search_text_index(query, *top, |point_id| {
                !self.id_tracker.is_deleted_point(point_id)
            })?
        };

        self.process_search_result(
            internal_results,
            &false.into(),
            &false.into(),
            hw_counter,
            is_stopped,
        )
    }
}
