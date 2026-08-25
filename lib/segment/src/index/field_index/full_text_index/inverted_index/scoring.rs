use std::sync::atomic::{AtomicBool, Ordering};

use common::generic_consts::Random;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};
use common::universal_io::{ReadRange, TypedStorage, UniversalRead};
use posting_list::{PostingIterator, PostingValue};

use super::length_norm::EncodedDocumentLength;
use super::posting_list::FrequencyPostingElement;
use super::{Bm25Params, Bm25Stats, TermFrequencyPostingValue};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::full_text_index::full_text_index_scoring::FullTextSearchScratchPool;

/// Making this larger reduces storage reads at the cost of pooled scratch memory.
const ADVANCE_BATCH_SIZE: usize = 10_000;

#[derive(Clone, Copy)]
pub(super) struct Bm25PostingElement {
    point_id: PointOffsetType,
}

pub(super) trait Bm25PostingListIter {
    fn peek(&mut self) -> Option<Bm25PostingElement>;

    fn last_id(&self) -> Option<PointOffsetType>;

    /// Advance to `point_id` and return its frequency if the exact ID exists.
    fn skip_to(&mut self, point_id: PointOffsetType) -> Option<u32>;

    fn for_each_till_id<Ctx: ?Sized>(
        &mut self,
        last_id: PointOffsetType,
        ctx: &mut Ctx,
        f: impl FnMut(&mut Ctx, PointOffsetType, u32),
    );
}

pub(super) struct MutableBm25PostingListIter<'a> {
    elements: &'a [FrequencyPostingElement],
    offset: usize,
}

impl<'a> MutableBm25PostingListIter<'a> {
    pub(super) fn new(elements: &'a [FrequencyPostingElement]) -> Self {
        Self {
            elements,
            offset: 0,
        }
    }
}

impl Bm25PostingListIter for MutableBm25PostingListIter<'_> {
    fn peek(&mut self) -> Option<Bm25PostingElement> {
        let element = self.elements.get(self.offset)?;
        Some(Bm25PostingElement {
            point_id: element.point_id(),
        })
    }

    fn last_id(&self) -> Option<PointOffsetType> {
        self.elements.last().map(|element| element.point_id())
    }

    fn skip_to(&mut self, point_id: PointOffsetType) -> Option<u32> {
        let remaining = self.elements.get(self.offset..)?;
        let relative = remaining.partition_point(|element| element.point_id() < point_id);
        self.offset += relative;
        let element = self.elements.get(self.offset)?;
        (element.point_id() == point_id).then(|| element.term_frequency())
    }

    fn for_each_till_id<Ctx: ?Sized>(
        &mut self,
        last_id: PointOffsetType,
        ctx: &mut Ctx,
        mut f: impl FnMut(&mut Ctx, PointOffsetType, u32),
    ) {
        while let Some(element) = self.elements.get(self.offset) {
            if element.point_id() > last_id {
                break;
            }
            f(ctx, element.point_id(), element.term_frequency());
            self.offset += 1;
        }
    }
}

pub(super) struct CompressedBm25PostingListIter<'a, V: PostingValue> {
    iterator: PostingIterator<'a, V>,
    current: Option<posting_list::PostingElement<V>>,
    last_id: Option<PointOffsetType>,
}

impl<'a, V: PostingValue> CompressedBm25PostingListIter<'a, V> {
    pub(super) fn new(iterator: PostingIterator<'a, V>, last_id: Option<PointOffsetType>) -> Self {
        Self {
            iterator,
            current: None,
            last_id,
        }
    }
}

impl<V: TermFrequencyPostingValue> Bm25PostingListIter for CompressedBm25PostingListIter<'_, V> {
    fn peek(&mut self) -> Option<Bm25PostingElement> {
        if self.current.is_none() {
            self.current = self.iterator.next();
        }
        let element = self.current.as_ref()?;
        Some(Bm25PostingElement {
            point_id: element.id,
        })
    }

    fn last_id(&self) -> Option<PointOffsetType> {
        self.last_id
    }

    fn skip_to(&mut self, point_id: PointOffsetType) -> Option<u32> {
        if self
            .current
            .as_ref()
            .is_some_and(|element| element.id < point_id)
        {
            self.current = None;
        }
        if self.current.is_none() {
            self.current = self.iterator.advance_until_greater_or_equal(point_id);
        }
        let element = self.current.as_ref()?;
        (element.id == point_id).then(|| element.value.term_frequency())
    }

    fn for_each_till_id<Ctx: ?Sized>(
        &mut self,
        last_id: PointOffsetType,
        ctx: &mut Ctx,
        mut f: impl FnMut(&mut Ctx, PointOffsetType, u32),
    ) {
        if let Some(element) = self.current.take() {
            if element.id > last_id {
                self.current = Some(element);
                return;
            }
            f(ctx, element.id, element.value.term_frequency());
        }
        V::for_each_till_id(&mut self.iterator, last_id, |element| {
            f(ctx, element.id, element.value.term_frequency());
        });
    }
}

pub(super) struct IndexedBm25Posting<I> {
    iterator: I,
    idf: f32,
}

#[derive(Clone, Copy)]
pub(super) struct Bm25SearchOptions<'a> {
    pub(super) params: Bm25Params,
    pub(super) average_document_length: Option<f64>,
    pub(super) top: usize,
    pub(super) is_stopped: &'a AtomicBool,
}

impl<I> IndexedBm25Posting<I> {
    pub(super) fn new(iterator: I, idf: f32) -> Self {
        Self { iterator, idf }
    }
}

pub(super) trait EncodedDocumentLengthProvider {
    fn read_range(
        &self,
        start: PointOffsetType,
        length: usize,
        output: &mut Vec<u8>,
    ) -> OperationResult<()>;

    fn read_points(
        &self,
        point_ids: &[PointOffsetType],
        output: &mut Vec<(PointOffsetType, u8)>,
    ) -> OperationResult<()>;
}

pub(super) struct ExactDocumentLengths<'a>(pub(super) &'a [u32]);

impl EncodedDocumentLengthProvider for ExactDocumentLengths<'_> {
    fn read_range(
        &self,
        start: PointOffsetType,
        length: usize,
        output: &mut Vec<u8>,
    ) -> OperationResult<()> {
        let start = start as usize;
        let end = start
            .checked_add(length)
            .ok_or_else(invalid_document_length_error)?;
        let lengths = self
            .0
            .get(start..end)
            .ok_or_else(invalid_document_length_error)?;
        output.clear();
        output.extend(
            lengths
                .iter()
                .map(|&length| EncodedDocumentLength::new(length).encoded()),
        );
        Ok(())
    }

    fn read_points(
        &self,
        point_ids: &[PointOffsetType],
        output: &mut Vec<(PointOffsetType, u8)>,
    ) -> OperationResult<()> {
        output.clear();
        output.reserve(point_ids.len());
        for &point_id in point_ids {
            let length = self
                .0
                .get(point_id as usize)
                .copied()
                .ok_or_else(invalid_document_length_error)?;
            output.push((point_id, EncodedDocumentLength::new(length).encoded()));
        }
        Ok(())
    }
}

pub(super) struct InMemoryEncodedDocumentLengths<'a>(pub(super) &'a [u8]);

impl EncodedDocumentLengthProvider for InMemoryEncodedDocumentLengths<'_> {
    fn read_range(
        &self,
        start: PointOffsetType,
        length: usize,
        output: &mut Vec<u8>,
    ) -> OperationResult<()> {
        let start = start as usize;
        let end = start
            .checked_add(length)
            .ok_or_else(invalid_document_length_error)?;
        let lengths = self
            .0
            .get(start..end)
            .ok_or_else(invalid_document_length_error)?;
        output.clear();
        output.extend_from_slice(lengths);
        Ok(())
    }

    fn read_points(
        &self,
        point_ids: &[PointOffsetType],
        output: &mut Vec<(PointOffsetType, u8)>,
    ) -> OperationResult<()> {
        output.clear();
        output.reserve(point_ids.len());
        for &point_id in point_ids {
            let length = self
                .0
                .get(point_id as usize)
                .copied()
                .ok_or_else(invalid_document_length_error)?;
            output.push((point_id, length));
        }
        Ok(())
    }
}

pub(super) struct OnDiskEncodedDocumentLengths<'a, S: UniversalRead>(
    pub(super) &'a TypedStorage<S, u8>,
);

impl<S: UniversalRead> EncodedDocumentLengthProvider for OnDiskEncodedDocumentLengths<'_, S> {
    fn read_range(
        &self,
        start: PointOffsetType,
        length: usize,
        output: &mut Vec<u8>,
    ) -> OperationResult<()> {
        let lengths = self.0.read(
            ReadRange {
                byte_offset: u64::from(start),
                length: length as u64,
            },
            Random,
        )?;
        if lengths.len() != length {
            return Err(invalid_document_length_error());
        }
        output.clear();
        output.extend_from_slice(&lengths);
        Ok(())
    }

    fn read_points(
        &self,
        point_ids: &[PointOffsetType],
        output: &mut Vec<(PointOffsetType, u8)>,
    ) -> OperationResult<()> {
        output.clear();
        output.reserve(point_ids.len());
        self.0.read_batch(
            point_ids
                .iter()
                .copied()
                .map(|point_id| (point_id, ReadRange::one(u64::from(point_id)))),
            Random,
            |point_id, values| {
                let Some(&length) = values.first() else {
                    return Err(invalid_document_length_error());
                };
                output.push((point_id, length));
                Ok(())
            },
        )?;
        output.sort_unstable_by_key(|(point_id, _)| *point_id);
        Ok(())
    }
}

fn invalid_document_length_error() -> OperationError {
    OperationError::service_error("BM25 document length is missing for a posting")
}

#[derive(Clone)]
struct Bm25TermScorer {
    norm_inverses: [f32; 256],
}

impl Bm25TermScorer {
    fn new(
        params: Bm25Params,
        stats: Bm25Stats,
        average_document_length: Option<f64>,
    ) -> Option<Self> {
        let average_document_length = average_document_length
            .filter(|average| average.is_finite() && *average > 0.0)
            .or_else(|| stats.average_document_length())?;
        let k1 = params.k1;
        let b = params.b;
        let norm_inverses = std::array::from_fn(|encoded| {
            let document_length =
                f64::from(EncodedDocumentLength::from_encoded(encoded as u8).decoded());
            (1.0 / (k1 * ((1.0 - b) + b * document_length / average_document_length))) as f32
        });
        Some(Self { norm_inverses })
    }

    #[inline]
    fn score(&self, idf: f32, term_frequency: u32, encoded_document_length: u8) -> f32 {
        let norm_inverse = self.norm_inverses[encoded_document_length as usize];
        let scaled_frequency = term_frequency as f32 * norm_inverse;
        idf - idf / (1.0 + scaled_frequency)
    }
}

pub(super) struct Bm25SearchContext<'a, I, L> {
    postings: Vec<IndexedBm25Posting<I>>,
    document_lengths: L,
    term_scorer: Bm25TermScorer,
    top: usize,
    is_stopped: &'a AtomicBool,
}

impl<'a, I, L> Bm25SearchContext<'a, I, L>
where
    I: Bm25PostingListIter,
    L: EncodedDocumentLengthProvider,
{
    pub(super) fn new(
        postings: Vec<IndexedBm25Posting<I>>,
        document_lengths: L,
        params: Bm25Params,
        stats: Bm25Stats,
        average_document_length: Option<f64>,
        top: usize,
        is_stopped: &'a AtomicBool,
    ) -> Option<Self> {
        Some(Self {
            postings,
            document_lengths,
            term_scorer: Bm25TermScorer::new(params, stats, average_document_length)?,
            top,
            is_stopped,
        })
    }

    pub(super) fn plain_search(
        mut self,
        search_scratch_pool: &FullTextSearchScratchPool,
        ordered_point_ids: &[PointOffsetType],
        is_active: impl Fn(PointOffsetType) -> bool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if self.postings.is_empty() || self.top == 0 {
            return Ok(Vec::new());
        }

        let mut top_results = TopK::new(self.top);
        let mut scratch_guard = search_scratch_pool.get();
        let scratch = scratch_guard.scratch();

        for candidates in ordered_point_ids.chunks(ADVANCE_BATCH_SIZE) {
            if self.is_stopped.load(Ordering::Relaxed) {
                break;
            }
            self.document_lengths
                .read_points(candidates, &mut scratch.selected_document_lengths)?;

            for &(point_id, encoded_document_length) in &scratch.selected_document_lengths {
                if !is_active(point_id) {
                    continue;
                }
                let mut score = 0.0;
                for posting in &mut self.postings {
                    if let Some(term_frequency) = posting.iterator.skip_to(point_id) {
                        score += self.term_scorer.score(
                            posting.idf,
                            term_frequency,
                            encoded_document_length,
                        );
                    }
                }
                if score > 0.0 {
                    top_results.push(ScoredPointOffset {
                        idx: point_id,
                        score,
                    });
                }
            }
        }

        Ok(top_results.into_vec())
    }

    pub(super) fn search(
        mut self,
        search_scratch_pool: &FullTextSearchScratchPool,
        filter: impl Fn(PointOffsetType) -> bool,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if self.postings.is_empty() || self.top == 0 {
            return Ok(Vec::new());
        }

        let mut top_results = TopK::new(self.top);
        let mut scratch_guard = search_scratch_pool.get();
        let scratch = scratch_guard.scratch();
        let max_point_id = self
            .postings
            .iter()
            .filter_map(|posting| posting.iterator.last_id())
            .max()
            .unwrap_or(0);

        while let Some(batch_start) = Self::next_min_id(&mut self.postings) {
            if self.is_stopped.load(Ordering::Relaxed) {
                break;
            }

            let batch_last = batch_start
                .saturating_add(ADVANCE_BATCH_SIZE as PointOffsetType - 1)
                .min(max_point_id);
            let batch_len = (batch_last - batch_start + 1) as usize;
            self.document_lengths.read_range(
                batch_start,
                batch_len,
                &mut scratch.document_lengths,
            )?;
            scratch.scores.resize(batch_len, 0.0);
            scratch.scores.fill(0.0);

            for posting in &mut self.postings {
                let idf = posting.idf;
                let term_scorer = &self.term_scorer;
                let document_lengths = &scratch.document_lengths;
                posting.iterator.for_each_till_id(
                    batch_last,
                    &mut scratch.scores,
                    |scores, point_id, term_frequency| {
                        let local = (point_id - batch_start) as usize;
                        let encoded_document_length = document_lengths[local];
                        scores[local] +=
                            term_scorer.score(idf, term_frequency, encoded_document_length);
                    },
                );
            }

            for (local, &score) in scratch.scores.iter().enumerate() {
                if score > 0.0 && score > top_results.threshold() {
                    let point_id = batch_start + local as PointOffsetType;
                    if filter(point_id) {
                        top_results.push(ScoredPointOffset {
                            idx: point_id,
                            score,
                        });
                    }
                }
            }
        }

        Ok(top_results.into_vec())
    }

    fn next_min_id(postings: &mut [IndexedBm25Posting<I>]) -> Option<PointOffsetType> {
        postings
            .iter_mut()
            .filter_map(|posting| posting.iterator.peek().map(|element| element.point_id))
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_score_uses_average_document_length() {
        let scorer = Bm25TermScorer::new(
            Bm25Params { k1: 1.2, b: 0.75 },
            Bm25Stats {
                doc_count: 2,
                sum_doc_len: 10,
            },
            None,
        )
        .unwrap();

        let short = EncodedDocumentLength::new(2).encoded();
        let long = EncodedDocumentLength::new(8).encoded();
        assert!(scorer.score(1.0, 1, short) > scorer.score(1.0, 1, long));
        assert!(scorer.score(1.0, 2, short) > scorer.score(1.0, 1, short));
    }

    #[test]
    fn term_score_accepts_query_average_document_length() {
        let stats = Bm25Stats {
            doc_count: 2,
            sum_doc_len: 10,
        };
        let local = Bm25TermScorer::new(Bm25Params { k1: 1.2, b: 0.75 }, stats, None).unwrap();
        let global =
            Bm25TermScorer::new(Bm25Params { k1: 1.2, b: 0.75 }, stats, Some(20.0)).unwrap();
        let document_length = EncodedDocumentLength::new(10).encoded();

        assert_ne!(
            local.score(1.0, 1, document_length),
            global.score(1.0, 1, document_length),
        );
    }
}
