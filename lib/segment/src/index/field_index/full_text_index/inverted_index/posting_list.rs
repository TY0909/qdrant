use common::types::PointOffsetType;
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;

use super::{DEFAULT_MAX_NEXT_WEIGHT, TokenWeight};

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub(super) struct PostingElementEx {
    /// Point ID
    point_id: PointOffsetType,
    /// Weight of the point in the dimension
    weight: TokenWeight,
    /// Max weight of the next elements in the posting list.
    max_next_weight: TokenWeight,
}

impl PostingElementEx {
    /// Initialize negative infinity as max_next_weight.
    /// Needs to be updated at insertion time.
    pub(super) fn new(point_id: PointOffsetType, weight: TokenWeight) -> PostingElementEx {
        Self {
            point_id,
            weight,
            max_next_weight: DEFAULT_MAX_NEXT_WEIGHT,
        }
    }

    pub(super) fn point_id(&self) -> u32 {
        self.point_id
    }

    pub(super) fn weight(&self) -> f32 {
        self.weight
    }

    pub(super) fn max_next_weight(&self) -> f32 {
        self.max_next_weight
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct PostingElementExList(Vec<PostingElementEx>);

impl PostingElementExList {
    fn insert(&mut self, idx: PointOffsetType, weight: TokenWeight) {
        // find insertion point in sorted posting list (most expensive operation for large posting list)
        let index = self.0.binary_search_by_key(&idx, |e| e.point_id);

        let modified_index = match index {
            Ok(found_index) => {
                // Update existing element for the same id
                let element = &mut self.0[found_index];
                if element.weight == weight {
                    // no need to update anything
                    None
                } else {
                    // the structure of the posting list is not changed, no need to update max_next_weight
                    element.weight = weight;
                    Some(found_index)
                }
            }
            Err(insert_index) => {
                let new_posting_element = PostingElementEx::new(idx, weight);
                // Insert new element by shifting elements to the right
                self.0.insert(insert_index, new_posting_element);
                // the structure of the posting list is changed, need to update max_next_weight
                if insert_index == self.0.len() - 1 {
                    // inserted at the end
                    Some(insert_index)
                } else {
                    // inserted in the middle - need to propagated max_next_weight from the right
                    Some(insert_index + 1)
                }
            }
        };
        // Propagate max_next_weight update to the previous entries
        if let Some(modified_index) = modified_index {
            self.propagate_max_next_weight_to_the_left(modified_index);
        }
    }

    fn remove(&mut self, idx: PointOffsetType) {
        let index = self.0.binary_search_by_key(&idx, |e| e.point_id);
        if let Ok(found_index) = index {
            self.0.remove(found_index);
            if let Some(last) = self.0.last_mut() {
                last.max_next_weight = DEFAULT_MAX_NEXT_WEIGHT;
            }
            if found_index < self.0.len() {
                self.propagate_max_next_weight_to_the_left(found_index);
            } else if !self.0.is_empty() {
                self.propagate_max_next_weight_to_the_left(self.0.len() - 1);
            }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn contains(&self, val: PointOffsetType) -> bool {
        self.0
            .binary_search_by_key(&val, |element| element.point_id)
            .is_ok()
    }

    fn iter_point_id(&self) -> impl Iterator<Item = PointOffsetType> + '_ {
        self.0.iter().map(|e| e.point_id)
    }

    fn iter_element(&self) -> impl Iterator<Item = &PostingElementEx> + '_ {
        self.0.iter()
    }

    /// Returns a slice-based cursor for efficient ordered traversal.
    fn as_slice(&self) -> &[PostingElementEx] {
        &self.0
    }

    fn serialized_size(&self) -> usize {
        let elem_size = std::mem::size_of::<PostingElementEx>();
        std::mem::size_of::<Vec<PostingElementEx>>() + self.0.capacity() * elem_size
    }

    /// Propagates `max_next_weight` from the entry at `up_to_index` to previous entries.
    /// If an entry has a weight larger than `max_next_weight`, the propagation stops.
    fn propagate_max_next_weight_to_the_left(&mut self, up_to_index: usize) {
        // used element at `up_to_index` as the starting point
        let starting_element = &self.0[up_to_index];
        let mut max_next_weight = core::cmp::max(
            OrderedFloat(starting_element.max_next_weight),
            OrderedFloat(starting_element.weight),
        )
        .0;

        // propagate max_next_weight update to the previous entries
        for element in self.0[..up_to_index].iter_mut().rev() {
            // update max_next_weight for element
            element.max_next_weight = max_next_weight;
            max_next_weight = max_next_weight.max(element.weight);
        }
    }
}

#[derive(Clone, Debug)]
pub enum PostingList {
    Ids { list: RoaringBitmap },
    WithWeight { list: PostingElementExList },
}

impl Default for PostingList {
    fn default() -> Self {
        Self::Ids {
            list: Default::default(),
        }
    }
}

impl PostingList {
    pub fn insert(&mut self, idx: PointOffsetType, weight: Option<TokenWeight>) {
        match self {
            PostingList::Ids { list } => {
                list.insert(idx);
            }
            PostingList::WithWeight { list } => {
                if let Some(weight) = weight {
                    list.insert(idx, weight);
                }
            }
        }
    }

    pub fn remove(&mut self, idx: PointOffsetType) {
        match self {
            PostingList::Ids { list } => {
                list.remove(idx);
            }
            PostingList::WithWeight { list } => {
                list.remove(idx);
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            PostingList::Ids { list } => list.len() as usize,
            PostingList::WithWeight { list } => list.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            PostingList::Ids { list } => list.is_empty(),
            PostingList::WithWeight { list } => list.is_empty(),
        }
    }

    #[inline]
    pub fn contains(&self, val: PointOffsetType) -> bool {
        match self {
            PostingList::Ids { list } => list.contains(val),
            PostingList::WithWeight { list } => list.contains(val),
        }
    }

    #[inline]
    pub fn iter(&self) -> Box<dyn Iterator<Item = PointOffsetType> + '_> {
        match self {
            PostingList::Ids { list } => Box::new(list.iter()),
            PostingList::WithWeight { list } => Box::new(list.iter_point_id()),
        }
    }

    #[inline]
    pub(super) fn iter_element(&self) -> Box<dyn Iterator<Item = &PostingElementEx> + '_> {
        match self {
            PostingList::Ids { list: _ } => Box::new(std::iter::empty()),
            PostingList::WithWeight { list } => Box::new(list.iter_element()),
        }
    }

    pub fn heap_bytes(&self) -> usize {
        match self {
            // Approximate heap usage with serialized size
            PostingList::Ids { list } => list.serialized_size(),
            PostingList::WithWeight { list } => list.serialized_size(),
        }
    }

    /// Returns a cursor for efficient ordered traversal of weighted posting lists.
    /// Returns `None` for ID-only posting lists.
    pub(super) fn weight_cursor(&self) -> Option<PostingWeightCursor<'_>> {
        match self {
            PostingList::Ids { .. } => None,
            PostingList::WithWeight { list } => Some(PostingWeightCursor {
                elements: list.as_slice(),
                current_index: 0,
            }),
        }
    }
}

/// Cursor over a sorted weighted posting list that supports efficient forward traversal.
///
/// Uses binary search from the current position to skip to a target point ID,
/// avoiding redundant work when the caller iterates in sorted order.
pub(super) struct PostingWeightCursor<'a> {
    elements: &'a [PostingElementEx],
    current_index: usize,
}

impl PostingWeightCursor<'_> {
    /// Advance the cursor to `point_id` and return its weight if present.
    ///
    /// The cursor is advanced so that subsequent calls with higher IDs
    /// only search the remaining (unconsumed) portion of the list.
    #[inline]
    pub(super) fn skip_to(&mut self, point_id: PointOffsetType) -> Option<TokenWeight> {
        let remaining = &self.elements[self.current_index..];
        match remaining.binary_search_by_key(&point_id, |e| e.point_id) {
            Ok(offset) => {
                self.current_index += offset;
                Some(self.elements[self.current_index].weight)
            }
            Err(offset) => {
                // Advance past elements that are smaller than point_id
                self.current_index += offset;
                None
            }
        }
    }
}
