use common::types::PointOffsetType;
use itertools::Either;
use roaring::RoaringBitmap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FrequencyPostingElement {
    point_id: PointOffsetType,
    term_frequency: u32,
}

impl FrequencyPostingElement {
    pub(super) fn point_id(self) -> PointOffsetType {
        self.point_id
    }

    pub(super) fn term_frequency(self) -> u32 {
        self.term_frequency
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct FrequencyPostingList(Vec<FrequencyPostingElement>);

impl FrequencyPostingList {
    fn insert(&mut self, point_id: PointOffsetType, term_frequency: u32) {
        debug_assert!(term_frequency > 0);
        match self
            .0
            .binary_search_by_key(&point_id, |element| element.point_id)
        {
            Ok(index) => self.0[index].term_frequency = term_frequency,
            Err(index) => self.0.insert(
                index,
                FrequencyPostingElement {
                    point_id,
                    term_frequency,
                },
            ),
        }
    }

    fn remove(&mut self, point_id: PointOffsetType) {
        if let Ok(index) = self
            .0
            .binary_search_by_key(&point_id, |element| element.point_id)
        {
            self.0.remove(index);
        }
    }

    fn contains(&self, point_id: PointOffsetType) -> bool {
        self.0
            .binary_search_by_key(&point_id, |element| element.point_id)
            .is_ok()
    }
}

#[derive(Clone, Debug)]
pub enum PostingList {
    Ids(RoaringBitmap),
    WithFrequencies(FrequencyPostingList),
}

impl PostingList {
    pub(super) fn new(with_frequencies: bool) -> Self {
        if with_frequencies {
            Self::WithFrequencies(FrequencyPostingList::default())
        } else {
            Self::default()
        }
    }

    pub fn insert(&mut self, idx: PointOffsetType) {
        match self {
            Self::Ids(list) => {
                list.insert(idx);
            }
            Self::WithFrequencies(_) => {
                panic!("cannot insert an ID-only element into a frequency posting list")
            }
        }
    }

    pub(super) fn insert_frequency(&mut self, point_id: PointOffsetType, term_frequency: u32) {
        match self {
            Self::WithFrequencies(list) => list.insert(point_id, term_frequency),
            Self::Ids(_) => panic!("cannot insert a frequency into an ID-only posting list"),
        }
    }

    pub fn remove(&mut self, idx: PointOffsetType) {
        match self {
            Self::Ids(list) => {
                list.remove(idx);
            }
            Self::WithFrequencies(list) => list.remove(idx),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Ids(list) => list.len() as usize,
            Self::WithFrequencies(list) => list.0.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Ids(list) => list.is_empty(),
            Self::WithFrequencies(list) => list.0.is_empty(),
        }
    }

    #[inline]
    pub fn contains(&self, val: PointOffsetType) -> bool {
        match self {
            Self::Ids(list) => list.contains(val),
            Self::WithFrequencies(list) => list.contains(val),
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = PointOffsetType> + '_ {
        match self {
            Self::Ids(list) => Either::Left(list.iter()),
            Self::WithFrequencies(list) => {
                Either::Right(list.0.iter().map(|element| element.point_id))
            }
        }
    }

    pub(super) fn iter_frequencies(&self) -> impl Iterator<Item = FrequencyPostingElement> + '_ {
        match self {
            Self::Ids(_) => None,
            Self::WithFrequencies(list) => Some(list.0.iter().copied()),
        }
        .into_iter()
        .flatten()
    }

    pub(super) fn frequencies(&self) -> Option<&[FrequencyPostingElement]> {
        match self {
            Self::Ids(_) => None,
            Self::WithFrequencies(list) => Some(&list.0),
        }
    }

    pub fn heap_bytes(&self) -> usize {
        match self {
            // Approximate heap usage with serialized size.
            Self::Ids(list) => list.serialized_size(),
            Self::WithFrequencies(list) => {
                list.0.capacity() * std::mem::size_of::<FrequencyPostingElement>()
            }
        }
    }
}

impl Default for PostingList {
    fn default() -> Self {
        Self::Ids(RoaringBitmap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_posting_keeps_ids_sorted_and_updates_values() {
        let mut posting = PostingList::new(true);
        posting.insert_frequency(3, 1);
        posting.insert_frequency(1, 2);
        posting.insert_frequency(2, 4);
        posting.insert_frequency(2, 5);

        assert_eq!(
            posting
                .iter_frequencies()
                .map(|element| (element.point_id(), element.term_frequency()))
                .collect::<Vec<_>>(),
            vec![(1, 2), (2, 5), (3, 1)],
        );

        posting.remove(2);
        assert_eq!(posting.iter().collect::<Vec<_>>(), vec![1, 3]);
    }
}
