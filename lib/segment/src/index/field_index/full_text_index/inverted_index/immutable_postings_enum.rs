#[cfg(test)]
use common::types::PointOffsetType;
use posting_list::PostingList;

use super::TokenId;
use super::positions::{Positions, WeightInfo, WeightInfoAndPositions};

#[cfg_attr(test, derive(Clone))]
#[derive(Debug)]
pub enum ImmutablePostings {
    Ids(Vec<PostingList<()>>),
    WithPositions(Vec<PostingList<Positions>>),
    WithWeight(Vec<PostingList<WeightInfo>>),
    WithWeightAndPositions(Vec<PostingList<WeightInfoAndPositions>>),
}

impl ImmutablePostings {
    pub fn len(&self) -> usize {
        match self {
            ImmutablePostings::Ids(lists) => lists.len(),
            ImmutablePostings::WithPositions(lists) => lists.len(),
            ImmutablePostings::WithWeight(lists) => lists.len(),
            ImmutablePostings::WithWeightAndPositions(lists) => lists.len(),
        }
    }

    pub fn posting_len(&self, token: TokenId) -> Option<usize> {
        match self {
            ImmutablePostings::Ids(postings) => {
                postings.get(token as usize).map(|posting| posting.len())
            }
            ImmutablePostings::WithPositions(postings) => {
                postings.get(token as usize).map(|posting| posting.len())
            }
            ImmutablePostings::WithWeight(postings) => {
                postings.get(token as usize).map(|posting| posting.len())
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                postings.get(token as usize).map(|posting| posting.len())
            }
        }
    }

    /// Approximate RAM usage in bytes.
    pub fn ram_usage_bytes(&self) -> usize {
        match self {
            ImmutablePostings::Ids(lists) => {
                lists.capacity() * std::mem::size_of::<PostingList<()>>()
                    + lists.iter().map(|p| p.heap_bytes()).sum::<usize>()
            }
            ImmutablePostings::WithPositions(lists) => {
                lists.capacity() * std::mem::size_of::<PostingList<Positions>>()
                    + lists.iter().map(|p| p.heap_bytes()).sum::<usize>()
            }
            ImmutablePostings::WithWeight(lists) => {
                lists.capacity() * std::mem::size_of::<PostingList<WeightInfo>>()
                    + lists.iter().map(|p| p.heap_bytes()).sum::<usize>()
            }
            ImmutablePostings::WithWeightAndPositions(lists) => {
                lists.capacity() * std::mem::size_of::<PostingList<WeightInfoAndPositions>>()
                    + lists.iter().map(|p| p.heap_bytes()).sum::<usize>()
            }
        }
    }

    #[cfg(test)]
    pub fn iter_ids(
        &self,
        token_id: TokenId,
    ) -> Option<Box<dyn Iterator<Item = PointOffsetType> + '_>> {
        match self {
            ImmutablePostings::Ids(postings) => postings.get(token_id as usize).map(|posting| {
                Box::new(posting.iter().map(|elem| elem.id))
                    as Box<dyn Iterator<Item = PointOffsetType>>
            }),
            ImmutablePostings::WithPositions(postings) => {
                postings.get(token_id as usize).map(|posting| {
                    Box::new(posting.iter().map(|elem| elem.id))
                        as Box<dyn Iterator<Item = PointOffsetType>>
                })
            }
            ImmutablePostings::WithWeight(postings) => {
                postings.get(token_id as usize).map(|posting| {
                    Box::new(posting.iter().map(|elem| elem.id))
                        as Box<dyn Iterator<Item = PointOffsetType>>
                })
            }
            ImmutablePostings::WithWeightAndPositions(postings) => {
                postings.get(token_id as usize).map(|posting| {
                    Box::new(posting.iter().map(|elem| elem.id))
                        as Box<dyn Iterator<Item = PointOffsetType>>
                })
            }
        }
    }
}
