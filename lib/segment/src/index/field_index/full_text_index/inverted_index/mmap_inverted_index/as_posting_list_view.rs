use crate::common::operation_error::OperationResult;
use posting_list::{PostingListView, PostingValue};

pub trait AsPostingListView<'a, V: PostingValue> {
    fn as_view(&'a self) -> OperationResult<PostingListView<'a, V>>;
}

impl<'a, V: PostingValue> AsPostingListView<'a, V> for PostingListView<'a, V> {
    fn as_view(&'a self) -> OperationResult<PostingListView<'a, V>> {
        Ok(self.clone())
    }
}
