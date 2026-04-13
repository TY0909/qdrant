use std::iter;
use std::marker::PhantomData;

use ::io_uring::types::Fd;
use ahash::AHashMap;

use super::*;

pub struct IoUringReadIter<T: 'static, Meta, I: Iterator> {
    ranges: iter::Peekable<I>,
    runtime: IoUringRuntime<'static, T, (usize, Meta)>,
    next_submit_seq: usize,
    next_seq: usize,
    buffer: AHashMap<usize, (Meta, Vec<T>)>,
    _phantom: PhantomData<*const ()>, // `!Send + !Sync`
}

impl<T, Meta, I> IoUringReadIter<T, Meta, I>
where
    T: bytemuck::Pod,
    I: Iterator<Item = (Meta, Fd, bool, ReadRange)>,
{
    pub fn new(ranges: I) -> Result<Self> {
        let iter = Self {
            ranges: ranges.peekable(),
            runtime: IoUringRuntime::new()?,
            next_submit_seq: 0,
            next_seq: 0,
            buffer: Default::default(),
            _phantom: PhantomData,
        };

        Ok(iter)
    }

    fn next_impl(&mut self) -> Result<Option<(Meta, Vec<T>)>> {
        loop {
            // If the next expected item is already buffered, return it.
            if let Some(item) = self.buffer.remove(&self.next_seq) {
                self.next_seq += 1;
                return Ok(Some(item));
            }

            // Nothing left to process — we're done.
            if self.ranges.peek().is_none() && self.runtime.in_progress == 0 {
                debug_assert!(self.buffer.is_empty());
                return Ok(None);
            }

            // Saturate the submission queue with new requests.
            self.runtime.enqueue_while(|state| {
                let Some((meta, fd, direct_io, range)) = self.ranges.next() else {
                    return Ok(None);
                };
                let seq = self.next_submit_seq;
                self.next_submit_seq += 1;
                let entry = state.read((seq, meta), fd, range, direct_io);
                Ok(Some(entry))
            })?;

            // Wait for at least one completion.
            self.runtime.submit_and_wait(1)?;

            // Drain all available completions into the buffer.
            for result in self.runtime.completed() {
                let ((seq, meta), resp) = result?;
                if seq == self.next_seq {
                    // This is the next expected item, return it immediately.
                    self.next_seq += 1;
                    return Ok(Some((meta, resp.expect_read())));
                } else {
                    self.buffer.insert(seq, (meta, resp.expect_read()));
                }
            }
        }
    }
}

impl<T, Meta, I> Iterator for IoUringReadIter<T, Meta, I>
where
    T: bytemuck::Pod,
    I: Iterator<Item = (Meta, Fd, bool, ReadRange)>,
{
    type Item = Result<(Meta, Vec<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl().transpose()
    }
}
