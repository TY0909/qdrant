use std::sync::Arc;

/// A counter to track the number of active processes.
///
/// Conceptually, similar to a simple integer counter, e.g.:
/// ```ignore
/// let mut counter: usize = 0;
/// counter += 1; // on process start
/// counter -= 1; // on process end
/// ```
/// but decrements on drop, making it impossible to forget decreasing the counter.
/// ```ignore
/// let mut counter = ProcessesCounter::default();
/// let handle = counter.inc(); // on process start
/// drop(handle); // on process end
/// ```
#[derive(Debug, Default)] // No `Clone` as it will break the counting logic.
pub struct ProcessesCounter(Arc<()>);

/// A handle that decreases [`ProcessesCounter`] by one when dropped.
#[derive(Debug)] // No `Clone` as it will break the counting logic.
#[must_use = "Dropping this handle will immediately decrease the process count"]
#[expect(dead_code, reason = "The inner values is used to increase ref count")]
pub struct CountedProcessHandle(Arc<()>);

impl ProcessesCounter {
    /// The current counter value.
    pub fn count(&self) -> usize {
        Arc::strong_count(&self.0).saturating_sub(1)
    }

    /// Increase the counter by one.
    /// Returns a handle that decreases it back when dropped.
    pub fn inc(&mut self) -> CountedProcessHandle {
        // NOTE: `&mut` to make it harder to misuse.
        CountedProcessHandle(Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processes_counter() {
        let mut counter = ProcessesCounter::default();
        assert_eq!(counter.count(), 0);

        let p1 = counter.inc();
        assert_eq!(counter.count(), 1);

        let p2 = counter.inc();
        assert_eq!(counter.count(), 2);

        let p3 = counter.inc();
        assert_eq!(counter.count(), 3);
        _ = std::panic::catch_unwind(move || {
            let _p3 = p3;
            panic!();
        });
        assert_eq!(counter.count(), 2);

        drop(p1);
        assert_eq!(counter.count(), 1);

        drop(p2);
        assert_eq!(counter.count(), 0);
    }
}
