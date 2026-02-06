use std::ops::Range;

use bitvec::vec::BitVec;
use common::ext::BitSliceExt;
use common::types::PointOffsetType;

// This const ratio defines the maximum allowed proportion of empty values when enabling direct access optimization
const MAX_NO_VALUES_RATIO: f32 = 0.2;

// Flatten points-to-values map
// It's an analogue of `Vec<Vec<N>>` but more RAM efficient because it stores values in a single Vec.
// This structure doesn't support adding new values, only removing.
// It's used in immutable field indices like `ImmutableMapIndex`, `ImmutableNumericIndex`, etc to store points-to-values map.
#[derive(Debug, Clone, Default)]
pub struct ImmutablePointToValues<N: Default> {
    // ranges in `point_to_values_container` which contains values for each point
    // `u32` is used instead of `usize` because it's more RAM efficient
    // We can expect that we will never have more than 4 billion values per segment
    point_to_values: Vec<Range<u32>>,
    // flattened values
    point_to_values_container: Vec<N>,

    // When `direct_access_container_enabled` is set to true,
    // we can get value from `point_to_values_container` instead of getting the range and then getting the values slice
    //
    // This optimization only works for immutable field indices that have at most one value for each point
    // which means that the `Vec<N>` in `Vec<Vec<N>>` has either one element or is empty
    //
    // For those points that have no value, we use a default placeholder and set a ratio which is `MAX_NO_VALUES_RATIO`
    // Because when there are too many points with no value, the default placeholder will lead to memory waste.
    // When there are too many points with no value, the optimization will not be enabled.
    //
    // This optimization should always be used with `values_container_access`.
    // There should be only one index works between `point_to_values` and `values_container_access`.
    // When this optimization is enabled, you should not access the `point_to_values`.
    // When this optimization is disabled, you should not access the `values_container_access`.
    direct_access_container_enabled: bool,
    // Store the access to the `point_to_values_container`.
    // When the corresponding access is false, you should not access this value.
    // Because this means that this value has been removed or empty value.
    values_container_access: BitVec,
}

impl<N: Default> ImmutablePointToValues<N> {
    pub fn new(src: Vec<Vec<N>>) -> Self {
        // Make sure that all values have at most one element and the empty values ratio is less than the threshold we set.
        // When all condition are matched, this optimization is used
        let direct_access_container_enabled =
            !src.is_empty() && src.iter().all(|values| values.len() <= 1) && {
                (src.iter().filter(|values| values.is_empty()).count() as f32 / src.len() as f32)
                    <= MAX_NO_VALUES_RATIO
            };

        if direct_access_container_enabled {
            let mut values_container_access = BitVec::with_capacity(src.len());
            let mut point_to_values_container = Vec::with_capacity(src.len());
            for values in src {
                if values.is_empty() {
                    values_container_access.push(false);
                    point_to_values_container.push(N::default());
                } else {
                    values_container_access.push(true);
                    point_to_values_container.extend(values);
                }
            }
            Self {
                point_to_values: Vec::new(),
                point_to_values_container,
                direct_access_container_enabled,
                values_container_access,
            }
        } else {
            let mut point_to_values = Vec::with_capacity(src.len());
            let all_values_count = src.iter().fold(0, |acc, values| acc + values.len());
            let mut point_to_values_container = Vec::with_capacity(all_values_count);
            for values in src {
                let container_len = point_to_values_container.len() as u32;
                let range = container_len..container_len + values.len() as u32;
                point_to_values.push(range.clone());
                point_to_values_container.extend(values);
            }
            Self {
                point_to_values,
                point_to_values_container,
                direct_access_container_enabled,
                values_container_access: BitVec::new(),
            }
        }
    }

    // Check if the value is accessible. It could be false or none
    // When it is false, it means that it could be removed or empty value
    // When it is none, it means that it's not exist
    fn is_values_accessible(&self, idx: PointOffsetType) -> Option<bool> {
        self.values_container_access.get_bit(idx as usize)
    }

    pub fn check_values_any(&self, idx: PointOffsetType, check_fn: impl FnMut(&N) -> bool) -> bool {
        if self.direct_access_container_enabled {
            if self.is_values_accessible(idx).unwrap_or(false) {
                self.point_to_values_container
                    .get(idx as usize)
                    .is_some_and(check_fn)
            } else {
                false
            }
        } else {
            let Some(range) = self.point_to_values.get(idx as usize).cloned() else {
                return false;
            };

            let range = range.start as usize..range.end as usize;
            if let Some(values) = self.point_to_values_container.get(range) {
                values.iter().any(check_fn)
            } else {
                false
            }
        }
    }

    pub fn get_values(&self, idx: PointOffsetType) -> Option<impl Iterator<Item = &N> + '_> {
        if self.direct_access_container_enabled {
            if self.is_values_accessible(idx)? {
                self.point_to_values_container
                    .get(idx as usize)
                    .map(|v| std::slice::from_ref(v).iter())
            } else {
                Some(self.point_to_values_container[0..0].iter())
            }
        } else {
            let range = self.point_to_values.get(idx as usize)?.clone();
            let range = range.start as usize..range.end as usize;
            Some(self.point_to_values_container[range].iter())
        }
    }

    pub fn get_values_count(&self, idx: PointOffsetType) -> Option<usize> {
        if self.direct_access_container_enabled {
            if self.is_values_accessible(idx)? {
                Some(1)
            } else {
                Some(0)
            }
        } else {
            self.point_to_values
                .get(idx as usize)
                .map(|range| (range.end - range.start) as usize)
        }
    }

    pub fn remove_point(&mut self, idx: PointOffsetType) -> Vec<N> {
        if self.direct_access_container_enabled {
            if self.is_values_accessible(idx).unwrap_or(false) {
                self.values_container_access.set(idx as usize, false);
                let value = std::mem::take(&mut self.point_to_values_container[idx as usize]);
                vec![value]
            } else {
                Default::default()
            }
        } else {
            if self.point_to_values.len() <= idx as usize {
                return Default::default();
            }

            let removed_values_range = self.point_to_values[idx as usize].clone();
            self.point_to_values[idx as usize] = Default::default();

            let mut result = Vec::with_capacity(removed_values_range.len());
            for value_index in removed_values_range {
                // deleted values still use RAM, but it's not a problem because optimizers will actually reduce RAM usage
                let value =
                    std::mem::take(&mut self.point_to_values_container[value_index as usize]);
                result.push(value);
            }

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immutable_point_to_values_remove() {
        let mut values = vec![
            vec![0, 1, 2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![0, 1, 2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![10, 11, 12],
            vec![],
            vec![13],
            vec![14, 15],
        ];

        let mut point_to_values = ImmutablePointToValues::new(values.clone());

        let check = |point_to_values: &ImmutablePointToValues<_>, values: &[Vec<_>]| {
            for (idx, values) in values.iter().enumerate() {
                let values_vec: Option<Vec<_>> = point_to_values
                    .get_values(idx as PointOffsetType)
                    .map(|i| i.copied().collect());
                assert_eq!(values_vec, Some(values.clone()),);
            }
        };

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(0);
        values[0].clear();

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(3);
        values[3].clear();

        check(&point_to_values, values.as_slice());
    }

    #[test]
    fn test_immutable_point_to_values_direct_access_enable_remove() {
        let check = |point_to_values: &ImmutablePointToValues<_>, values: &[Vec<_>]| {
            for (idx, values) in values.iter().enumerate() {
                let values_vec: Option<Vec<_>> = point_to_values
                    .get_values(idx as PointOffsetType)
                    .map(|i| i.copied().collect());
                assert_eq!(values_vec, Some(values.clone()),);
            }
        };

        // Test normal situation
        let mut values = vec![
            vec![0, 1, 2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![0, 1, 2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![10, 11, 12],
            vec![],
            vec![13],
            vec![14, 15],
        ];

        let mut point_to_values = ImmutablePointToValues::new(values.clone());
        assert_eq!(point_to_values.direct_access_container_enabled, false);
        assert_eq!(point_to_values.values_container_access.len(), 0);
        assert_ne!(point_to_values.point_to_values.len(), 0);

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(0);
        values[0].clear();

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(3);
        values[3].clear();

        check(&point_to_values, values.as_slice());

        // Test only one value situation
        let mut values = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![5],
            vec![10],
            vec![11],
            vec![13],
            vec![14],
        ];

        let mut point_to_values = ImmutablePointToValues::new(values.clone());
        assert_eq!(point_to_values.direct_access_container_enabled, true);
        assert_ne!(point_to_values.values_container_access.len(), 0);
        assert_eq!(point_to_values.point_to_values.len(), 0);

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(0);
        values[0].clear();

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(3);
        values[3].clear();

        check(&point_to_values, values.as_slice());

        // Test at most one value situation
        let mut values = vec![
            vec![0],
            vec![],
            vec![2],
            vec![5],
            vec![10],
            vec![11],
            vec![13],
            vec![14],
        ];

        let mut point_to_values = ImmutablePointToValues::new(values.clone());
        assert_eq!(point_to_values.direct_access_container_enabled, true);
        assert_ne!(point_to_values.values_container_access.len(), 0);
        assert_eq!(point_to_values.point_to_values.len(), 0);

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(0);
        values[0].clear();

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(3);
        values[3].clear();

        check(&point_to_values, values.as_slice());

        // Test no values ratio bigger than the threshold we set situation
        let mut values = vec![
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![11],
            vec![13],
            vec![14],
        ];

        let mut point_to_values = ImmutablePointToValues::new(values.clone());
        assert_eq!(point_to_values.direct_access_container_enabled, false);
        assert_ne!(point_to_values.point_to_values.len(), 0);
        assert_eq!(point_to_values.values_container_access.len(), 0);

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(0);
        values[0].clear();

        check(&point_to_values, values.as_slice());

        point_to_values.remove_point(3);
        values[3].clear();

        check(&point_to_values, values.as_slice());
    }
}
