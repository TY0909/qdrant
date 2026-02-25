use common::types::PointOffsetType;

#[derive(Debug, Clone, Default)]
struct PointMeta<N: Default> {
    // `u32` is used instead of `usize` because it's more RAM efficient
    // We can expect that we will never have more than 4 billion values per segment
    offset: u32,
    length: u32,
    // For single values, we store them directly in the meta to eliminate an extra query.
    value: N,
}

// This structure doesn't support adding new values, only removing.
// It's used in immutable field indices like `ImmutableMapIndex`, `ImmutableNumericIndex`, etc to store points-to-values map.
#[derive(Debug, Clone, Default)]
pub struct ImmutablePointToValues<N: Default> {
    metas: Vec<PointMeta<N>>,
    // Flatten points-to-values map
    // It's an analogue of `Vec<Vec<N>>` but more RAM efficient because it stores values in a single Vec.
    multi_value_container: Vec<N>,
}

impl<N: Default> ImmutablePointToValues<N> {
    pub fn new(src: Vec<Vec<N>>) -> Self {
        let (multi_value_count, point_count) = src.iter().fold((0, 0), |(multi, point), values| {
            let length = values.len();
            match length {
                0 | 1 => (multi, point + 1),
                _ => (multi + length, point + 1),
            }
        });

        let mut metas = Vec::with_capacity(point_count);
        let mut multi_value_container = Vec::with_capacity(multi_value_count);

        for values in src {
            let length = values.len() as u32;
            match length {
                0 => {
                    metas.push(PointMeta {
                        offset: 0,
                        length,
                        value: N::default(),
                    });
                }
                1 => {
                    let mut values = values;
                    let value = values.pop().unwrap();
                    metas.push(PointMeta {
                        offset: 0,
                        length,
                        value,
                    });
                }
                _ => {
                    metas.push(PointMeta {
                        offset: multi_value_container.len() as u32,
                        length,
                        value: N::default(),
                    });
                    multi_value_container.extend(values);
                }
            }
        }

        Self {
            metas,
            multi_value_container,
        }
    }

    pub fn check_values_any(&self, idx: PointOffsetType, check_fn: impl FnMut(&N) -> bool) -> bool {
        let Some(meta) = self.metas.get(idx as usize) else {
            return false;
        };

        let vlen = meta.length as usize;

        match vlen {
            1 => std::iter::once(&meta.value).any(check_fn),
            // Since zero-length cases are uncommon, handling them here improves performance.
            _ => {
                let start = meta.offset as usize;
                self.multi_value_container
                    .get(start..start + vlen)
                    .map_or(false, |values| values.iter().any(check_fn))
            }
        }
    }

    pub fn get_values(&self, idx: PointOffsetType) -> Option<impl Iterator<Item = &N> + '_> {
        let meta = self.metas.get(idx as usize)?;

        match meta.length {
            0 => Some(std::slice::from_ref(&meta.value)[0..0].iter()),
            1 => Some(std::slice::from_ref(&meta.value).iter()),
            _ => self
                .multi_value_container
                .get(meta.offset as usize..(meta.offset + meta.length) as usize)
                .map(|s| s.iter()),
        }
    }

    pub fn get_values_count(&self, idx: PointOffsetType) -> Option<usize> {
        self.metas
            .get(idx as usize)
            .map(|meta| meta.length as usize)
    }

    pub fn remove_point(&mut self, idx: PointOffsetType) -> Vec<N> {
        let Some(meta) = self.metas.get_mut(idx as usize) else {
            return Default::default();
        };

        let offset = meta.offset;
        let length = meta.length;

        match length {
            0 => Default::default(),
            1 => {
                meta.length = Default::default();
                vec![std::mem::take(&mut meta.value)]
            }
            _ => {
                meta.length = Default::default();
                let mut result = Vec::with_capacity(length as usize);
                for value_index in offset as usize..(offset + length) as usize {
                    result.push(std::mem::take(&mut self.multi_value_container[value_index]));
                }
                result
            }
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
        point_to_values.remove_point(6);
        let removed_values: Vec<_> = point_to_values.get_values(6).unwrap().copied().collect();
        assert!(removed_values.is_empty());
        assert_eq!(Some(0), point_to_values.get_values_count(6));
    }
}
