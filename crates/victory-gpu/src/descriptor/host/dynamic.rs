//! Reusable contiguous ranges in the dynamic part of a view heap.

use std::collections::BTreeMap;

pub(super) struct DynamicSlots {
    limit: u32,
    cursor: u32,
    available: u32,
    /// Smallest failed request since the free ranges last grew.
    too_long: u32,
    /// Free ranges by inclusive start, with exclusive ends. Neighbors coalesce.
    ranges: BTreeMap<u32, u32>,
}

impl DynamicSlots {
    pub(super) fn new(limit: u32) -> Self {
        let mut ranges = BTreeMap::new();
        if limit != 0 {
            ranges.insert(0, limit);
        }
        Self {
            limit,
            cursor: 0,
            available: limit,
            too_long: u32::MAX,
            ranges,
        }
    }

    pub(super) fn take(&mut self, count: u32) -> Option<u32> {
        if count == 0 || count > self.available || count >= self.too_long {
            return None;
        }
        let cursor = self.cursor;
        let from_cursor = self
            .ranges
            .range(..=cursor)
            .next_back()
            .and_then(|(&start, &end)| {
                (end.saturating_sub(cursor) >= count).then_some((start, end, cursor))
            });
        let fits = |&(&start, &end): &(&u32, &u32)| end - start >= count;
        let found = from_cursor.or_else(|| {
            let (&start, &end) = self
                .ranges
                .range(cursor..)
                .find(fits)
                .or_else(|| self.ranges.range(..cursor).find(fits))?;
            Some((start, end, start))
        });
        let Some((start, end, first)) = found else {
            self.too_long = count;
            return None;
        };
        self.ranges.remove(&start);
        if start < first {
            self.ranges.insert(start, first);
        }
        let next = first + count;
        if next < end {
            self.ranges.insert(next, end);
        }
        self.available -= count;
        self.cursor = next % self.limit;
        Some(first)
    }

    #[cfg(feature = "diagnostics")]
    pub(super) fn is_taken(&self, slot: u32) -> Option<bool> {
        (slot < self.limit).then(|| {
            self.ranges
                .range(..=slot)
                .next_back()
                .is_none_or(|(_, &end)| end <= slot)
        })
    }

    pub(super) fn release(&mut self, first: u32, count: u32) {
        let end = first.saturating_add(count).min(self.limit);
        if first >= end {
            return;
        }
        let start = self
            .ranges
            .range(..=first)
            .next_back()
            .filter(|&(_, &end)| end >= first)
            .map_or(first, |(&start, _)| start);
        let mut merged_end = end;
        let mut already_free = 0;
        while let Some((&start, &end)) = self.ranges.range(start..=merged_end).next() {
            self.ranges.remove(&start);
            merged_end = merged_end.max(end);
            already_free += end - start;
        }
        let given = merged_end - start - already_free;
        self.ranges.insert(start, merged_end);
        self.available += given;
        if given != 0 {
            self.too_long = u32::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DynamicSlots;

    #[test]
    fn free_ranges_match_slot_scanning_in_a_fragmented_heap() {
        const LIMIT: u32 = 511;
        let mut ranges = DynamicSlots::new(LIMIT);
        let mut taken = [false; LIMIT as usize];
        let mut cursor = 0;
        let mut seed = 17u32;
        let mut live = Vec::new();
        for _ in 0..20_000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if seed >> 30 == 0 && !live.is_empty() {
                let at = (seed >> 8) as usize % live.len();
                let (first, count) = live.swap_remove(at);
                ranges.release(first, count);
                taken[first as usize..(first + count) as usize].fill(false);
            } else {
                let count = (seed >> 18) % 31 + 1;
                let expected = (cursor..LIMIT).chain(0..cursor).find(|&first| {
                    let end = first + count;
                    end <= LIMIT && taken[first as usize..end as usize].iter().all(|&s| !s)
                });
                assert_eq!(
                    ranges.take(count),
                    expected,
                    "cursor {cursor}, count {count}"
                );
                if let Some(first) = expected {
                    taken[first as usize..(first + count) as usize].fill(true);
                    live.push((first, count));
                    cursor = (first + count) % LIMIT;
                }
            }
        }
        for (first, count) in live {
            ranges.release(first, count);
        }
        assert_eq!(ranges.take(LIMIT), Some(0));
    }

    #[test]
    fn partial_and_overlapping_releases_merge_without_creating_free_slots() {
        let mut slots = DynamicSlots::new(16);
        assert_eq!(slots.take(16), Some(0));
        slots.release(2, 2);
        slots.release(6, 2);
        assert_eq!(slots.take(3), None);
        slots.release(3, 4);
        slots.release(4, 1);
        assert_eq!(slots.take(7), None);
        assert_eq!(slots.take(6), Some(2));
        assert_eq!(slots.take(1), None);
    }
}
