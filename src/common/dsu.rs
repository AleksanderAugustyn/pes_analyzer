//! Disjoint-set union with union-by-rank and path halving.
//!
//! Stores `parent: Vec<u32>` and `rank: Vec<u8>`. Compact (1 byte) rank is
//! safe because tree height is bounded by ~log2(n) and rank-by-union grows
//! the rank counter by at most 1 per merge.

pub struct DisjointSetUnion {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DisjointSetUnion {
    pub fn new(n: usize) -> Self {
        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];
        Self { parent, rank }
    }

    /// Find the representative of `x`, applying path halving.
    pub fn find(&mut self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize];
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
    }

    /// Union the sets containing `x` and `y`. Returns `true` if a merge
    /// happened, `false` if they were already in the same set.
    pub fn union(&mut self, x: u32, y: u32) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        let (small, large) = if self.rank[rx as usize] < self.rank[ry as usize] {
            (rx, ry)
        } else {
            (ry, rx)
        };
        self.parent[small as usize] = large;
        if self.rank[small as usize] == self.rank[large as usize] {
            self.rank[large as usize] = self.rank[large as usize].saturating_add(1);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_singletons() {
        let mut d = DisjointSetUnion::new(5);
        for i in 0..5u32 {
            assert_eq!(d.find(i), i);
        }
    }

    #[test]
    fn union_merges_two_singletons() {
        let mut d = DisjointSetUnion::new(5);
        d.union(0, 1);
        assert_eq!(d.find(0), d.find(1));
        assert_ne!(d.find(0), d.find(2));
    }

    #[test]
    fn union_is_idempotent() {
        let mut d = DisjointSetUnion::new(3);
        d.union(0, 1);
        d.union(0, 1);
        d.union(1, 0);
        assert_eq!(d.find(0), d.find(1));
    }

    #[test]
    fn union_chains_connect_transitively() {
        let mut d = DisjointSetUnion::new(10);
        for i in 0..9 {
            d.union(i, i + 1);
        }
        let root = d.find(0);
        for i in 1..10u32 {
            assert_eq!(d.find(i), root);
        }
    }

    #[test]
    fn union_returns_true_on_merge_false_on_same_root() {
        let mut d = DisjointSetUnion::new(4);
        assert!(d.union(0, 1));
        assert!(!d.union(0, 1));
        assert!(d.union(2, 3));
        assert!(d.union(1, 3));
        assert!(!d.union(0, 2));
    }
}
