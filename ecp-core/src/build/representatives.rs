use ndarray::Array1;
use rand::seq::index::sample;

/// How to pick which items become cluster leaders. `"custom"` (the caller
/// already has their own leader ids/embeddings) needs no algorithm here -
/// it's handled by the builder directly.
pub enum RepresentativeStrategy {
    Offset,
    Random,
}

/// Picks `total_clusters` leader ids out of `0..total_items`.
pub fn select_representative_ids(
    total_items: usize,
    target_cluster_items: usize,
    strategy: RepresentativeStrategy,
) -> Array1<u32> {
    match strategy {
        RepresentativeStrategy::Offset => {
            (0..total_items as u32).step_by(target_cluster_items).collect()
        }
        RepresentativeStrategy::Random => {
            let total_clusters = total_items.div_ceil(target_cluster_items);
            let ids: Vec<u32> = sample(&mut rand::rng(), total_items, total_clusters)
                .into_iter()
                .map(|i| i as u32)
                .collect();
            Array1::from_vec(ids)
        }
    }
}

#[cfg(test)]
#[path = "utests/representatives.rs"]
mod tests;
