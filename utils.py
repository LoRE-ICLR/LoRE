# Utilities: a bidirectional mapping (item <-> idx) and an IterableDataset
# that yields LoRE subgraph batches using weighted, slightly augmented sampling.
# Comments explain intent, invariants, and why the weighted shuffle/augmentation works.

import math
import random
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np


class BidirectionalMapping:
    """
    Simple 1-1 mapping between arbitrary hashable items and integer indices.

    Supports:
      - item -> idx and idx -> item lookups
      - insertion that assigns the next available idx
      - membership tests and basic introspection

    Notes
    -----
    - Indices are assigned densely starting at 0 in insertion order.
    - No deletion API (indices remain stable once assigned).
    """

    def __init__(self, items=None):
        if items is None:
            items = []
        # Build both directions in O(n)
        self.item2idx = {item: idx for idx, item in enumerate(items)}
        self.idx2item = {idx: item for idx, item in enumerate(items)}

    def get_idx(self, item):
        """Return index for `item`, or None if unknown."""
        return self.item2idx.get(item)

    def contains(self, item):
        """Legacy predicate; prefer `item in mapping` via __contains__."""
        if item in self.item2idx.keys():
            return True
        return False

    def get_item(self, index):
        """Return item for `index`, or None if unknown."""
        return self.idx2item.get(index)

    def items(self):
        """Iterable view over known items (keys of item2idx)."""
        return self.item2idx.keys()

    def idxs(self):
        """Iterable view over known indices (keys of idx2item)."""
        return self.idx2item.keys()

    def insert(self, item):
        """
        Insert `item` if missing and assign the next free index.
        Idempotent: returns immediately if item already present.
        """
        if item not in self.item2idx:
            next_idx = len(self.item2idx)
            self.item2idx[item] = next_idx
            self.idx2item[next_idx] = item

    def __len__(self):
        return len(self.item2idx)

    def __repr__(self):
        return f"BidirectionalMapping({len(self.item2idx)} items)"

    def __contains__(self, item):
        """Enable `item in mapping` checks."""
        return item in self.item2idx


class LoreIterableDataset(IterableDataset):
    """
    Iterable dataset that streams LoRE subgraph batches.

    For each chosen center node idx (with weighted/augmented ordering), it calls:
        lore_graph.training_batch(...)
    and yields (center_idx, LoreBatch).

    Parameters
    ----------
    lore_graph : LoreGraph
        Graph with sampling/batch-generation utilities.
    node_idxs : list[int]
        Candidate center-node indices (in *graph-index* space).
    node_weights : dict[int -> float]
        Positive weights used to bias which node centers are visited earlier/more often.
    batch_specs : dict
        Extra kwargs forwarded to lore_graph.training_batch (e.g., max_edges/max_nodes/...).

    Worker splitting
    ----------------
    If num_workers > 0, indices are partitioned deterministically by modulo so each
    worker processes a disjoint slice of the sequence.
    """

    def __init__(self, lore_graph, node_idxs, node_weights, batch_specs):
        self.lore_graph = lore_graph
        # Build an augmented ordering with interleaved re-samples to improve coverage
        self.node_idxs = create_augmented_list(node_idxs, [node_weights[x] for x in node_idxs])
        self.node_weights = node_weights
        self.batch_specs = batch_specs if batch_specs is not None else {}

    def __iter__(self):
        # Worker-aware sharding: deterministic split by index modulo num_workers.
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        local_indices = [idx for i, idx in enumerate(self.node_idxs) if i % num_workers == worker_id]

        # Produce (center_idx, LoreBatch) pairs lazily.
        for idx in local_indices:
            batch = self.lore_graph.training_batch(
                device="cpu",                 # batches are moved to target device later
                parent_start_idx=idx,
                node_weights=self.node_weights,
                **self.batch_specs
            )
            yield idx, batch


def weighted_shuffle(entries, weights):
    """
    Draw a random permutation where earlier positions are more likely for large weights.

    Method
    ------
    Keys ~ Exp(rate = w) sampling trick:
       For weight w > 0, key = -log(U) / w has an exponential distribution whose
       order statistics yield a weighted random order. w == 0 places item at +inf.

    Returns
    -------
    list
        `entries` reordered by ascending keys.
    """
    keys = [-np.log(random.random()) / w if w > 0 else float('inf') for w in weights]
    return [x for _, x in sorted(zip(keys, entries))]


def create_augmented_list(entries, weights):
    """
    Create a length-(N + ceil(N/2)) sequence that interleaves:
      1) a weighted shuffle of all entries, and
      2) ~N/2 additional draws (with replacement) inserted at random positions.

    Purpose
    -------
    - The weighted shuffle prioritizes higher-weight nodes earlier.
    - The extra sampled insertions increase the chance that high-importance nodes
      appear more than once in the stream without starving low-weight nodes.

    Parameters
    ----------
    entries : list
        Source items (e.g., node indices).
    weights : list[float]
        Positive weights aligned with entries.

    Returns
    -------
    list
        Augmented sequence of entries with size N + ceil(N/2).
    """
    num_entries = len(entries)
    assert len(weights) == num_entries, "Lists 'a' and 'b' must be the same length"

    # 1) Weighted random order of all entries (no replacement)
    shuffled = weighted_shuffle(entries, weights)

    # 2) Sample ~N/2 extra entries *with* replacement, still weight-biased
    sample_size = math.ceil(num_entries / 2)
    sampled = random.choices(entries, weights=weights, k=sample_size)

    # Plan random insertion positions into a stream of length N + sample_size
    insert_positions = sorted(random.sample(range(num_entries + sample_size), k=sample_size))
    result = []
    shuffled_index = 0
    sample_index = 0

    # Merge by walking through the target positions
    for i in range(num_entries + sample_size):
        if sample_index < sample_size and i == insert_positions[sample_index]:
            result.append(sampled[sample_index])
            sample_index += 1
        else:
            result.append(shuffled[shuffled_index])
            shuffled_index += 1

    return result
