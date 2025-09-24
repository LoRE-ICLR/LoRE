# LoRE manager: wraps the LoreGraph, embeddings, training loop, reconstruction,
# and on-the-fly KG updates. Comments explain design choices and non-obvious steps.

from collections import deque, defaultdict
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from lore import TransE, LoreCos, BatchLoreAttentionLayer, LoreBatch
from graph import LoreGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from torch import optim
from utils import LoreIterableDataset


class TrainingInfo:
    """
    Bookkeeping for a (re)training run.

    Attributes
    ----------
    batch_specs : dict or None
        Parameters passed to the iterable dataset to shape batches (sampling limits etc.).
    base_nodes : list[str] or None
        If given, limit training to subgraphs centered around these item ids.
    epochs : int
        Number of passes over base_nodes (via streaming subgraph sampling).
    exponent : float
        Node-weight exponent; higher favors high-degree nodes in sampling.
    node_counter : dict[item -> int]
        How many times a node appeared as a center in training so far.
    """

    def __init__(self, batch_specs=None, base_nodes=None, epochs=1, exponent=2):

        self.batch_specs = batch_specs
        self.base_nodes = base_nodes
        self.epochs = epochs
        self.exponent = exponent
        self.start_time = datetime.now()
        self.end_time = None
        self.finalized = False
        self.epoch_counter = 0
        self.node_counter = defaultdict(int)

    def finalize(self):
        """Mark training as finished and stamp an end time."""
        self.finalized = True
        self.end_time = datetime.now()


class LoreManager(nn.Module):
    """
    Orchestrates the LoRE pipeline:
      - maintains a LoreGraph
      - holds entity/relation embeddings and attention module
      - provides training/reconstruction/update utilities
    """

    def __init__(self, device, graph_data, lore_config={}):
        super().__init__()
        self.device = device
        self.graph = LoreGraph(graph_data=graph_data)

        self.training_counter = defaultdict(int)
        self.lore_config = lore_config

        # Embedding dimension
        self.dim = lore_config.get('dim', 256)

        # Loss setup (LoreCos supports optional margin)
        self.loss_margin = lore_config.get('margin', None)
        self.loss = LoreCos(self.loss_margin)

        # Regularization and buffers
        self.drop_rate = lore_config.get('drop', 0.2)
        self.dropout = nn.Dropout(p=self.drop_rate).to(device)
        self.buffer = lore_config.get('buffer', 1000)  # capacity to grow entity table without reallocations

        # Embedding tables
        self.entity_embeddings = nn.Embedding(
            len(self.graph.node_mapping.items()) + self.buffer, self.dim)

        self.relation_embeddings = nn.Embedding(
            len(self.graph.relation_mapping.items()), self.dim)

        # Init: entities ~ N(0,1), relations ~ Xavier
        nn.init.normal_(self.entity_embeddings.weight, mean=0.0, std=1.0)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Composition module and neighborhood attention
        self.transe = TransE(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        ).to(device)
        self.attention = BatchLoreAttentionLayer(embed_dim=self.dim).to(device)

        # Move everything to device and normalize entity vectors to unit L2
        self.to(self.device)
        self.normalize_embeddings()

        # Optimizer setup
        self.lr = lore_config.get('lr', 0.00005)
        self.weight_decay = lore_config.get('weight_decay', 0.00001)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.init_time = datetime.now()

    def get_base(self):
        """
        Export current node/relation bases (URIs + embeddings).
        Relations are taken from the TransE module to reflect any decoupling.
        """
        return {
            'nodes': {
                'uris': list(self.graph.node_mapping.items()),
                'embeddings': self.entity_embeddings.weight.data.detach().cpu().numpy().tolist()
            },
            'relations': {
                'uris': list(self.graph.relation_mapping.items()),
                'embeddings': self.transe.get_relation_embeddings()
            }
        }

    def get_config(self):
        """Return current training/architecture hyperparameters."""
        return {
            "dim": self.dim,
            "margin": self.loss_margin,
            "drop": self.drop_rate,
            "buffer": self.buffer,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }

    def update_kg(self, nodes, relations, updates):
        """
        On-the-fly KG updates: add/remove edges and (optionally) initialize embeddings for new nodes.

        Parameters
        ----------
        nodes : list[item]
            Items corresponding to the indices inside updates['add'] / updates['remove'].
        relations : list[item]
            Relation items corresponding to the indices inside updates.
        updates : dict
            { "add": [(s_i, o_i, p_i), ...], "remove": [(s_i, o_i, p_i), ...] }
            The integers refer to positions inside `nodes` / `relations`.

        Notes
        -----
        - New nodes get initialized via reconstruction from their neighborhoods.
        - Entity table extends by `buffer` rows if capacity is exceeded, without re-init of existing rows.
        """

        # Sanity check: all relations must already exist
        for relation in relations:
            if relation not in self.graph.relation_mapping.items():
                raise Exception(f"Relation {relation} is unknown!")

        # Identify genuinely new nodes (not yet in mapping)
        new_nodes = []
        for node in nodes:
            if node not in self.graph.node_mapping.items():
                # Make sure we don't attempt to remove edges touching unknown nodes
                for s, o, _ in updates["remove"]:
                    s = nodes[s]
                    o = nodes[o]
                    if s == node or o == node:
                        raise Exception(f"Tried to remove edge with unknown node {node}!")
                new_nodes.append(node)

        # Apply removals
        removals = []
        for s, o, p in updates["remove"]:
            s = nodes[s]; o = nodes[o]; p = relations[p]
            removals.append(self.graph.remove_edge((s, o, p)))

        # Apply additions
        additions = []
        for s, o, p in updates["add"]:
            s = nodes[s]; o = nodes[o]; p = relations[p]
            additions.append(self.graph.add_edge((s, o, p)))

        # Rebuild direction-aware edge cache for touched nodes
        self.graph.set_edges_rs(node_idxs=[self.graph.node_mapping.get_idx(x) for x in nodes])

        len_new = len(new_nodes)
        print(f"Removed {sum(removals)}/{len(removals)} edges. Added {sum(additions)}/{len(additions)} edges.")

        # Initialize embeddings for *new* nodes (if any)
        if len_new > 0:

            len_emb = self.entity_embeddings.weight.shape[0]
            len_graph = len(self.graph.node_mapping.items())
            overflow = len_graph - len_emb

            if overflow > 0:
                print("Update overflow: Extending entity embeddings..")
                self.extend_entity_embeddings(num_new_rows=overflow + self.buffer)
                print("Entity embeddings successfully extended!")

            # Reconstruct embeddings from neighborhoods for new nodes
            new_embedding_idxs = [self.graph.node_mapping.get_idx(x) for x in new_nodes]
            initial_embeddings = self.reconstruct(reconstruct_items=new_nodes)

            with torch.no_grad():
                indices = torch.tensor(new_embedding_idxs, device=self.device)
                self.entity_embeddings.weight[indices] = initial_embeddings

            print(f"\nInitialized new embeddings: {len(new_embedding_idxs)}.")

        return additions, removals

    def extend_entity_embeddings(self, num_new_rows: int):
        """
        Extend entity embedding table by `num_new_rows` with normalized random vectors.

        Preserves:
        - existing weights
        - existing gradients (if present)
        - optimizer state (by adding a new param group pointing to the extended weight)
        """
        old_weight = self.entity_embeddings.weight.data
        old_grad = self.entity_embeddings.weight.grad.detach().clone() \
            if self.entity_embeddings.weight.grad is not None else None
        D = old_weight.shape[1]

        # New rows: unit-normalized random vectors
        new_rows = F.normalize((torch.randn(num_new_rows, D, device=self.device)), dim=1, p=2)
        new_weight = torch.cat([old_weight, new_rows], dim=0)

        # Update embedding module
        self.entity_embeddings.num_embeddings += num_new_rows
        with torch.no_grad():
            self.entity_embeddings.weight = nn.Parameter(new_weight)

        # Restore gradient buffer if it existed
        if old_grad is not None:
            new_grad = torch.zeros_like(new_weight)
            new_grad[:old_grad.shape[0]] = old_grad
            self.entity_embeddings.weight.grad = new_grad

        # Keep TransE module in sync
        self.transe.entity_embeddings = self.entity_embeddings

        # Let optimizer track the extended parameter
        self.optimizer.add_param_group({'params': [self.entity_embeddings.weight]})

    def reconstruct(self, reconstruct_items=None, batch_size=16):
        """
        Compute LoRE reconstructions for specified nodes.

        If reconstruct_items is None: reconstruct for the entire graph in index order.
        Returns a tensor of shape [num_items, dim].
        """
        with torch.no_grad():
            if reconstruct_items is None:
                idxs = list(self.graph.node_mapping.idxs())
            else:
                idxs = [self.graph.node_mapping.get_idx(x) for x in reconstruct_items]

            total_nodes = len(idxs)
            all_outputs = []
            num_batches = ceil(len(idxs) / batch_size)

            for b in range(num_batches):
                batch_indices = idxs[b * batch_size: (b + 1) * batch_size]
                processed = b * batch_size + len(batch_indices)

                print(
                    f"\rReconstructing Batch {b + 1}/{num_batches} "
                    f"({processed}/{total_nodes} nodes)...",
                    end='',
                    flush=True
                )

                reconstruction_batch = self.graph.reconstruction_batch(node_idxs=batch_indices, device=self.device)
                reconstruction_batch.to(self.device)

                # TransE over neighborhoods -> attention pooling
                geo_padded = self.transe(reconstruction_batch)
                lore_fp = self.attention(geo_padded, reconstruction_batch.mask_T)

                all_outputs.append(lore_fp)

            return torch.cat(all_outputs, dim=0)

    def what_if(self, neighbors, relations, inverse):
        """
        Construct a hypothetical neighborhood embedding:
        given lists of neighbors/relations/inverse flags, produce an attended vector.
        """
        neighbors = torch.tensor([self.graph.node_mapping.get_idx(x) for x in neighbors], device=self.device)
        relations = torch.tensor([self.graph.relation_mapping.get_idx(x) for x in relations], device=self.device)

        out = self.transe(neighbors, relations, torch.tensor(inverse, dtype=torch.bool, device=self.device))

        # Wrap into a single "sequence" for attention pooling
        batch_embeddings = [out]
        batch_masks = [torch.zeros(out.size(0), dtype=torch.bool, device=self.device)]

        padded = pad_sequence(batch_embeddings, batch_first=True)
        mask = pad_sequence(batch_masks, batch_first=True, padding_value=True)
        attended = self.attention(padded, mask)

        return attended

    def train_lore(self, training_info: TrainingInfo, num_workers=0):
        """
        Streaming training loop:
          - samples subgraphs around base nodes (or all nodes if None)
          - computes LoRE embeddings and self-supervised loss
          - yields printable progress lines for external logging

        Timing queues (last 100 iters) print: total, creation, forward, loss, backward times.
        """

        base_nodes = list(self.graph.node_mapping.items()) \
            if training_info.base_nodes is None else training_info.base_nodes

        base_idxs = [self.graph.node_mapping.get_idx(item) for item in base_nodes]
        epochs = training_info.epochs
        batch_specs = training_info.batch_specs
        exponent = training_info.exponent

        # Moving averages for logs
        recent_losses = deque(maxlen=100)
        recent_batch_sizes = deque(maxlen=100)

        full_times = deque(maxlen=100)
        batch_creation_times = deque(maxlen=100)
        forward_times = deque(maxlen=100)
        loss_times = deque(maxlen=100)
        backward_times = deque(maxlen=100)

        # Ensure neighbor caches are ready
        self.graph.set_edges_rs()
        self.train()

        for epoch in range(epochs):

            dataset = LoreIterableDataset(
                lore_graph=self.graph,
                node_idxs=base_idxs[:],
                node_weights={
                    idx: len(node) ** exponent
                    for idx, node in self.graph.graph_nodes.items()
                },
                batch_specs=batch_specs
            )

            dataloader = DataLoader(
                dataset,
                batch_size=None,         # iterable dataset yields already batched objects
                num_workers=num_workers,
                pin_memory=True
            )

            i = 0

            for idx, batch in dataloader:
                full_times_start = datetime.now()

                batch.to(self.device)
                batch_creation_times.append((datetime.now() - full_times_start).total_seconds())

                # Forward: TransE + attention
                forward_start = datetime.now()
                lore_fp = self(batch)
                forward_times.append((datetime.now() - forward_start).total_seconds())

                # Loss
                loss_start = datetime.now()
                loss = self.get_loss(lore_fp=lore_fp, batch=batch)
                loss_times.append((datetime.now() - loss_start).total_seconds())

                # Backward + step + renormalize entities (keeps them on unit sphere)
                backward_start = datetime.now()
                loss.backward()
                self.optimizer.step()
                self.normalize_embeddings()
                backward_times.append((datetime.now() - backward_start).total_seconds())

                # Logging stats
                loss_item = loss.item()
                recent_losses.append(loss_item)
                recent_loss = sum(recent_losses) / len(recent_losses)

                batch_size = len(batch.node_idx_top)
                recent_batch_sizes.append(batch_size)
                recent_batch_size = sum(recent_batch_sizes) / len(recent_batch_sizes)

                full_times.append((datetime.now() - full_times_start).total_seconds())
                recent_full_times = sum(full_times) / len(full_times)
                recent_batch_creation_times = sum(batch_creation_times) / len(batch_creation_times)
                recent_forward_times = sum(forward_times) / len(forward_times)
                recent_loss_times = sum(loss_times) / len(loss_times)
                recent_backward_times = sum(backward_times) / len(backward_times)

                # Console line (same string also yielded for external consumers)
                print(
                    f"\rEpoch {epoch + 1}/{epochs}: {i + 1}/{int(1.5 * len(base_nodes))}, "
                    f"Loss: {recent_loss:.5f}, "
                    f"Batch Size {recent_batch_size:.5f}, ",
                    f"Time: {recent_full_times:.3f} ({recent_batch_creation_times:.3f}, {recent_forward_times:.3f}, ",
                    f"{recent_loss_times:.3f}, {recent_backward_times:.3f})",
                    end='',
                    flush=True
                )

                for node in batch.node_idx_top:
                    training_info.node_counter[self.graph.node_mapping.get_item(node)] += 1

                yield (f"\rEpoch {epoch + 1}/{epochs}: {i + 1}/{int(1.5 * len(base_nodes))}, "
                       + f"Loss: {recent_loss:.5f}, "
                       + f"Batch Size {recent_batch_size:.5f}, "
                       + f"Time: {recent_full_times:.3f} ({recent_batch_creation_times:.3f}, {recent_forward_times:.3f}, "
                       + f"{recent_loss_times:.3f}, {recent_backward_times:.3f})")

                i += 1

            training_info.epoch_counter += 1

        training_info.finalize()

    def forward(self, batch: LoreBatch):
        """
        Model forward: TransE composition over neighborhoods + masked attention pooling.
        Returns a [B, D] tensor (one embedding per top node).
        """
        geo_padded = self.transe(batch, dropout=self.dropout)
        lore_fp = self.attention(geo_padded, batch.mask_T)
        return lore_fp

    def normalize_embeddings(self):
        """Keep entity vectors on the unit L2 sphere (helps stability with cosine-based objectives)."""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, dim=1, p=2)

    def get_loss(self, lore_fp, batch: LoreBatch, temperature=0.1, mean=True):
        """
        Self-supervised contrast via 'soft hard-negative' sampling within the batch.

        Steps
        -----
        - Compute pairwise distances among top-node entity embeddings.
        - Convert to logits and sample a negative index per anchor with a Categorical
          (lower distance -> higher probability due to negative sign).
        - Feed anchor (lore_fp), pos (entity embedding), neg (sampled) into LoreCos.

        Parameters
        ----------
        temperature : float
            Softer (higher) vs greedier (lower) negative sampling.
        mean : bool
            Return mean over batch or vector of per-sample losses.
        """
        # Degenerate case safeguard
        if lore_fp is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Positive/anchor embeddings (entity table, not the LoRE pooled outputs)
        node_embeddings = self.entity_embeddings(batch.node_idx_top_T)

        # Pairwise distances; inf on diagonal to avoid self as negative
        distance_matrix = torch.cdist(node_embeddings, node_embeddings, p=2)
        distance_matrix.fill_diagonal_(float('inf'))

        # Higher probability for closer (harder) negatives
        logits = -distance_matrix / temperature
        closest_indices = torch.distributions.Categorical(logits=logits).sample()

        raw_loss = self.loss.compute(
            prediction=lore_fp,
            target_pos_embeddings=node_embeddings,
            target_neg_embeddings=node_embeddings[closest_indices]
        )

        return raw_loss.mean() if mean else raw_loss

    def save_embedding(self, path, items, reconstruct=False):
        """
        Save embeddings for a list of items to a .npz file with fields:
          - 'embeddings' : float32 [N, D]
          - 'identifiers': object [N]

        If reconstruct=True, use LoRE reconstructions; else use raw entity table rows.
        """
        reconstruct_idxs = [self.graph.node_mapping.get_idx(item) for item in items]

        if reconstruct:
            ent_embeddings_numpy = self.reconstruct(reconstruct_items=items).detach().cpu().numpy()
            np.savez(path, embeddings=ent_embeddings_numpy, identifiers=np.array(items, dtype=object))
        else:
            ent_embeddings_numpy = self.entity_embeddings(
                torch.tensor(reconstruct_idxs, device=self.device)).detach().cpu().numpy()
            np.savez(path, embeddings=ent_embeddings_numpy, identifiers=np.array(items, dtype=object))
