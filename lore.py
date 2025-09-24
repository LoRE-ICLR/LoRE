# Core components for LoRE-style batching, a cosine-based loss, a TransE step,
# and a lightweight attention/pooling layer.
# The goal of the comments below is to clarify *what* each piece expects and *why*
# certain masks/operations are applied.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence


class LoreBatch:
    """
    Container for one mini-batch of neighborhood-structured inputs.

    Parameters
    ----------
    node_idx_top : list[int] or None
        Indices of the "top" (center) nodes for the batch.
        If None, the model should consume neighbor indices directly.
    relation_idx_top : list[int] or None
        Relation indices aligned with node_idx_top; used to lookup relation embeddings
        for each neighborhood row. If None, relations are taken directly from neigh_rels.
    neigh_nodes : list[list[int]]
        For each top node, a (possibly variable-length) list of neighbor node indices.
    neigh_rels : list[list[int]]
        For each top node, a list of relation indices aligned with neigh_nodes.
    inverse : list[list[bool]]
        For each top node, a boolean list aligned with neigh_rels indicating whether
        the relation is in inverse direction (True: use +r, False: use -r).

    After calling `.to(device)`, the *_T attributes contain padded tensors suitable
    for batched GPU computation. Padding is to the max neighborhood length in the batch.
    """

    def __init__(self, node_idx_top, relation_idx_top, neigh_nodes, neigh_rels, inverse):
        # Raw (python) lists as provided by the dataloader:
        self.node_idx_top = node_idx_top
        self.relation_idx_top = relation_idx_top
        self.neigh_nodes = neigh_nodes
        self.neigh_rels = neigh_rels
        self.inverse = inverse

        # Tensorized / padded counterparts (filled by .to(device)):
        self.node_idx_top_T = None
        self.relation_idx_top_T = None
        self.neigh_nodes_T = None
        self.neigh_rels_T = None
        self.inverse_T = None
        self.mask_T = None  # True at padded positions (useful for attention masking)

    def to(self, device):
        """
        Move lists to the given device and pad variable-length neighborhoods.

        Notes
        -----
        - pad_sequence(..., batch_first=True) yields shape [B, L_max] for each field.
        - mask_T is True at padded positions, False at valid tokens.
        """
        # Convert top-level node / relation indices if they exist.
        if self.node_idx_top:
            self.node_idx_top_T = torch.tensor(self.node_idx_top, device=device)
        if self.relation_idx_top:
            self.relation_idx_top_T = torch.tensor(self.relation_idx_top, device=device)

        # Pad neighbor node and relation indices to common length across batch.
        self.neigh_nodes_T = pad_sequence(
            [torch.tensor(x, device=device) for x in self.neigh_nodes],
            batch_first=True
        )
        self.neigh_rels_T = pad_sequence(
            [torch.tensor(x, device=device) for x in self.neigh_rels],
            batch_first=True
        )

        # Boolean inverse flags, padded with False (non-inverse by default).
        self.inverse_T = pad_sequence(
            [torch.tensor(x, device=device, dtype=torch.bool) for x in self.inverse],
            batch_first=True, padding_value=False
        )

        # Build a padding mask: False for real entries, True for padded positions.
        # Here we create per-row zero-vectors of length len(neigh), then pad with True.
        batch_masks = [torch.zeros(len(x), dtype=torch.bool, device=device) for x in self.neigh_nodes]
        self.mask_T = pad_sequence(batch_masks, batch_first=True, padding_value=True)


class LoreCos:
    """
    Cosine-based loss around a movable 'anchor' between positive and negative targets.

    Idea
    ----
    We define an anchor as a convex combination between negative and positive targets:
        anchor = neg + hook * (pos - neg)
    We then measure cosine distance between (prediction - anchor) and (pos - anchor),
    optionally applying a margin.

    Parameters
    ----------
    margin : float
        If > 0, apply hinge on (distance - margin).
    """

    def __init__(self, margin=0):
        self.margin = margin

    def compute(self, prediction, target_pos_embeddings, target_neg_embeddings, hook=0.25):
        """
        Parameters
        ----------
        prediction : Tensor [B, D]
        target_pos_embeddings : Tensor [B, D]
        target_neg_embeddings : Tensor [B, D]
        hook : float in [0, 1]
            Interpolation factor: 0 -> anchor at neg, 1 -> anchor at pos.

        Returns
        -------
        Tensor [B]
            Per-sample distances (hinged if margin > 0).
        """
        # Interpolate anchor between negative and positive targets.
        anchor = target_neg_embeddings + (target_pos_embeddings - target_neg_embeddings) * hook

        # Cosine distance between the anchored prediction and anchored positive target.
        distance = 1 - F.cosine_similarity(
            prediction - anchor, target_pos_embeddings - anchor, dim=1
        )

        # Optional margin (hinge) to encourage a minimum separation.
        if self.margin:
            return torch.clamp(distance - self.margin, min=0)
        else:
            return distance


class TransE(nn.Module):
    """
    Minimal TransE-style composition on neighborhoods:
        prediction = e(node) (+/-) r(edge)

    Notes
    -----
    - If batch.node_idx_top is provided, we first index entity embeddings by top nodes
      and then gather neighbor rows from that tensor. Otherwise we index by neigh_nodes directly.
    - Relation inversion is handled via a boolean mask: False => use -r, True => use +r.
    """

    def __init__(self, entity_embeddings, relation_embeddings):
        super().__init__()
        self.entity_embeddings = entity_embeddings  # nn.Embedding for entities
        self.relation_embeddings = relation_embeddings  # nn.Embedding for relations

    def get_relation_embeddings(self):
        """Return relation embedding weights as a (CPU) Python list for inspection/export."""
        return self.relation_embeddings.weight.data.detach().cpu().numpy().tolist()

    def forward(self, batch, dropout=None):
        """
        Parameters
        ----------
        batch : LoreBatch
            Must have *_T tensor fields prepared via batch.to(device).
        dropout : callable or None
            Optional dropout module to apply to embeddings.

        Returns
        -------
        Tensor [B, L_max, D]
            Per-neighbor predictions for each top node.
        """
        # Entity embeddings aligned to neighbor indices.
        if batch.node_idx_top is not None:
            # Lookup top-node embeddings, then gather rows for each neighbor index.
            # Shape: [B, D] -> index with [B, L_max] => [B, L_max, D]
            node_embeddings = self.entity_embeddings(batch.node_idx_top_T)[batch.neigh_nodes_T]
        else:
            # Direct lookup by neighbor indices (when no explicit top set is used).
            node_embeddings = self.entity_embeddings(batch.neigh_nodes_T)

        # Relation embeddings aligned to neighbor relations.
        if batch.relation_idx_top is not None:
            relation_embeddings = self.relation_embeddings(batch.relation_idx_top_T)[batch.neigh_rels_T]
        else:
            relation_embeddings = self.relation_embeddings(batch.neigh_rels_T)

        # Optional regularization.
        if dropout is not None:
            node_embeddings = dropout(node_embeddings)
            relation_embeddings = dropout(relation_embeddings)

        # Apply inverse flag: True -> +r, False -> -r.
        if batch.inverse_T is not None:
            mask = batch.inverse_T.unsqueeze(-1)  # [B, L_max, 1] for broadcasting
            relation_embeddings = torch.where(mask, relation_embeddings, -relation_embeddings)

        # TransE composition.
        prediction = node_embeddings + relation_embeddings
        return prediction


class BatchLoreAttentionLayer(nn.Module):
    """
    Lightweight self-attention over per-node neighborhoods with padding support.

    forward(embeddings, padding_mask)
    ---------------------------------
    embeddings : Tensor [B, L, D]
        Per-batch neighborhood embeddings.
    padding_mask : Bool Tensor [B, L] or None
        True at padded positions (to be ignored).

    Returns
    -------
    Tensor [B, D]
        Mask-aware pooled representation (tanh-projected).
    """

    def __init__(self, embed_dim):
        super().__init__()
        # Single-head attention: shared dimensions for Q and K.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        # Xavier for linear layers; zero biases.
        for proj in [self.q_proj, self.k_proj]:
            init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                init.zeros_(proj.bias)

    def forward(self, embeddings, padding_mask=None):
        """
        Compute attention weights and produce a masked, mean-like pooled vector.

        Steps
        -----
        1) Project to Q, K.
        2) Scaled dot-product attention scores.
        3) Mask out padded tokens (if provided).
        4) Softmax -> attention weights.
        5) Weighted sum => attended sequence.
        6) Average only over valid tokens (if masked), else simple mean.
        7) Nonlinear squashing with tanh.
        """
        # Q, K: [B, L, D]
        Q = self.q_proj(embeddings)
        K = self.k_proj(embeddings)

        # Scaled dot-product scores: [B, L, L]
        d = embeddings.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)

        # Mask padded positions in the *keys* dimension: they should receive -inf score.
        if padding_mask is not None:
            # Expand mask to [B, L_q, L_k]
            mask = padding_mask.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            scores = scores.masked_fill(mask, float('-inf'))

        # Attention weights over neighbors.
        attn_weights = F.softmax(scores, dim=-1)  # [B, L, L]
        attended = torch.matmul(attn_weights, embeddings)  # [B, L, D]

        # Pooling across the L dimension (neighborhood length):
        if padding_mask is not None:
            # Zero out padded vectors before summing.
            valid_mask = ~padding_mask  # True where tokens are real
            attended = attended * valid_mask.unsqueeze(-1)  # [B, L, D]

            # Sum and divide by count of valid tokens (avoid div-by-zero).
            summed = attended.sum(dim=1)                               # [B, D]
            counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            pooled = summed / counts
        else:
            pooled = attended.mean(dim=1)  # [B, D]

        # Optional non-linearity to bound outputs and add capacity.
        return torch.tanh(pooled)
