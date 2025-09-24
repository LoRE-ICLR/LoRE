# Graph utilities for LoRE: node/edge containers, a main LoreGraph with
# add/remove/(de)serialize, random subgraph sampling, and batch builders.
# Comments focus on data structures, invariants, and why certain steps exist.

import random
import time
from collections import defaultdict
import copy
from datetime import datetime

from lore import LoreBatch
from utils import BidirectionalMapping
from enum import Enum


def default_graph_node_factory(key):
    """Factory for GraphNodeDefaultDict: creates a GraphNode with idx=key."""
    return GraphNode(key)


class DIR(Enum):
    """Edge direction from the *perspective of the current node*."""
    OUT = -1   # current -> neighbor
    IN = 1     # neighbor -> current
    SELF = 0   # self-loop

    def inv(self):
        """Inverse direction (OUT <-> IN; SELF stays SELF)."""
        return DIR(-self.value)


class STAT(Enum):
    """State machine for subgraph expansion nodes."""
    OK = 1       # can continue expanding
    DEPTH = 2    # reached depth limit (not directly used outside of logic)
    EMPTY = 3    # no more edges (unused final state here)
    KILL = 4     # flagged to be removed from active frontier
    KILLED = 5   # already removed


class GraphNode:
    """
    Local adjacency container for a single node id (idx).

    Internals
    ---------
    edges_out : dict[int -> set[int]]
        For each object node idx, the set of relation ids from self.idx -> object.
    edges_in : dict[int -> set[int]]
        For each subject node idx, the set of relation ids from subject -> self.idx.
        (Stored here as a convenience for symmetric removal/iteration.)
    edges_self : set[int]
        Relations for self-loops (self.idx -> self.idx).
    """

    def __init__(self, idx=None, edges_out=None, edges_in=None, edges_self=None):
        self.edges_out = defaultdict(set, edges_out or {})
        self.edges_in = defaultdict(set, edges_in or {})
        self.edges_self = set() if edges_self is None else edges_self
        self.idx = idx

    def __sub__(self, other):
        """
        Node-wise edge difference: returns a deep-copied node where edges present
        in `other` are removed from this node's adjacency.
        """
        if isinstance(other, GraphNode):
            new_graph_node = copy.deepcopy(self)

            # Remove overlapping relations on overlapping neighbor indices.
            for o_idx in new_graph_node.edges_out.keys() & other.edges_out.keys():
                new_graph_node.edges_out[o_idx].difference_update(other.edges_out[o_idx])
            for s_idx in new_graph_node.edges_in.keys() & other.edges_in.keys():
                new_graph_node.edges_in[s_idx].difference_update(other.edges_in[s_idx])

            new_graph_node.edges_self.difference_update(other.edges_self)
            return new_graph_node

        raise TypeError("Subtraction only supported between instances of GraphNode")

    def __add__(self, other):
        """
        Node-wise edge union: returns a deep-copied node where edges from `other`
        are added to this node's adjacency.
        """
        if isinstance(other, GraphNode):
            new_graph_node = copy.deepcopy(self)

            for o_idx in new_graph_node.edges_out.keys() & other.edges_out.keys():
                new_graph_node.edges_out[o_idx].update(other.edges_out[o_idx])
            for s_idx in new_graph_node.edges_in.keys() & other.edges_in.keys():
                new_graph_node.edges_in[s_idx].update(other.edges_in[s_idx])

            new_graph_node.edges_self.update(other.edges_self)
            return new_graph_node

        raise TypeError("Subtraction only supported between instances of GraphNode")

    def __len__(self):
        """Total number of incident relations (in + out + self)."""
        return self.len_out() + self.len_in() + self.len_self()

    def len_out(self):
        """Count of outgoing relations."""
        return sum(len(relations) for relations in self.edges_out.values())

    def len_in(self):
        """Count of incoming relations."""
        return sum(len(relations) for relations in self.edges_in.values())

    def len_self(self):
        """Count of self-loop relations."""
        return len(self.edges_self)

    def add_edge(self, edge):
        """
        Insert an edge touching this node.

        Parameters
        ----------
        edge : tuple[int, int, int]
            (s_idx, o_idx, p_idx) with s_idx = subject, o_idx = object, p_idx = relation id.
        """
        s_idx, o_idx, p_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            raise ValueError("Neither subject nor object are equivalent to graph node identifier!")
        elif s_idx != self.idx:
            # Edge points into this node: subject elsewhere, object == self.idx
            self.edges_in[s_idx].add(p_idx)
        elif o_idx != self.idx:
            # Edge points out of this node: subject == self.idx, object elsewhere
            self.edges_out[o_idx].add(p_idx)
        else:
            # Self-loop
            self.edges_self.add(p_idx)

    def check_exists(self, edge):
        """Return True iff (s,o,p) exists in this node's adjacency."""
        s_idx, o_idx, p_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            return False
        elif s_idx != self.idx or o_idx != self.idx:
            if s_idx != self.idx:
                if s_idx not in self.edges_in:
                    return False
                if p_idx not in self.edges_in[s_idx]:
                    return False
            else:
                if o_idx not in self.edges_out:
                    return False
                if p_idx not in self.edges_out[o_idx]:
                    return False
        else:
            if p_idx not in self.edges_self:
                return False
        return True

    def remove_edge(self, edge):
        """
        Remove edge (s,o,p) from this node's adjacency if present.

        Returns
        -------
        int
            1 if something was removed, 0 otherwise.
        """
        if not self.check_exists(edge):
            return 0

        s_idx, o_idx, p_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            raise ValueError("Neither subject nor object are equivalent to graph node identifier!")
        elif s_idx != self.idx or o_idx != self.idx:
            if s_idx != self.idx:
                try:
                    self.edges_in[s_idx].remove(p_idx)
                    if len(self.edges_in[s_idx]) == 0:
                        del self.edges_in[s_idx]
                except KeyError:
                    return 0
            else:
                try:
                    self.edges_out[o_idx].remove(p_idx)
                    if len(self.edges_out[o_idx]) == 0:
                        del self.edges_out[o_idx]
                except KeyError:
                    return 0
        else:
            try:
                self.edges_self.remove(p_idx)
            except KeyError:
                return 0

        return 1

    def iter(self, neighbour_idxs=None):
        """
        Iterate over outgoing and self-loop edges as triples (s_idx, o_idx, p_idx).
        Optionally filter to a set of neighbor indices.
        """
        for o_idx, p_idxs in self.edges_out.items():
            for p_idx in p_idxs:
                if neighbour_idxs and o_idx not in neighbour_idxs:
                    continue
                yield self.idx, o_idx, p_idx
        for p_idx in self.edges_self:
            yield self.idx, self.idx, p_idx


class GraphNodeDefaultDict(defaultdict):
    """defaultdict that passes the key into the default_factory when missing."""
    def __missing__(self, key):
        # Create a new value using the key
        self[key] = value = self.default_factory(key)
        return value


class LoreGraph:
    """
    Main graph with integer ids for nodes/relations and per-node adjacency.
    Provides:
      - add/remove edges with automatic id mapping
      - (de)serialization via populate()
      - random subgraph sampling for training
      - neighbor extraction and batch builders
    """

    def __init__(self, graph_data=None, super_graph=None):

        # Mappings: item<->idx for nodes/relations
        self.node_mapping = BidirectionalMapping()
        self.relation_mapping = BidirectionalMapping()

        # idx -> GraphNode; auto-creates nodes via default_graph_node_factory
        self.graph_nodes = GraphNodeDefaultDict(default_graph_node_factory)

        # optional back-reference (useful when subgraphs are derived from a parent graph)
        self.super_graph = super_graph

        self.num_edges = 0        # count of directed edge *records* stored in this graph
        self.edges_rs = {}        # cache of per-node neighbor->(rel,DIR) sets (see set_edges_rs)

        if graph_data:
            self.populate(graph_data=graph_data)

    def populate(self, graph_data):
        """Restore the graph from serialized json data with fields: nodes, relations, edges."""

        print(f'Loading Graph from serialized json data...')
        start = time.perf_counter()

        nodes = graph_data["nodes"]
        relations = graph_data["relations"]
        edges = graph_data["edges"]

        # Insert all items to stabilize indices
        for node in nodes:
            self.node_mapping.insert(node)
        for rel in relations:
            self.relation_mapping.insert(rel)

        num_edges = len(edges)

        # Restore edges using numeric indices (s_idx,o_idx,p_idx)
        for i, (s_idx, o_idx, p_idx) in enumerate(edges):
            if i % 1000 == 0 or i == len(edges) - 1:
                print(f"\rProgress: {i + 1}/{num_edges} edges inserted", end='', flush=True)
            self.add_edge((s_idx, o_idx, p_idx), item=False)

        # Precompute direction-aware neighbor sets
        self.set_edges_rs()

        print('\nGraph loaded within', f"{(time.perf_counter() - start):.3f}", 'seconds.')

    def set_edges_rs(self, node_idxs=None):
        """
        Precompute for each node a dict:
            neighbor_idx -> {(rel_id, DIR.IN|OUT|SELF), ...}
        Used by sampling/training to step efficiently across directed edges.
        """

        self.edges_rs = {}

        if node_idxs is not None:
            nodes = [node for idx, node in self.graph_nodes.items() if idx in node_idxs]
        else:
            nodes = self.graph_nodes.values()

        for node in nodes:

            if node_idxs is not None and node.idx not in node_idxs:
                continue

            # SELF edges: map self.idx -> {(p, SELF)}
            self_edges = (
                {node.idx: {(p_idx, DIR.SELF) for p_idx in node.edges_self}}
                if node.edges_self else {}
            )

            # IN edges: map subject -> {(p, IN)}
            in_edges = {
                s_idx: {(p_idx, DIR.IN) for p_idx in p_idxs}
                for s_idx, p_idxs in node.edges_in.items()
                if p_idxs
            }

            # OUT edges: map object -> {(p, OUT)}
            out_edges = {
                o_idx: {(p_idx, DIR.OUT) for p_idx in p_idxs}
                for o_idx, p_idxs in node.edges_out.items()
                if p_idxs
            }

            # Merge maps; values are sets of (rel,DIR)
            combined = defaultdict(set)
            for d in [self_edges, in_edges, out_edges]:
                for idx, edges in d.items():
                    combined[idx].update(edges)

            self.edges_rs[node.idx] = dict(combined)

    def add(self, nodes, relations, edges):
        """
        Add multiple items by *item id* (not indices).

        nodes/relations are lists of items; edges contain (s_i, o_i, p_i) where each
        index refers to position in the given nodes/relations lists.
        """
        for node in nodes:
            self.node_mapping.insert(node)
        for rel in relations:
            self.relation_mapping.insert(rel)

        node_idxs = [self.node_mapping.get_idx(node) for node in nodes]
        rel_idxs = [self.relation_mapping.get_idx(rel) for rel in relations]

        add_counter = 0
        for s_idx, o_idx, p_idx in edges:
            add_counter += self.add_edge(edge=(node_idxs[s_idx], node_idxs[o_idx], rel_idxs[p_idx]), item=False)

        return add_counter

    def add_edge(self, edge, item=True):
        """
        Add a directed edge (s, o, p) to the graph.

        Parameters
        ----------
        edge : tuple
            If item=True: (s_item, o_item, p_item)
            If item=False: (s_idx, o_idx, p_idx) are *already indices*.
        """
        if item:
            # Map items -> indices
            s, o, p = edge
            self.node_mapping.insert(s)
            self.node_mapping.insert(o)
            self.relation_mapping.insert(p)
            s_idx = self.node_mapping.get_idx(s)
            o_idx = self.node_mapping.get_idx(o)
            p_idx = self.relation_mapping.get_idx(p)
        else:
            s_idx, o_idx, p_idx = edge

        # Avoid duplicate (s,o,p) at storage-time (idempotent add)
        try:
            if p_idx in self.graph_nodes[s_idx].edges_out[o_idx]:
                return 0
        except KeyError:
            pass

        # Store edge in both incident GraphNodes for symmetric ops/removals.
        if s_idx == o_idx:
            self.graph_nodes[s_idx].add_edge(edge=(s_idx, o_idx, p_idx))
        else:
            self.graph_nodes[s_idx].add_edge(edge=(s_idx, o_idx, p_idx))
            self.graph_nodes[o_idx].add_edge(edge=(s_idx, o_idx, p_idx))

        self.num_edges += 1
        return 1

    def remove(self, nodes, relations, edges):
        """
        Remove multiple edges given *item ids*.

        Only edges whose mapped indices exist in the current graph are considered.
        """
        node_mapping = {i: self.node_mapping.get_idx(node) for i, node in enumerate(nodes)
                        if node in self.node_mapping.items()}
        rel_mapping = {i: self.relation_mapping.get_idx(rel) for i, rel in enumerate(relations)
                       if rel in self.relation_mapping.items()}

        remove_counter = 0
        for s_idx, o_idx, p_idx in edges:
            if s_idx in node_mapping and o_idx in node_mapping and p_idx in rel_mapping:
                remove_counter += self.remove_edge(
                    edge=(node_mapping[s_idx], node_mapping[o_idx], rel_mapping[p_idx]), item=False)

        return remove_counter

    def remove_edge(self, edge, item=True):
        """
        Remove a directed edge (s, o, p) from the graph.

        Returns
        -------
        int
            1 if removed, 0 if not found.
        """
        if item:
            # Map items -> indices; if any item is unknown, nothing to remove.
            s, o, p = edge
            if s not in self.node_mapping.items():
                return 0
            if o not in self.node_mapping.items():
                return 0
            if p not in self.relation_mapping.items():
                return 0
            s_idx = self.node_mapping.get_idx(s)
            p_idx = self.relation_mapping.get_idx(p)
            o_idx = self.node_mapping.get_idx(o)
        else:
            s_idx, o_idx, p_idx = edge

        if s_idx != o_idx:
            # Remove from both incident nodes; ensure consistent result
            out1 = self.graph_nodes[s_idx].remove_edge((s_idx, o_idx, p_idx))
            out2 = self.graph_nodes[o_idx].remove_edge((s_idx, o_idx, p_idx))
            if 1 == out1 == out2:
                out = 1
            elif 0 == out1 == out2:
                out = 0
            else:
                raise ValueError("Different removal results!")
        else:
            out = self.graph_nodes[s_idx].remove_edge((s_idx, o_idx, p_idx))

        self.num_edges -= out
        return out

    def __sub__(self, other):
        """Deep-copied graph with edges in `other` removed."""
        if isinstance(other, LoreGraph):
            new_lore_graph = copy.deepcopy(self)

            # Iterate edges of `other` and remove by *items* in the copy.
            for graph_node in other.graph_nodes.values():
                for s_idx, o_idx, p_idx in graph_node.iter():
                    new_lore_graph.remove_edge((
                        other.node_mapping.get_item(s_idx),
                        other.node_mapping.get_item(o_idx),
                        other.relation_mapping.get_item(p_idx)
                    ))

            return new_lore_graph

        raise TypeError("Subtraction only supported between instances of LoreGraph")

    def __add__(self, other):
        """Deep-copied graph with edges from `other` added."""
        if isinstance(other, LoreGraph):
            new_lore_graph = copy.deepcopy(self)

            for graph_node in other.graph_nodes.values():
                for s_idx, o_idx, p_idx in graph_node.iter():
                    new_lore_graph.add_edge((
                        other.node_mapping.get_item(s_idx),
                        other.node_mapping.get_item(o_idx),
                        other.relation_mapping.get_item(p_idx)
                    ))

            return new_lore_graph

        raise TypeError("Subtraction only supported between instances of LoreGraph")

    def get_random_subgraph(self,
                            parent_start_idx,
                            node_weights,
                            max_edges=256,
                            max_nodes=64,
                            max_degree=16,
                            max_depth=5,
                            spawn_rate=0.2):
        """
        Sample a random connected subgraph for training.

        Strategy (frontier expansion)
        -----------------------------
        - Maintain a dict of DepthNode objects tracking per-node candidate edges and depth.
        - Repeatedly:
            * pick a neighbor (weighted by node_weights) and a (rel,DIR)
            * add the corresponding edge to the subgraph
            * update frontier; occasionally "spawn" a new current node biased to shallower depth
        - Enforce limits on edges/nodes/degree/depth and prune nodes marked KILL.

        Returns
        -------
        LoreGraph
            A subgraph whose node_mapping contains a contiguous 0..|V_sub|-1 mapping.
        """

        if parent_start_idx not in self.node_mapping.idx2item.keys():
            raise Exception(f'Node with idx {parent_start_idx} is unknown!')

        class DepthNode:
            """
            Tracks expandable edges for a node along with depth and kill logic.

            edges: dict[neighbor_idx -> set[(rel_id, DIR)]]
            depth: shortest discovered distance from the starting node
            """

            def __init__(self, node, edges_rs, delta, depth=None, killed=None, inserted=None):

                start = datetime.now()
                self.depth = depth if depth is not None else max_depth + 1
                killed = killed if killed is not None else set()
                inserted = inserted if inserted is not None else set()
                keep = inserted - killed

                # Local copy of available edges to expand from this node.
                edges_rs = edges_rs[node.idx]
                self.edges = {idx: edges_rs[idx].copy() for idx in keep if idx in edges_rs}

                # If below degree budget, bring in more candidates.
                needed = max(0, max_degree - len(self.edges))
                if needed > 0:
                    candidates = [idx for idx in edges_rs.keys() if idx not in inserted | killed]
                    if candidates:
                        if len(candidates) > needed:
                            sample = random.sample(candidates, min(needed, len(candidates)))
                            for s in sample:
                                self.edges[int(s)] = edges_rs[int(s)].copy()
                        else:
                            for s in candidates:
                                self.edges[s] = edges_rs[s].copy()

                self.depth = depth if depth is not None else max_depth + 1
                self.edge_counter = 0
                self.idx = node.idx
                self._killed = False

                # bookkeeping: track construction time (debug/telemetry)
                end = datetime.now()
                delta.append(end - start)

            def __len__(self):
                """Number of candidate (neighbor, relation) pairs still available."""
                return sum([len(x) for x in self.edges.values()])

            def add(self, new_node):
                """
                When a neighbor joins the frontier elsewhere, ensure *this* node has
                a back-link to it with inverted direction semantics.
                """
                new_idx = new_node.idx
                if new_idx not in self.edges:
                    self.edges[new_idx] = {(rel, dir.inv()) for rel, dir in new_node.edges[self.idx]}
                    if new_node.depth + 1 < self.depth:
                        self.depth = new_node.depth + 1

            def status(self):
                """Return the current lifecycle status for pruning logic."""
                if self._killed:
                    return STAT.KILLED
                if len(self) <= 0 or self.edge_counter >= max_degree:
                    return STAT.KILL
                if self.depth >= max_depth:
                    return STAT.DEPTH
                return STAT.OK

            def kill(self, idx):
                """Remove neighbor idx from this frontier node; mark killed if self."""
                if idx == self.idx:
                    self._killed = True
                self.edges.pop(idx, None)

            def update(self, neighbour_idx, edge, depth, stepped=False):
                """
                Bookkeeping after taking a step via (neighbour_idx, edge).

                stepped=False means another node stepped *towards* us (invert direction);
                stepped=True means we stepped *from* this node.
                """
                if self.depth > depth + 1:
                    self.depth = depth + 1
                if neighbour_idx in self.edges:
                    rel, dir = edge
                    if stepped:
                        self.edges[neighbour_idx].discard((rel, dir))
                    else:
                        self.edges[neighbour_idx].discard((rel, dir.inv()))
                    if not self.edges[neighbour_idx]:
                        self.edges.pop(neighbour_idx)

                # Increment degree usage regardless; guard handled in status().
                if self.edge_counter == max_degree:
                    pass
                self.edge_counter += 1

            def keep_idxs(self, idxs):
                """Restrict candidate neighbors to a given set (e.g., after max_nodes hit)."""
                self.edges = {idx: edges for idx, edges in self.edges.items() if idx in idxs}

            def step(self):
                """
                Sample a neighbor and an edge (rel,DIR) to traverse.

                Neighbor sampling is weighted by node_weights to bias expansion.
                """
                candidates = [idx for idx in self.edges]
                weights = [node_weights[key] for key in candidates]
                neighbour_idx = random.choices(candidates, weights=weights, k=1)[0]
                return neighbour_idx, random.choice(list(self.edges[neighbour_idx]))

        # Initialize the output subgraph with the parent start node as item 0.
        subgraph = LoreGraph(super_graph=self)
        subgraph.node_mapping.insert(item=parent_start_idx)
        current_idx = parent_start_idx

        delta = []          # timing/telemetry
        killed = set()      # nodes removed from frontier
        inserted = {current_idx}  # nodes already in subgraph/frontier

        # Active frontier
        depth_nodes = {
            current_idx: DepthNode(node=self.graph_nodes[current_idx], depth=0, edges_rs=self.edges_rs, delta=delta,
                                   killed=killed)
        }

        def killer():
            """
            Remove nodes in KILL state from every frontier node's candidate list.
            Called whenever a frontier node reaches KILL.
            """
            while kill_list := [node for node in depth_nodes.values() if node.status() == STAT.KILL]:
                kill_me = kill_list[0]
                for idx, node in ((idx, node) for idx, node in depth_nodes.items() if idx not in killed):
                    node.kill(kill_me.idx)
                killed.add(kill_me.idx)

        # Expand until reaching edge budget or frontier is exhausted.
        while subgraph.num_edges < max_edges:

            dNode = depth_nodes[current_idx]
            neighbour_idx, edge = dNode.step()
            rel_idx, dir = edge

            # Materialize edge into the subgraph according to direction
            if dir == DIR.OUT:
                subgraph.add_edge((dNode.idx, neighbour_idx, rel_idx))
            elif dir == DIR.IN:
                subgraph.add_edge((neighbour_idx, dNode.idx, rel_idx))
            else:
                subgraph.add_edge((dNode.idx, dNode.idx, rel_idx))

            if neighbour_idx in depth_nodes:
                # Neighbor already on frontier: symmetrical updates
                nNode = depth_nodes[neighbour_idx]
                nNode.update(dNode.idx, edge, dNode.depth)
                dNode.update(neighbour_idx, edge, dNode.depth, stepped=True)
                if dNode.status() == STAT.KILL or nNode.status() == STAT.KILL:
                    killer()

            else:
                # New frontier node joins
                dNode.update(neighbour_idx, edge, dNode.depth, stepped=True)
                inserted.add(neighbour_idx)
                nNode = DepthNode(node=self.graph_nodes[neighbour_idx],
                                  depth=dNode.depth + 1,
                                  edges_rs=self.edges_rs,
                                  delta=delta,
                                  killed=killed,
                                  inserted=inserted)

                depth_nodes[neighbour_idx] = nNode
                nNode.update(dNode.idx, edge, dNode.depth)

                # Maintain back-links across existing frontier nodes
                for node_idx in nNode.edges:
                    if node_idx in depth_nodes and node_idx != neighbour_idx:
                        depth_nodes[node_idx].add(nNode)

                # Enforce node budget; restrict candidate neighbor sets to current frontier
                if len(depth_nodes) >= max_nodes:
                    for node in depth_nodes.values():
                        node.keep_idxs(depth_nodes.keys())
                    killer()
                else:
                    if dNode.status() == STAT.KILL or nNode.status() == STAT.KILL:
                        killer()

            # Decide next current node:
            # - If neighbor is not OK or random spawn triggers, pick a frontier node biased to shallower depth.
            if nNode.status() != STAT.OK or random.random() < spawn_rate:
                node_depths = defaultdict(set)
                for idx, node in depth_nodes.items():
                    if node.status() == STAT.OK:
                        node_depths[node.depth].add(idx)
                if len(node_depths) == 0:
                    break
                depth_candidates = list(node_depths.keys())
                depth_weights = [0.5 ** depth for depth in depth_candidates]  # bias to shallow
                selected_depth = random.choices(depth_candidates, weights=depth_weights, k=1)[0]
                current_idx = random.choice(list(node_depths[selected_depth]))
            else:
                current_idx = neighbour_idx

        return subgraph

    def get_neighbors(self, node_idxs=None):
        """
        Build (direction-aware) neighbor lists.

        Returns
        -------
        dict[int, list[tuple[int,int,DIR]]]
            node_idx -> list of (neighbor_idx, relation_idx, DIR)
            (SELF edges are not included here; training uses IN/OUT neighbors.)
        """
        neighbors = {}

        if node_idxs is not None:
            nodes = [node for idx, node in self.graph_nodes.items() if idx in node_idxs]
        else:
            nodes = self.graph_nodes.values()

        for node in nodes:

            if node_idxs is not None and node.idx not in node_idxs:
                continue

            in_edges = {
                (s_idx, p_idx, DIR.IN)
                for s_idx, p_idxs in node.edges_in.items()
                for p_idx in p_idxs
            }

            out_edges = {
                (o_idx, p_idx, DIR.OUT)
                for o_idx, p_idxs in node.edges_out.items()
                for p_idx in p_idxs
            }

            neighbors[node.idx] = list(in_edges | out_edges)

        return neighbors

    def training_batch(self,
                       device,
                       parent_start_idx,
                       node_weights,
                       max_edges=256,
                       max_nodes=32,
                       max_degree=16,
                       max_depth=5,
                       spawn_rate=0.2):
        """
        Create a LoRE training batch by sampling a subgraph and converting to LoreBatch.

        Returns
        -------
        LoreBatch
            node_idx_top, relation_idx_top enumerate the *subgraph-local* ids (0..|V|-1 / 0..|R|-1).
            neigh_* hold ragged neighbor rows, later padded by LoreBatch.to(device).
        """

        subgraph = self.get_random_subgraph(parent_start_idx=parent_start_idx,
                                            node_weights=node_weights,
                                            max_edges=max_edges,
                                            max_nodes=max_nodes,
                                            max_degree=max_degree,
                                            max_depth=max_depth,
                                            spawn_rate=spawn_rate)

        neighbors_batches = subgraph.get_neighbors()

        neigh_nodes, neigh_rels, inverse, masks = [], [], [], []

        # Iterate in subgraph-local index order (0..|V_sub|-1)
        for neighbors in [neighbors_batches[i] for i in range(len(subgraph.graph_nodes))]:
            # Split channels
            nodes_row = [n for (n, r, d) in neighbors]
            rels_row = [r for (n, r, d) in neighbors]
            inv = [int(d.value == -1) for (n, r, d) in neighbors]  # DIR.OUT -> 1, DIR.IN -> 0

            neigh_nodes.append(nodes_row)
            neigh_rels.append(rels_row)
            inverse.append(inv)

        batch = LoreBatch(node_idx_top=list(subgraph.node_mapping.items()),
                          relation_idx_top=list(subgraph.relation_mapping.items()),
                          neigh_nodes=neigh_nodes,
                          neigh_rels=neigh_rels,
                          inverse=inverse)
        batch.to(device)

        return batch

    def reconstruction_batch(self, node_idxs, device):
        """
        Build a LoreBatch for reconstruction (no top-level nodes supplied).
        Useful for full-graph passes where neighbor rows are indexed directly.
        """

        neighbors_batches = self.get_neighbors(node_idxs=node_idxs)

        neigh_nodes, neigh_rels, inverse, masks = [], [], [], []

        for neighbors in [neighbors_batches[i] for i in node_idxs]:
            # Split channels
            nodes_row = [n for (n, r, d) in neighbors]
            rels_row = [r for (n, r, d) in neighbors]
            inv = [int(d.value == -1) for (n, r, d) in neighbors]

            neigh_nodes.append(nodes_row)
            neigh_rels.append(rels_row)
            inverse.append(inv)

        batch = LoreBatch(node_idx_top=None,
                          relation_idx_top=None,
                          neigh_nodes=neigh_nodes,
                          neigh_rels=neigh_rels,
                          inverse=inverse)
        batch.to(device)

        return batch
