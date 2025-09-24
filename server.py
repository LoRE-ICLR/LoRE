# Minimal Flask server that wraps LoreManager for training, reconstruction,
# "what-if" queries, and live graph updates. The comments explain route behavior,
# streaming, and the N3 change-log for applied updates.

import zipfile
from flask import Flask, request, jsonify, stream_with_context, Response
import torch
import json
from rdflib import Graph, URIRef
from manager import LoreManager, TrainingInfo
import argparse
from typing import Set, Tuple
import sys


class LoreApp(Flask):
    """
    Flask application exposing LoRE functionality.

    Endpoints
    ---------
    GET  /             -> health/config probe
    GET  /base         -> export current node/relation URIs + embeddings
    GET  /log?type=... -> retrieve N3-encoded change-log (add/remove)
    POST /train        -> start streaming training logs over a long-lived response
    POST /rec          -> reconstruct embeddings for a set of URIs
    POST /what-if      -> hypothetical neighborhood embedding
    POST /update       -> apply add/remove edge updates (and log them)

    Notes
    -----
    - Training streams plain text lines; the client should keep the connection open.
    - Updates maintain a simple in-memory RDFLib Graph for 'add' and 'remove' logs.
    """

    def __init__(self, import_name, graph_data, lore_config, **kwargs):
        super().__init__(import_name, **kwargs)

        # Pick GPU if available; the rest of the stack is device-aware.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model manager with the provided graph and config.
        self.manager = LoreManager(device=self.device,
                                   graph_data=graph_data,
                                   lore_config=lore_config)

        # Make sure optimizer state is clean before first use.
        self.manager.optimizer.zero_grad()

        # Bookkeeping for historical training runs (if needed for UI/monitoring).
        self.training_infos = []

        # RDFLib graphs that track deltas since last training call / since startup.
        self.log = {"add": Graph(), "remove": Graph()}

        # Route registration
        self.add_url_rule("/", view_func=self.home)
        self.add_url_rule("/train", view_func=self.train, methods=["POST"])
        self.add_url_rule("/base", view_func=self.get_base, methods=["GET"])
        self.add_url_rule("/rec", view_func=self.reconstruct, methods=["POST"])
        self.add_url_rule("/what-if", view_func=self.what_if, methods=["POST"])
        self.add_url_rule("/update", view_func=self.update, methods=["POST"])
        self.add_url_rule("/log", view_func=self.get_log, methods=["GET"])

    def home(self):
        """
        Simple health endpoint. Also returns current LoRE configuration.
        """
        response = {"message": "Hi, I am LoRE!"}
        if self.manager is not None:
            response['config'] = self.manager.get_config()
        return jsonify(response), 200

    def get_log(self):
        """
        Return the current change-log as N3 (either 'add' or 'remove').

        Example
        -------
        GET /log?type=add
        GET /log?type=remove
        """
        log_type = request.args.get("type")

        if log_type == "add":
            graph = self.log['add']
        elif log_type == "remove":
            graph = self.log['remove']
        else:
            return Response("Invalid log type. Use ?type=add or ?type=remove.", status=400)

        n3_data = graph.serialize(format="n3")
        return Response(n3_data, mimetype='text/plain')

    def get_base(self):
        """
        Export current URIs and embeddings for nodes and relations.
        """
        try:
            return jsonify(self.manager.get_base()), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def train(self):
        """
        Kick off streaming training. The response body is a text stream of progress lines.

        Body (JSON)
        -----------
        {
          "batch_specs": {...},  # optional
          "base_nodes": [...],   # optional list of URIs; default: all nodes
          "epochs": 10,
          "exponent": 2,
          "num_workers": 0
        }

        Notes
        -----
        - Uses Flask's stream_with_context to push lines as they become available.
        - Handles client disconnects gracefully (no stack traces on broken pipe).
        """

        def generator(gen):
            # Wrap the manager's generator to handle common network interruptions.
            try:
                for chunk in gen:
                    yield chunk
            except (ConnectionResetError, BrokenPipeError):
                print("Client disconnected", file=sys.stderr)
            except GeneratorExit:
                print("Generator closed (possibly due to disconnect)", file=sys.stderr)
            finally:
                print("Generator finished or interrupted", file=sys.stderr)

        config = request.get_json(silent=True) or {}

        # Reset change-log on new training session (optional choice).
        self.log = {"add": Graph(), "remove": Graph()}

        training_info = TrainingInfo(
            batch_specs=config.get("batch_specs", None),
            base_nodes=config.get("base_nodes", None),
            epochs=config.get("epochs", 10),
            exponent=config.get("exponent", 2)
        )
        self.training_infos.append(training_info)

        return Response(
            stream_with_context(generator(self.manager.train_lore(
                training_info=training_info,
                num_workers=config.get("num_workers", 0)
            ))),
            mimetype='text/plain'
        )

    def reconstruct(self):
        """
        Reconstruct embeddings for provided URIs.

        Body (form-data)
        ----------------
        uris: JSON-encoded list of string URIs

        Returns
        -------
        { "result": [[...], ...] }  # list of vectors
        """
        uris = json.loads(request.form.get("uris"))
        try:
            result = self.manager.reconstruct(reconstruct_items=uris).detach().cpu().numpy().tolist()
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def what_if(self):
        """
        Hypothetical neighborhood embedding:
        Given neighbors/relations/inverse flags, compute an attended embedding.

        Body (JSON)
        -----------
        {
          "neighbors": [ "node_uri", ... ],
          "relations": [ "rel_uri", ... ],
          "inverse":   [ 0|1, ... ]  # aligns with neighbors/relations
        }
        """
        config = request.get_json()
        if not config:
            return jsonify({"error": "Missing JSON body"}), 400

        neighbors = config.get("neighbors")
        relations = config.get("relations")
        inverse = config.get("inverse")

        try:
            result = self.manager.what_if(neighbors=neighbors,
                                          relations=relations,
                                          inverse=inverse)
            # NOTE: If jsonify fails (Tensor not serializable), convert to list:
            # result = result.detach().cpu().numpy().tolist()
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def update_log(self, triples_add: Set[Tuple[str, str, str]], triples_del: Set[Tuple[str, str, str]]):
        """
        Internal helper to update the in-memory change-log graphs.

        If a triple is added and was previously marked removed, we cancel the removal (and vice versa).
        """
        if triples_add is not None:
            for s, p, o in triples_add:
                edge = (URIRef(s), URIRef(p), URIRef(o))
                if edge in self.log["remove"]:
                    self.log["remove"].remove(edge)
                else:
                    self.log["add"].add(edge)

        if triples_del is not None:
            for s, p, o in triples_del:
                edge = (URIRef(s), URIRef(p), URIRef(o))
                if edge in self.log["add"]:
                    self.log["add"].remove(edge)
                else:
                    self.log["remove"].add(edge)

    def update(self):
        """
        Apply add/remove updates to the KG and mirror them in the change-log.

        Body (JSON)
        -----------
        {
          "nodes": ["uri_s0", "uri_s1", ...],
          "relations": ["uri_p0", ...],
          "updates": {
            "add":    [[s_i, o_i, p_i], ...],
            "remove": [[s_i, o_i, p_i], ...]
          }
        }

        Returns
        -------
        {
          "added":   [0/1, ...],  # per-edge add success flags
          "deleted": [0/1, ...],  # per-edge remove success flags
          "message": "Update applied"
        }
        """

        config = request.get_json()

        nodes = config.get("nodes")
        relations = config.get("relations")
        updates = config.get("updates")

        # Update local change-log first (for immediate visibility via /log)
        if 'add' in updates:
            for s_idx, o_idx, p_idx in updates['add']:
                edge = (
                    URIRef(nodes[s_idx]),
                    URIRef(relations[p_idx]),
                    URIRef(nodes[o_idx])
                )
                if edge in self.log["remove"]:
                    self.log["remove"].remove(edge)
                else:
                    self.log["add"].add(edge)

        if 'remove' in updates:
            for s_idx, o_idx, p_idx in updates['remove']:
                edge = (
                    URIRef(nodes[s_idx]),
                    URIRef(relations[p_idx]),
                    URIRef(nodes[o_idx])
                )
                if edge in self.log["add"]:
                    self.log["add"].remove(edge)
                else:
                    self.log["remove"].add(edge)

        # Apply to the actual graph + embeddings
        additions, removals = self.manager.update_kg(nodes=nodes, relations=relations, updates=updates)

        return jsonify({
            "added": list(additions),
            "deleted": list(removals),
            "message": "Update applied"
        }), 200


if __name__ == "__main__":
    # CLI interface:
    #   --graph  path/to/graph.json.zip  (required; expects an inner file named 'graph.json')
    #   --config path/to/lore_config.json (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to the graph file (.json.zip)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to Lore config file"
    )
    args = parser.parse_args()

    graph_file = args.graph
    lore_config_file = args.config

    # Load serialized graph (zip containing a single 'graph.json')
    with zipfile.ZipFile(graph_file, "r") as zf:
        with zf.open("graph.json") as f:
            graph_data = json.load(f)

    # Load optional LoRE configuration
    if lore_config_file is not None:
        with open(lore_config_file) as f:
            lore_config = json.load(f)
    else:
        lore_config = {}

    # Start Flask app (no reloader/tracebacks in production mode).
    app = LoreApp(__name__, graph_data, lore_config)
    app.run(host="0.0.0.0", port=5000, debug=False)
