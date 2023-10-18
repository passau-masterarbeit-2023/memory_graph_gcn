from dataclasses import dataclass
import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import Node2VecHyperparams

def generate_node2vec_graph_embedding(
    params: ProgramParams,
    graph: nx.Graph,
    hyperparams: Node2VecHyperparams,
    add_node_semantic_embedding: bool = False,
):
    # Generate Node2Vec embeddings
    node2vec = Node2Vec(
        graph, 
        dimensions=hyperparams.node2vec_dimensions,
        walk_length=hyperparams.node2vec_walk_length,
        num_walks=hyperparams.node2vec_num_walks,
        workers=hyperparams.node2vec_workers,
        seed=params.RANDOM_SEED,
        p=hyperparams.node2vec_p,
        q=hyperparams.node2vec_q,
    )
    model = node2vec.fit(
        window=hyperparams.node2vec_window,
        min_count=1, 
        batch_words=hyperparams.node2vec_batch_words,
    )
    
    embeddings = []
    for node, data in graph.nodes(data=True):
        node_node2vec_embedding = model.wv[str(node)]

        if add_node_semantic_embedding and "comment" in data:
            # additional semantic embedding
            additional_node_semantic_embedding: list[int | float] = json.loads(data["comment"].replace("\"", ""))
            node_semantic_embedding = np.array(
                additional_node_semantic_embedding, dtype=np.float32
            )

            node_embedding = np.concatenate(
                (node_node2vec_embedding, node_semantic_embedding),
                axis=None,
            )
            embeddings.append(node_embedding)
        else:
            embeddings.append(node_node2vec_embedding)
    
    return embeddings