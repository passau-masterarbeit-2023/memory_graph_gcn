from dataclasses import dataclass
import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import Node2VecHyperparams, NodeEmbeddingType

def generate_node_embedding(
    params: ProgramParams,
    graph: nx.Graph,
    hyperparams: Node2VecHyperparams,
):
    USE_NODE2VEC_EMBEDDING = NodeEmbeddingType.Semantic.value in params.cli_args.args.node_embedding
    USE_SEMANTIC_EMBEDDING = NodeEmbeddingType.Node2Vec.value in params.cli_args.args.node_embedding
    
    model = None
    if USE_NODE2VEC_EMBEDDING: 
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
        node_node2vec_embedding = None
        node_semantic_embedding = None
        
        if USE_NODE2VEC_EMBEDDING:
            assert model is not None
            node_node2vec_embedding = model.wv[str(node)]

        if USE_SEMANTIC_EMBEDDING:
            assert data["comment"] is not None
            # additional semantic embedding
            additional_node_semantic_embedding: list[int | float] = json.loads(data["comment"].replace("\"", ""))
            node_semantic_embedding = np.array(
                additional_node_semantic_embedding, dtype=np.float32
            )

        # save node embedding
        if node_node2vec_embedding is not None and node_semantic_embedding is not None:
            node_embedding = np.concatenate(
                (node_node2vec_embedding, node_semantic_embedding),
                axis=None,
            )
            embeddings.append(node_embedding)
        elif node_node2vec_embedding is not None:
            embeddings.append(node_node2vec_embedding)
        elif node_semantic_embedding is not None:
            embeddings.append(node_semantic_embedding)
        else:
            raise Exception("No node embedding generated!")
    
    return embeddings