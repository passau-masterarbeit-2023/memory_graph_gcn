import json
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from graph_conv_net.pipelines.hyperparams import Node2VecHyperparams
from graph_conv_net.params.params import ProgramParams

def generate_node_embedding(
    params: ProgramParams,
    graph: nx.Graph,
    hyperparams: Node2VecHyperparams,
):
    """
    Generate node embedding for the graph.
    Parse every node in the graph, and generate a node embedding.
    """
    use_node2vec_embedding = hyperparams.node_embedding.is_using_node2vec()
    use_comment_embedding = hyperparams.node_embedding.is_using_custom_comment_embedding()

    model = None
    if use_node2vec_embedding: 
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
        node_comment_embedding = None
        
        if use_node2vec_embedding:
            assert model is not None
            node_node2vec_embedding = model.wv[str(node)]

        if use_comment_embedding:
            assert data["comment"] is not None
            # additional semantic embedding
            additional_node_comment_embedding: list[int | float] = json.loads(data["comment"].replace("\"", ""))
            node_comment_embedding = np.array(
                additional_node_comment_embedding, dtype=np.float32
            )

        # save node embedding
        if node_node2vec_embedding is not None and node_comment_embedding is not None:
            node_embedding = np.concatenate(
                (node_node2vec_embedding, node_comment_embedding),
                axis=None,
            )
            embeddings.append(node_embedding)
        elif node_node2vec_embedding is not None:
            embeddings.append(node_node2vec_embedding)
        elif node_comment_embedding is not None:
            embeddings.append(node_comment_embedding)
        else:
            raise Exception("No node embedding generated!")
    
    return embeddings