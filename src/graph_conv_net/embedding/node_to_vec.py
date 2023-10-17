from dataclasses import dataclass
import networkx as nx
from node2vec import Node2Vec

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import Node2VecHyperparams

def generate_node2vec_graph_embedding(
    params: ProgramParams,
    graph: nx.Graph,
    hyperparams: Node2VecHyperparams,
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
    for node in graph.nodes():
        embeddings.append(model.wv[str(node)])
    
    return embeddings