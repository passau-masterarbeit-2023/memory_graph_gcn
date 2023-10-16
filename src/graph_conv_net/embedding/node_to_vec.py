from dataclasses import dataclass
import networkx as nx
from node2vec import Node2Vec

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.results.base_result_writer import BaseResultWriter

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class FirstGCNPipelineHyperparams(object):
    """
    This class contains the hyperparameters for the first GCN pipeline.
    """
    node2vec_dimensions: int
    node2vec_walk_length: int
    node2vec_num_walks: int
    node2vec_p: float
    node2vec_q: float
    node2vec_window: int
    node2vec_batch_words: int

def add_hyperparams_to_result_writer(
    result_writer: BaseResultWriter,
    hyperparams: FirstGCNPipelineHyperparams,
):
    """
    Add the hyperparams to the result writer.
    """
    for field in hyperparams.__dataclass_fields__.keys():
        result_writer.set_result(
            field, 
            getattr(hyperparams, field)
        )

def generate_node2vec_graph_embedding(
    params: ProgramParams,
    graph: nx.Graph,
    hyperparams: FirstGCNPipelineHyperparams,
):
    # Generate Node2Vec embeddings
    node2vec = Node2Vec(
        graph, 
        dimensions=hyperparams.node2vec_dimensions,
        walk_length=hyperparams.node2vec_walk_length,
        num_walks=hyperparams.node2vec_num_walks,
        workers=1, #workers=params.MAX_ML_WORKERS,
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