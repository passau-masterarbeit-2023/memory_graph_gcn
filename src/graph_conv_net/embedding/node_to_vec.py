import networkx as nx
from node2vec import Node2Vec

from graph_conv_net.params.params import ProgramParams

def generate_node2vec_graph_embedding(
    params: ProgramParams,
    graph: nx.Graph,
):
    # Generate Node2Vec embeddings
    node2vec = Node2Vec(
        graph, 
        dimensions=16, 
        walk_length=16, 
        num_walks=20, 
        workers=params.MAX_ML_WORKERS,
        seed=params.RANDOM_SEED,
    )
    model = node2vec.fit(
        window=10, 
        min_count=1, 
        batch_words=4
    )
    
    embeddings = []
    for node in graph.nodes():
        embeddings.append(model.wv[str(node)])
    
    return embeddings