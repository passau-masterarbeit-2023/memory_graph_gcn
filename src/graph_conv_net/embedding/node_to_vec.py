import json
import networkx as nx
from node2vec import Node2Vec
from graph_conv_net.graph.memgraph import MemGraph
from graph_conv_net.utils.debugging import dp
import numpy as np

from graph_conv_net.pipelines.hyperparams import BaseHyperparams, Node2VecHyperparams
from graph_conv_net.params.params import ProgramParams

def generate_node_embedding(
    params: ProgramParams,
    memgraph: MemGraph,
    hyperparams: BaseHyperparams,
    custom_comment_embedding_length: int,
):
    """
    Generate node embedding for the graph.
    Parse every node in the graph, and generate a node embedding.
    """
    use_node2vec_embedding = hyperparams.node_embedding.is_using_node2vec()
    use_comment_embedding = hyperparams.node_embedding.is_using_custom_comment_embedding()

    model = None
    if use_node2vec_embedding: 
        assert isinstance(hyperparams, Node2VecHyperparams), (
            f"ERROR: Expected hyperparams to be of type Node2VecHyperparams, but got {type(hyperparams)}"
        )
        # Generate Node2Vec embeddings
        node2vec = Node2Vec(
            memgraph.graph, 
            dimensions=hyperparams.node2vec_dimensions,
            walk_length=hyperparams.node2vec_walk_length,
            num_walks=hyperparams.node2vec_num_walks,
            workers=hyperparams.node2vec_workers,
            seed=params.RANDOM_SEED,
            p=hyperparams.node2vec_p,
            q=hyperparams.node2vec_q,
            quiet = not params.cli.args.debug,
        )
        model = node2vec.fit(
            window=hyperparams.node2vec_window,
            min_count=1, 
            batch_words=hyperparams.node2vec_batch_words,
        )
    
    embeddings = []
    for node, data in memgraph.graph.nodes(data=True):
        node_node2vec_embedding = None
        node_comment_embedding = None
        
        if use_node2vec_embedding:
            assert model is not None
            node_node2vec_embedding = model.wv[str(node)]

        if use_comment_embedding:
            node_comment: str | None = data["comment"]

            assert node_comment is not None
            # WARN: The statistical embedding can generate NaN values
            # In that case, the embedding is not generated for this node (skipped node)
            if "NaN" in data["comment"]:
                dp(f"WARNING: NaN value in comment embedding for node {node}. Replacing by 0.")
                node_comment = node_comment.replace("NaN", "0")
                

            # additional semantic embedding
            additional_node_comment_embedding: list[int | float] = json.loads(node_comment.replace("\"", ""))
            assert(len(additional_node_comment_embedding) == custom_comment_embedding_length), (
                f"ERROR: Expected comment embedding length to be {custom_comment_embedding_length}, but got {len(additional_node_comment_embedding)}, "
                f"for node {node} with comment {data['comment']} "
                f"for GV file path: {memgraph.gv_file_path}."
            )
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