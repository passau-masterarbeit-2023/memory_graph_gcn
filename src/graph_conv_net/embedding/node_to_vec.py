import json
import os
import pickle
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

    # node2vec embedding
    is_node2vec_embedding_cached = False
    model = None
    node_to_node2vec_embedding: dict[str, np.ndarray] | None = None
    node2vec_embedding_pickle_path: str | None = None

    if use_node2vec_embedding:
        assert isinstance(hyperparams, Node2VecHyperparams), (
            f"ERROR: Expected hyperparams to be of type Node2VecHyperparams, but got {type(hyperparams)}"
        )

        # caching: file name
        file_name = os.path.basename(memgraph.gv_file_path)
        last_dir_folder_name = os.path.basename(os.path.dirname(memgraph.gv_file_path))
        node2vec_hyperparams_str = hyperparams.node2vec_hyperparams_to_str()
        node2vec_embedding_pickle_path = params.PICKLE_CACHED_NODE2VEC_EMBEDDINGS + "/" + last_dir_folder_name[:2] + "__" + file_name + "__" + node2vec_hyperparams_str + ".pickle"
        is_node2vec_embedding_cached = os.path.exists(node2vec_embedding_pickle_path)

        # caching: skip node2vec embedding if cached
        if is_node2vec_embedding_cached:
            print(f"󰃨 CACHE: Node2Vec embedding already cached for graph {memgraph.gv_file_path}.")

            # load node2vec embedding for each node from pickle file
            with open(node2vec_embedding_pickle_path, 'rb') as file:
                node_to_node2vec_embedding = pickle.load(file)
                assert isinstance(node_to_node2vec_embedding, dict), (
                    f"ERROR: Expected node_to_node2vec_embedding to be of type dict, but got {type(node_to_node2vec_embedding)}"
                )
                assert len(node_to_node2vec_embedding) == len(memgraph.graph.nodes), (
                    f"ERROR: Expected node_to_node2vec_embedding to have {len(memgraph.graph.nodes)} nodes, but got {len(node_to_node2vec_embedding)} nodes."
                )
        else:
            # No cached node2vec embedding, generate its model from scratch
            node2vec = Node2Vec(
                memgraph.graph, 
                dimensions=hyperparams.node2vec_dimensions,
                walk_length=hyperparams.node2vec_walk_length,
                num_walks=hyperparams.node2vec_num_walks,
                workers=hyperparams.node2vec_workers,
                seed=params.RANDOM_SEED,
                p=hyperparams.node2vec_p,
                q=hyperparams.node2vec_q,
                quiet = params.cli.args.quiet,
            )
            model = node2vec.fit(
                window=hyperparams.node2vec_window,
                min_count=1, 
                batch_words=hyperparams.node2vec_batch_words,
            )
    
    embeddings = []
    node_to_node2vec_embedding_for_cache: dict[str, np.ndarray] = {}
    for node, data in memgraph.graph.nodes(data=True):
        node_node2vec_embedding = None
        node_comment_embedding = None
        
        if use_node2vec_embedding:
            if is_node2vec_embedding_cached:
                assert node_to_node2vec_embedding is not None
                node_node2vec_embedding = node_to_node2vec_embedding[str(node)]
            else:
                assert model is not None
                node_node2vec_embedding = model.wv[str(node)]
                node_to_node2vec_embedding_for_cache[str(node)] = node_node2vec_embedding

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
    
    # cache: save node2vec embedding for each node to pickle file
    if not is_node2vec_embedding_cached and use_node2vec_embedding:
        assert node2vec_embedding_pickle_path is not None
        with open(node2vec_embedding_pickle_path, 'wb') as file:
            pickle.dump(node_to_node2vec_embedding_for_cache, file)
    
    return embeddings