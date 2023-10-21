import argparse
from enum import Enum
import re
import networkx as nx
import json

class NodeEmbeddingType(Enum):
    Node2Vec = "node2vec"
    Comment = "comment" # custom node embedding from Mem2Graph stored in the graph's node comment field

def determine_embeddings_used(
    args: argparse.Namespace,
    input_mem2graph_dataset_dir_path: str,
):
    """
    Determine which embeddings are used in the graph, using the dir path, and CLI args.
    Returns a string with the embeddings used, that can be
    logged or stored.
    """
    list_node_embedding_types: list[str] = []
    if NodeEmbeddingType.Node2Vec.value in args.node_embedding:
        list_node_embedding_types.append(NodeEmbeddingType.Node2Vec.value)

    if NodeEmbeddingType.Comment.value in args.node_embedding:
        # the dir path must contain a '-c' flag followed by the embedding type
        # e.g. 'chunk-semantic-embedding', 'chunk-statistic-embedding', 'chunk-start-bytes-embedding'
        custom_comment_from_dir_path = "" 
        match = re.search(
            r'-c_([^_-]+)', input_mem2graph_dataset_dir_path
        )
        if match:
            custom_comment_from_dir_path = match.group(1)
        else:
            print(
                "🚩 PANIC: embedding type not found in dir path. "
                "Should be mentioned after '-c_' in mem2grapph dataset dir path. "
                f"Dir path: {input_mem2graph_dataset_dir_path}"
            )
            exit(1)

        list_node_embedding_types.append(custom_comment_from_dir_path)
    
    return list_node_embedding_types

def check_custom_comment_embeddings_used(
    declared_node_embedding_types: list[str],
    graph: nx.Graph,
) -> None:
    """
    Check that the embeddings used in the graph are the same as
    the embeddings specified the mem2graph dir path.
    The custom embedding stored in node comment fields is specified in the graph comment.
    """
    graph_comment = graph.graph['comment']
    print("graph_comment:", graph_comment)

    comment_object = json.loads(graph_comment)
    
    # check if the graph comment is a valid object with field "embedding-type"
    assert "embedding-type" in comment_object.keys()
    embedding_in_node_comments = comment_object["embedding-type"]
    
    assert embedding_in_node_comments in declared_node_embedding_types, (
        f"🚩 PANIC: Embedding mismatch. Embedding in node comments ({embedding_in_node_comments}) not in declared node embeddings ({declared_node_embedding_types})"
    )