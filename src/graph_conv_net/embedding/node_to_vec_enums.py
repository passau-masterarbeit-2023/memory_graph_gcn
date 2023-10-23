import argparse
from enum import Enum
import re
import networkx as nx
import json

class NodeEmbeddingType(Enum):
    Node2Vec = "node2vec"
    CustomCommentEmbedding = "comment" # custom node embedding from Mem2Graph stored in the graph's node comment field
    Node2VecAndComment = "node2vec-comment" # both node2vec and custom node embedding from Mem2Graph stored in the graph's node comment field

    def is_using_node2vec(self):
        return self.value == NodeEmbeddingType.Node2Vec.value or self.value == NodeEmbeddingType.Node2VecAndComment.value

    def is_using_custom_comment_embedding(self):
        return self.value == NodeEmbeddingType.CustomCommentEmbedding.value or self.value == NodeEmbeddingType.Node2VecAndComment.value

    def to_string(
        self, input_mem2graph_dataset_dir_path
    ):
        """
        Convert the enum to a string that can be used in the result writer.
        Replace the "comment" embedding by the real embedding name.
        NOTE: This name should be provided on the :input_mem2graph_dataset_dir_path:
        """
        if self.is_using_custom_comment_embedding(): 
            comment_embedding_name = embedding_name_from_dir_path(
                input_mem2graph_dataset_dir_path
            )
            if self == NodeEmbeddingType.Node2VecAndComment:
                return f"node2vec-{comment_embedding_name}"
            else:
                return comment_embedding_name
        else:
            return self.value
    
    @staticmethod
    def get_list_of_embeddings():
        return [e for e in NodeEmbeddingType]

def embedding_name_from_dir_path(
    input_mem2graph_dataset_dir_path: str
):
    """
    Return the name of the embedding used in the given mem2graph dataset dir path.
    """
    match = re.search(
        r'(?<=_-c_)\w+[-a-zA-Z+]*', input_mem2graph_dataset_dir_path
    )
    if match:
        return match.group(0)
    else:
        print(
            "🚩 PANIC: embedding type not found in dir path. "
            "Should be mentioned after '-c_' in mem2grapph dataset dir path. "
            f"Dir path: {input_mem2graph_dataset_dir_path}"
        )
        exit(1)

def check_comment_embedding_coherence(
    input_mem2graph_dataset_dir_path: str,
    graph: nx.Graph,
) -> None:
    """
    Check that the embeddings used in the graph are the same as
    the embeddings specified the mem2graph dir path.
    The custom embedding stored in node comment fields is specified in the graph comment.
    """
    # get name from dir path
    embedding_name_from_path = embedding_name_from_dir_path(
        input_mem2graph_dataset_dir_path
    )

    # get name from graph comment
    graph_comment = graph.graph['comment']
    comment_object = json.loads(graph_comment)
    
    # check if the graph comment is a valid object with field "embedding-type"
    assert "embedding-type" in comment_object.keys()
    embedding_in_node_comments = comment_object["embedding-type"]
    
    assert embedding_name_from_path == embedding_in_node_comments, (
        f"🚩 PANIC: Embedding mismatch. "
        f"Embedding in node comments ({embedding_in_node_comments}) "
        f"not equal to the one from dir path ({embedding_name_from_path})"
    )
