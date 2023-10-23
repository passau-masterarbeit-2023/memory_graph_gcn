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

def get_graph_comment(
    input_mem2graph_dataset_dir_path: str
):
    """
    The second line of any graph generated by Mem2Graph contains a 'comment' field,
    that is a JSON object.
    This function returns the comment object.
    """
    with open(input_mem2graph_dataset_dir_path, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 2, (
            f"🚩 PANIC: Expected at least 2 lines in file {input_mem2graph_dataset_dir_path}, but got {len(lines)}"
        )
        full_comment_line = lines[1].strip()
        assert full_comment_line.startswith("comment"), (
            f"🚩 PANIC: Expected the second line of file {input_mem2graph_dataset_dir_path} to start with 'comment', but got {full_comment_line}"
        )
        comment_content = full_comment_line.replace("comment=\"", "")
        comment_content = comment_content[:-1] # remove the last character, which is a double quote
        comment_content = comment_content.replace("'", "\"") # WARN: a real JSON uses " characters, not '
        comment_object = json.loads(comment_content)
        return comment_object

