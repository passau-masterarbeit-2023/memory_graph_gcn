from dataclasses import dataclass
import json
import networkx as nx

from graph_conv_net.embedding.node_to_vec_enums import embedding_name_from_dir_path, get_graph_comment

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class MemGraph:
    graph: nx.Graph
    gv_file_path: str
    custom_node_embedding: str
    custom_embedding_fields: list[str]


def build_memgraph(
    nx_graph: nx.Graph,
    gv_file_path: str,
)  -> MemGraph:
    """
    Build a MemGraph object from the given NetworkX graph and the path to the .gv file.
    """
    # custom embedding name from gv file path
    embedding_name_from_path = embedding_name_from_dir_path(
        gv_file_path
    )

    # custom embedding name from graph
    graph_comment = get_graph_comment(
        gv_file_path
    )
    assert "embedding-fields" in graph_comment.keys(), (
        f"ERROR: No 'embedding-fields' found in graph comment of {gv_file_path}."
    )
    assert "embedding-type" in graph_comment.keys(), (
        f"ERROR: No 'embedding-type' found in graph comment of {gv_file_path}."
    )
    embedding_name_from_comment = graph_comment["embedding-type"]

    # check that the custom embedding name from the path and the comment are the same
    assert embedding_name_from_path == embedding_name_from_comment, (
        f"ERROR: The custom embedding name from the path and the comment are not the same. "
        f"From path: {embedding_name_from_path}, from comment: {embedding_name_from_comment}. "
        f"For GV file path: {gv_file_path}."
    )

    # get embedding fields
    embeddings_fields: list[str] = graph_comment["embedding-fields"]
    assert isinstance(embeddings_fields, list)
    assert isinstance(embeddings_fields[0], str) 
    assert len(embeddings_fields) > 0

    # check the embeddings in nodes
    nb_features = len(embeddings_fields)
    for node, data in nx_graph.nodes(data=True):
        assert "comment" in data.keys(), (
            f"ERROR: No 'comment' field found in node {node} of graph {gv_file_path}. "
            f"For GV file path: {gv_file_path}."
        )
        # check the number of features
        node_embedding: list[int | float] = json.loads(data["comment"].replace("\"", ""))
        assert(len(node_embedding) == nb_features), (
            f"ERROR: Expected comment embedding length to be {nb_features}, but got {len(node_embedding)}, "
            f"for node {node} with comment {data['comment']}. "
            f"For GV file path: {gv_file_path}."
        )

    memgraph = MemGraph(
        graph=nx_graph,
        gv_file_path=gv_file_path,
        custom_node_embedding=embedding_name_from_path,
        custom_embedding_fields=embeddings_fields,
    )
    return memgraph
