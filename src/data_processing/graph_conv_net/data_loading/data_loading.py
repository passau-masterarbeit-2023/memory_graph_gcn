import graphviz

def load_annotated_graph(annotated_graph_dot_gv_file_path: str):
    """
    Load annotated graph from given path.
    Return a DOT graph using graphviz library.
    """
    # Read the file
    with open(annotated_graph_dot_gv_file_path, 'r') as f:
        dot_graph = f.read()
    
    # Create a graph from the dot data
    graph = graphviz.Source(dot_graph)

    return graph

