import networkx as nx
import glob
import os
import pickle
from torch_geometric.utils import from_networkx
import glob
import os
from multiprocessing import Pool

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import BaseHyperparams

def graph_labelling(nx_graph: nx.Graph):
    """
    Convert the graph attributes to labels for ML.
    """
    # add node labels
    nb_positive_labels = 0
    for node, data in nx_graph.nodes(data=True):
        # remove color attribute
        if 'color' in data:
            del data['color']
        if 'style' in data:
            del data['style']
        
        # add class label
        if 'label' in data:
            label = 0
            if "KEY" in data['label']:
                label = 1
                nb_positive_labels += 1
            nx.set_node_attributes(nx_graph, {node: label}, 'label')

    # assert that at least one node has a label 1
    assert nb_positive_labels > 0, "ERROR: No node has the label 1."
    
    return nx_graph

def convert_graph_to_ml_data(nx_graph: nx.Graph):
    """
    Convert the given NetworkX graph to a PyTorch Geometric data object.
    """
    return from_networkx(nx_graph)

def load_annotated_graph(
    params: ProgramParams ,
    annotated_graph_dot_gv_file_path: str
):
    """
    Load annotated graph from given path.
    Use pickle to save the graph to a file or load it from a file 
    if it already exists.
    Perform graph cleaning.
    Convert the graph to a PyTorch Geometric data object.
    Returns None if there was an error loading the graph.
    """    
    # load annotated graph
    file_name = os.path.basename(annotated_graph_dot_gv_file_path)
    nx_graph_pickle_path = params.PICKLE_DATASET_DIR_PATH + "/" + file_name + ".pickle"

    # Check if the save file exists
    if os.path.exists(nx_graph_pickle_path):
        # Load the NetworkX graph from the save file
        with open(nx_graph_pickle_path, 'rb') as file:
            nx_graph = pickle.load(file)
    else:
        # load the graph from the .gv file
        try:
            nx_graph = nx.Graph(nx.nx_pydot.read_dot(annotated_graph_dot_gv_file_path))
        except Exception as e:
            print(f"Error reading {annotated_graph_dot_gv_file_path}: {e}")
            os.remove(annotated_graph_dot_gv_file_path)
            print(f" 󰆴 -> Removed {annotated_graph_dot_gv_file_path}")
            return None
        
        # add node labels
        nx_graph = graph_labelling(nx_graph)

        # add edge weights
        for u, v, data in nx_graph.edges(data=True):
            weight = 1
            if 'weight' in data:
                weight = int(data['weight'])
            
            nx.set_edge_attributes(nx_graph, {(u, v): weight}, 'weight')
        
        # Save the NetworkX graph to a file using pickle
        with open(nx_graph_pickle_path, 'wb') as file:
            pickle.dump(nx_graph, file)

    return nx_graph

def find_gv_files(directory_path):
    # Create a pattern to match .gv files in all subdirectories
    pattern = os.path.join(directory_path, '**', '*.gv')
    
    # Use glob.glob with the recursive pattern
    gv_files = glob.glob(pattern, recursive=True)
    
    return gv_files

def dev_load_training_graphs(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    annotated_graph_dot_gv_dir_path: str
):
    """
    Load all annotated graphs from given path.
    """
    # get all files in the folder
    annotated_graph_dot_gv_file_paths = find_gv_files(
        annotated_graph_dot_gv_dir_path
    )
    print("Found " + str(len(annotated_graph_dot_gv_file_paths)) + " graphs inside " + annotated_graph_dot_gv_dir_path)
    for i in range(0, (len(annotated_graph_dot_gv_file_paths) // 100)):
        print("path:", annotated_graph_dot_gv_file_paths[i])

    # for now, as a test, filter only "Training" graphs
    annotated_graph_dot_gv_file_paths = [
        annotated_graph_dot_gv_file_path for annotated_graph_dot_gv_file_path 
        in annotated_graph_dot_gv_file_paths if "Training" in annotated_graph_dot_gv_file_path
    ]

    # for now, only load a certain number of graphs
    if hyperparams.nb_input_graphs is not None:
        annotated_graph_dot_gv_file_paths = annotated_graph_dot_gv_file_paths[:hyperparams.nb_input_graphs]
    print("Loading " + str(len(annotated_graph_dot_gv_file_paths)) + " graphs...")

    # Parallelize the loading of the graphs into data objects
    with Pool() as pool:
        datas = pool.starmap(load_annotated_graph, [
            (params, path) for path in annotated_graph_dot_gv_file_paths]
        )
    
    return datas