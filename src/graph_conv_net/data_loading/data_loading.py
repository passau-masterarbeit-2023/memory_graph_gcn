import networkx as nx
from graph_conv_net.data_loading.file_loading import find_gv_files
from graph_conv_net.graph.memgraph import MemGraph, build_memgraph
import os
import pickle
from torch_geometric.utils import from_networkx
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.hyperparams import BaseHyperparams

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
    pickle_dataset_dir_path: str,
    annotated_graph_dot_gv_file_path: str,
    remove_file_if_error: bool,
):
    """
    Load annotated graph from given path.
    Use pickle to save the graph to a file or load it from a file 
    if it already exists.
    Perform graph cleaning.
    Convert the graph to a PyTorch Geometric data object.
    Returns None if there was an error loading the graph.
    """    

    # 0_graph_with_embedding_comments_-v_-a_chunk-header-node_-c_chunk-semantic-embedding_-e_none_-s_none__GraphWithEmbeddingComments_Training_Training_basic_V_6_8_P1_24_25634-1643890740-heap.raw_dot.gv.pickle
    # load annotated graph
    file_name = os.path.basename(annotated_graph_dot_gv_file_path)
    last_dir_folder_name = os.path.basename(os.path.dirname(annotated_graph_dot_gv_file_path))
    memgraph_pickle_path = pickle_dataset_dir_path + "/" + last_dir_folder_name[:2] + "__" + file_name + ".pickle"

    memgraph = None
    # Check if the save file exists
    if os.path.exists(memgraph_pickle_path):
        # Load the NetworkX graph from the save file
        with open(memgraph_pickle_path, 'rb') as file:
            memgraph = pickle.load(file)
    else:
        # load the graph from the .gv file
        try:
            # check that the file contains at least 2 lines
            with open(annotated_graph_dot_gv_file_path, "r") as f:
                lines = f.readlines()
                if not len(lines) >= 2:
                    raise ValueError(
                        f"🚩 ERROR: Expected at least 2 lines in file {annotated_graph_dot_gv_file_path}, "
                        f"but got {len(lines)} lines only."
                    )

            # NOTE: The DOT file actually has a problem with the 'comment' attribute.
            # A better solution would be to write a custom parser for the DOT file.
            # Because replacing by '-' actually does really solve the issue.
            # The comment is either considered a node, or removed...
            nx_graph = nx.Graph(nx.nx_pydot.read_dot(annotated_graph_dot_gv_file_path))
            # nx_graph = nx.Graph(nx.nx_agraph.read_dot(annotated_graph_dot_gv_file_path))

        except ModuleNotFoundError as module_not_found_error:
            print(f" 󰮘 'ModuleNotFoundError' reading {annotated_graph_dot_gv_file_path}: {module_not_found_error}")
            exit(1)
        except AssertionError as assertion_error:
            print(f" 󰮘 'AssertionError' reading {annotated_graph_dot_gv_file_path}: {assertion_error}")
            exit(1)
        except Exception as e:
            print(f" 󰮘 Error ('{type(e)}') reading {annotated_graph_dot_gv_file_path}: {e}")
            if remove_file_if_error:
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
        
        # build memgraph
        memgraph = build_memgraph(
            nx_graph,
            annotated_graph_dot_gv_file_path,
        )

        # Save the memgraph to a pickle file
        with open(memgraph_pickle_path, 'wb') as file:
            pickle.dump(memgraph, file)
    
    assert isinstance(memgraph, MemGraph), (
        f"ERROR: memgraph should be of type MemGraph, but got {type(memgraph)}."
        f"For GV file path: {annotated_graph_dot_gv_file_path}."
    )
    return memgraph

def dev_load_training_graphs(
    params: ProgramParams,
    annotated_graph_dot_gv_dir_path: str
):
    """
    Load all annotated graphs from given path.
    """
    # get all files in the folder
    annotated_graph_dot_gv_file_paths = find_gv_files(
        annotated_graph_dot_gv_dir_path
    )
    print("Found {} graphs inside {}".format(
        str(len(annotated_graph_dot_gv_file_paths)),
        annotated_graph_dot_gv_dir_path
    ))
    for i in range(0, (len(annotated_graph_dot_gv_file_paths) % 10)):
        print("path:", annotated_graph_dot_gv_file_paths[i])

    # for now, as a test, filter only "Training" graphs
    annotated_graph_dot_gv_file_paths = [
        annotated_graph_dot_gv_file_path for annotated_graph_dot_gv_file_path 
        in annotated_graph_dot_gv_file_paths if "Training" in annotated_graph_dot_gv_file_path
    ]

    # for now, only load a certain number of graphs
    nb_requested_input_graphs = params.cli.args.nb_input_graphs
    if nb_requested_input_graphs is not None:
        annotated_graph_dot_gv_file_paths = annotated_graph_dot_gv_file_paths[:nb_requested_input_graphs]
    print("Loading " + str(len(annotated_graph_dot_gv_file_paths)) + " graphs...")

    memgraphs: list[MemGraph] = []
    with ProcessPoolExecutor() as executor:
        # Submit all tasks and keep their futures
        futures = {
            executor.submit(
                load_annotated_graph, 
                params.PICKLE_DATASET_DIR_PATH, 
                path, 
                params.cli.args.remove_file_if_error
            ): path for path in annotated_graph_dot_gv_file_paths
        }
        
        # Create a TQDM progress bar
        progress_bar = tqdm(total=len(futures), desc="Loading graphs", dynamic_ncols=True)

        # Collect the results as they come in
        for future in as_completed(futures):
            path = futures[future]
            try:
                memgraph = future.result()
                if memgraph:
                    memgraphs.append(memgraph) # only append if not None
            except Exception as err:
                print(f'Generated an exception: {err} with graph {path}')

            # Update the progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()
    
    return memgraphs