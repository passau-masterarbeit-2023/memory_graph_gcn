"""
This file checks the validity of the gv files in the given directory.
This includes that the files can be loaded into a graph,
and that each node has a comment attribute, whose length is equal to the number of features
provided in the graph top level comment.
"""

from datetime import datetime
import os
import resource
import traceback
from dotenv import load_dotenv

from graph_conv_net.data_loading.data_loading import load_annotated_graph
from graph_conv_net.data_loading.file_loading import find_gv_files, find_pickle_files
from graph_conv_net.graph.memgraph import MemGraph
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from graph_conv_net.utils.utils import datetime_to_human_readable_str

# -------------------- Memory limit -------------------- #
MAX_MEMORY_GB = 250  # 250 GB
MAX_MEMORY_IN_BYTES = MAX_MEMORY_GB * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, 
    (MAX_MEMORY_IN_BYTES, MAX_MEMORY_IN_BYTES)
)

# -------------------- CLI arguments -------------------- #
import sys
import argparse
import psutil

# wrapped program flags
class CLIArguments:
    args: argparse.Namespace

    def __init__(self) -> None:
        self.__log_raw_argv()
        self.__parse_argv()
    
    def __log_raw_argv(self) -> None:
        print("Passed program params:")
        for i in range(len(sys.argv)):
            print("param[{0}]: {1}".format(
                i, sys.argv[i]
            ))
    
    def __parse_argv(self) -> None:
        """
        python main [ARGUMENTS ...]
        """
        parser = argparse.ArgumentParser(description='Program [ARGUMENTS]')
        # no delete old output files
        parser.add_argument(
            '-k',
            '--keep-old-output',
            action='store_true',
            help="Keep old output files."
        )
        # skip directory starting with number
        parser.add_argument(
            '-s',
            '--skip-dir-starting-with-number',
            type=int,
            default=None,
            help="Skip directory starting with provided number."
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help="Dry run. Don't load data and perform tests."
        )

        # save parsed arguments
        self.args = parser.parse_args()

        # log parsed arguments
        print("Parsed program params:")
        for arg in vars(self.args):
            print("{0}: {1}".format(
                arg, getattr(self.args, arg)
            ))


def load_env_file():
    """
    Load the environment variables from the .env file.
    """
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

def check_memory():
    # Get the memory usage in GB
    memory_info = psutil.virtual_memory()
    used_memory_gb = (memory_info.total - memory_info.available) / (1024 ** 3)
    return used_memory_gb

def simple_get_mem2graph_dataset_dir_paths(cli: CLIArguments):
    """
    Determine the Mem2Graph dataset directory paths.
    """
    mem2graph_dataset_dir_paths = []
    ALL_MEM2GRAPH_DATASET_DIR_PATH = os.environ.get("ALL_MEM2GRAPH_DATASET_DIR_PATH")
    assert ALL_MEM2GRAPH_DATASET_DIR_PATH is not None, (
        "🚩 PANIC: ALL_MEM2GRAPH_DATASET_DIR_PATH is None. "
        "Please set it in the .env file."
    )

    # Using all Mem2Graph datasets
    print("🔷 Looking for Mem2Graph dataset directories in {0}...".format(
        ALL_MEM2GRAPH_DATASET_DIR_PATH
    ))

    for dir_name in os.listdir(ALL_MEM2GRAPH_DATASET_DIR_PATH):
        to_be_added = False
        if ".gitignore" not in dir_name:
            if cli.args.skip_dir_starting_with_number is not None:
                str_number_to_skip = str(cli.args.skip_dir_starting_with_number)
                last_dir_component = os.path.basename(dir_name)
                print("last_dir_component:", last_dir_component)

                if not last_dir_component.startswith(str_number_to_skip):
                    to_be_added = True
            else:
                to_be_added = True
        
        if to_be_added:
            mem2graph_dataset_dir_paths.append(
                os.path.join(ALL_MEM2GRAPH_DATASET_DIR_PATH, dir_name)
            )
        else:
            print("󱧴 Skipping {0}...".format(dir_name))

    print("📁 Found {0} Mem2Graph dataset directories.".format(
        str(len(mem2graph_dataset_dir_paths))
    ))
    mem2graph_dataset_dir_paths.sort()
    
    return mem2graph_dataset_dir_paths

def remove_pickled_cached_graphs():
    """
    Remove all cached graphs in the given directory.
    """
    PICKLE_DATASET_DIR_PATH = os.environ.get("PICKLE_DATASET_DIR_PATH")
    nb_removed_cached_graphs = 0
    
    print(f"Removing all cached graphs in {PICKLE_DATASET_DIR_PATH}...")
    # get file path of all .pickle files in the dir
    cached_graphs_file_paths = find_pickle_files(
        PICKLE_DATASET_DIR_PATH
    )

    for cached_graphs_file_path in cached_graphs_file_paths:
        os.remove(cached_graphs_file_path)
        print(f" 󰆴 -> Removed pickeled cache {cached_graphs_file_path}")
        nb_removed_cached_graphs += 1
    
    return nb_removed_cached_graphs

def load_graph(gv_file_path: str):
    PICKLE_DATASET_DIR_PATH = os.environ.get("PICKLE_DATASET_DIR_PATH")
    assert PICKLE_DATASET_DIR_PATH is not None, (
        "🚩 PANIC: PICKLE_DATASET_DIR_PATH is None. "
        "Please set it in the .env file."
    )
    memgraph = load_annotated_graph(
        PICKLE_DATASET_DIR_PATH,
        gv_file_path,
    )
    if memgraph is None:
        print("🔄 MemGraph from {0} is None, skipping...".format(gv_file_path))
        return None
    
    # don't return the full graph, for memory optimization
    nb_features = len(memgraph.custom_embedding_fields)
    del memgraph

    return nb_features

def load_and_check_graph_in_dir(dir_path: str): 
    gv_file_paths = find_gv_files(dir_path)
    if len(gv_file_paths) == 0:
        print("󰡯 No .gv files found in {0}, skipping...".format(dir_path))
        return 0, 0

    list_of_nb_of_features_per_graph: list[int | None] = []
    nb_skipped = 0

    print(" 🔘 Loading graphs in {0}...".format(dir_path))

    # load memgraphs from gv files in dir in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all tasks and keep their futures
        futures = {
            executor.submit(
                load_graph, path
            ): path for path in gv_file_paths
        }

        # Create a TQDM progress bar
        progress_bar = tqdm(total=len(futures), desc="Loading graphs", dynamic_ncols=True)

        # Collect the results as they come in
        for future in as_completed(futures):
            path = futures[future]
            if check_memory() > MAX_MEMORY_GB:
                progress_bar.close()
                raise MemoryError("🚩 Memory limit reached, exiting...")
            
            try:
                assert path is not None, (
                    f"ERROR: path should not be None."
                )
                assert path.endswith(".gv"), (
                    f"ERROR: path should end with '.gv', but got {path}."
                )

                res = future.result()
                list_of_nb_of_features_per_graph.append(res)
            except Exception as err:
                print(f'Generated an exception: {err}\n with graph at path: {path}')    
                #traceback.print_exc() # Print the traceback
                nb_skipped += 1

            # Update the progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()

    # check that the list of results has the same length as the list of dir paths
    assert len(list_of_nb_of_features_per_graph) == len(gv_file_paths), (
        f"ERROR: Expected the list of results to have the same length as the list of dir paths, "
        f"but got {len(list_of_nb_of_features_per_graph)} results and {len(gv_file_paths)} dir paths."
        f"for Mem2Graph dataset dir: {dir_path}"
    )

    # Check that the embedding length is consistent across all graphs in the dir
    print(" 🔘 Checking embedding length of graphs in {0}...".format(dir_path))
    max_feature_length = max([nb_features for nb_features in list_of_nb_of_features_per_graph if nb_features is not None])

    for i in range(1, len(list_of_nb_of_features_per_graph)):
        nb_features = list_of_nb_of_features_per_graph[i]
        if nb_features is None:
            continue
        elif nb_features != max_feature_length:
            gv_current_file_path = gv_file_paths[i]
            raise ValueError(
                "🚩 PANIC: embedding length is not consistent across all graphs in the dir. "
                f"Expected (max nb of features): {max_feature_length}, but got: {nb_features}"
                f", for graph at path: {gv_current_file_path}"
            )

    return len(list_of_nb_of_features_per_graph), nb_skipped
        
def main(cli: CLIArguments):
    # remove all cached graphs
    if not cli.args.keep_old_output:
        nb_removed_cached_graphs = remove_pickled_cached_graphs()
        print(f"💥 Removed {nb_removed_cached_graphs} cached graphs from {os.environ.get('PICKLE_DATASET_DIR_PATH')}.")
    print("🔷 Now, performing data loading and sanity checks...")

    # get Mem2Graph dataset path list
    mem2graph_dataset_dir_paths = simple_get_mem2graph_dataset_dir_paths(cli)
    print("📁 Mem2Graph dataset dir paths:")
    for mem2graph_dataset_dir_path in mem2graph_dataset_dir_paths:
        nb_gv_files_in_dir = len(find_gv_files(mem2graph_dataset_dir_path))
        print(
            f" -> 📁 {mem2graph_dataset_dir_path}, "
            f"which contains {nb_gv_files_in_dir} files."
        )
    
    if cli.args.dry_run:
        print("🔶 Dry run, exiting...")
        return


    # iterate over all Mem2Graph dataset dir paths
    nb_memgraphs = 0
    nb_skipped = 0
    for mem2graph_dir_path in mem2graph_dataset_dir_paths:
        try:
            nb_memgraphs_, nb_skipped_ = load_and_check_graph_in_dir(
                mem2graph_dir_path,
            )
            print(f" -> ✅ {nb_memgraphs_} graphs in {mem2graph_dir_path} have been loaded and checked.")
            print(f" -> 💥 {nb_skipped_} graphs in {mem2graph_dir_path} have been skipped (deleted).")

            nb_memgraphs += nb_memgraphs_
            nb_skipped += nb_skipped_
        except Exception as err:
            print(f'Generated an exception: {err} in dir: {mem2graph_dir_path}')
            #traceback.print_exc() # Print the traceback
            exit(1)
    
    print(f"✅ {nb_memgraphs} total graphs in the input mem2graph dataset dir paths have been loaded and checked.")
    print(f"💥 {nb_skipped} total graphs in the input mem2graph dataset dir paths have been skipped (deleted).")
    

if __name__ == "__main__":
    print("🚀 Running program...")

    start = datetime.now()

    cli = CLIArguments()

    load_env_file()
    main(cli)

    end = datetime.now()
    duration = end - start
    duration_human_readable = datetime_to_human_readable_str(duration)
    print("🏁 Program took: {0}".format(duration_human_readable))