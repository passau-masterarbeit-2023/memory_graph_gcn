"""
This file checks the validity of the gv files in the given directory.
This includes that the files can be loaded into a graph,
and that each node has a comment attribute, whose length is equal to the number of features
provided in the graph top level comment.
"""

from datetime import datetime
import os
from dotenv import load_dotenv

from graph_conv_net.data_loading.data_loading import load_annotated_graph
from graph_conv_net.data_loading.file_loading import find_gv_files, find_pickle_files
from graph_conv_net.graph.memgraph import MemGraph
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from graph_conv_net.utils.utils import datetime_to_human_readable_str

# -------------------- CLI arguments -------------------- #
import sys
import argparse

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

def load_graph(gv_file_path: str) -> MemGraph | None:
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
    return memgraph

def load_and_check_graph_in_dir(dir_path: str): 
    gv_file_paths = find_gv_files(dir_path)
    if len(gv_file_paths) == 0:
        print("󰡯 No .gv files found in {0}, skipping...".format(dir_path))
        return 0, 0

    memgraphs: list[MemGraph] = []
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
            try:
                assert path is not None, (
                    f"ERROR: path should not be None."
                )
                assert path.endswith(".gv"), (
                    f"ERROR: path should end with '.gv', but got {path}."
                )

                memgraph = future.result()
                if memgraph:
                    memgraphs.append(memgraph)
                else: 
                    nb_skipped += 1
            except Exception as err:
                print(f'Generated an exception: {err}\n with graph at path: {path}')
                nb_skipped += 1

            # Update the progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()

    # Check that the embedding length is consistent across all graphs in the dir
    print(" 🔘 Checking embedding length of graphs in {0}...".format(dir_path))
    nb_feature_first = len(memgraphs[0].custom_embedding_fields)

    # Create a TQDM progress bar
    progress_bar = tqdm(memgraphs, desc="Checking Embedding Length", dynamic_ncols=True)

    for memgraph in progress_bar:
        if len(memgraph.custom_embedding_fields) != nb_feature_first:
            # Close the progress bar before raising an exception
            progress_bar.close()
            raise ValueError(
                "🚩 PANIC: embedding length is not consistent across all graphs in the dir. "
                f"Expected: {nb_feature_first} (from first graph in dir, with path: {memgraphs[0].gv_file_path}), but got: {len(memgraph.custom_embedding_fields)}"
                f", for graph at path: {memgraph.gv_file_path}"
            )

    return len(memgraphs), nb_skipped
        
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


    # get all .gv files in the dir paths
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
            print(f'Generated an exception: {err} with graph at path: {mem2graph_dir_path}')
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