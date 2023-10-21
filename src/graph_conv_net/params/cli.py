# direct raw access to params
import os
import sys
import argparse

from graph_conv_net.embedding.node_to_vec_enums import NodeEmbeddingType
from graph_conv_net.pipelines.pipelines import PipelineNames

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

        Parse program arguments.
            -w max ml workers (threads for ML threads pool, -1 for illimited)
            -d debug
            -fad path to annotated DOT graph directory
            -fa load file containing annotated DOT graph
        """
        parser = argparse.ArgumentParser(description='Program [ARGUMENTS]')
        parser.add_argument(
            '-d',
            '--debug', 
            action='store_true',
            help="Run in debug mode."
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Dry run. Do not launch compute instances."
        )
        parser.add_argument(
            '-w',
            '--max_ml_workers', 
            type=int, 
            default=None,
            help="max ml workers (threads for ML threads pool, -1 for illimited)"
        )
        parser.add_argument(
            '--dir_annotated_graph_dot_gv_path', 
            type=str, 
            default=None,
            help="path to annotated DOT graph directory"
        )
        parser.add_argument(
            '-p',
            '--pipelines',
            type=str,
            nargs='+',  # this allows multiple values for this argument
            choices=[e.value for e in PipelineNames],  # limit choices to the Enum values
            help=f"List of pipeline names: {[e.value for e in PipelineNames]}"
        )
        parser.add_argument(
            '-e',
            '--node-embedding',
            type=str,
            default=None,
            choices=[e.value for e in NodeEmbeddingType],
            help=f"Node embedding type: {[e.value for e in NodeEmbeddingType]}. Custom comment embedding depends on the input Mem2Graph dataset."
        )
        parser.add_argument(
            '-n',
            '--nb-input-graphs',
            type=int,
            default=None,
            help="Number of input graphs to use. If None, use all available graphs."
        )
        parser.add_argument(
            '-a',
            '--all-mem2graph-datasets',
            action='store_true',
            help="Use all Mem2Graph datasets. Uses env var ALL_MEM2GRAPH_DATASET_DIR_PATH which specifies where are the Mem2Graph-generated graphs."
        )
        parser.add_argument(
            '-b',
            '--parallel-batch-size',
            type=int,
            default=None,
            help="Batch size for parallel processing of graphs."
        )

        # save parsed arguments
        self.args = parser.parse_args()

        # set os env var DEBUG to 0 or 1
        if self.args.debug:
            os.environ["DEBUG"] = "1"
        else:
            os.environ["DEBUG"] = "0"

        # pipelines to launch
        if self.args.pipelines:
            for pipeline in self.args.pipelines:
                if pipeline == PipelineNames.FirstGCNPipeline.value:
                    print("🔷 Launching First GCN Pipeline")
                elif pipeline == PipelineNames.RandomForestPipeline.value:
                    print("🔷 Launching Random Forest Pipeline")
                else:
                    print(f"Unknown pipeline: {pipeline}")
                    exit(1)
        else:
            print("🔴 No pipelines specified. Stopping...")
            exit(1)
        
        # node embedding types
        if self.args.node_embedding is not None:
            if self.args.node_embedding == NodeEmbeddingType.Node2Vec.value:
                print("🔷 Using Node2Vec node embedding")
            elif self.args.node_embedding == NodeEmbeddingType.CustomCommentEmbedding.value:
                print("🔷 Using custom node embedding stored in comment fields of graph nodes.")
            elif self.args.node_embedding == NodeEmbeddingType.Node2VecAndComment.value:
                print("🔷 Using both Node2Vec and custom node embedding stored in comment fields of graph nodes.")
            else:
                print(f"Unknown node embedding type: {self.args.node_embedding}")
                exit(1)
        else:
            print("🔷 No node embedding types specified. Iterating over all available embeddings combinations.")

        # nb input graphs
        if self.args.nb_input_graphs:
            print(f"🔷 Using {self.args.nb_input_graphs} input graphs")
        else:
            print("🔷 Using all available input graphs")

        # all Mem2Graph datasets
        if self.args.all_mem2graph_datasets:
            print("🔷 Using all Mem2Graph datasets to generate compute instances")

        # log parsed arguments
        print("Parsed program params:")
        for arg in vars(self.args):
            print("{0}: {1}".format(
                arg, getattr(self.args, arg)
            ))