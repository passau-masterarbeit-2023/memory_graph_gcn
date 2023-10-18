# direct raw access to params
import sys
import argparse

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
            '--debug', 
            action='store_true',
            help="Run in debug mode."
        )
        parser.add_argument(
            '-w',
            '--max_ml_workers', 
            type=int, 
            default=None,
            help="max ml workers (threads for ML threads pool, -1 for illimited)"
        )
        parser.add_argument(
            '-fad',
            '--dir_annotated_graph_dot_gv_path', 
            type=str, 
            default=None,
            help="path to annotated DOT graph directory"
        )
        parser.add_argument(
            '-fa',
            '--file_annotated_graph_dot_gv_path',
            type=str,
            default=None,
            help="load file containing annotated DOT graph"
        )
        parser.add_argument(
            '-e',
            '--embedding',
            action='store_true',
            help="Add additional node embedding to the GCN model. Note that the embedding must be provided in the 'comment' field of the DOT graph."
        )
        parser.add_argument(
            '-p',
            '--pipelines',
            type=str,
            nargs='+',  # this allows multiple values for this argument
            choices=[e.value for e in PipelineNames],  # limit choices to the Enum values
            help=f"List of pipeline names: {[e.value for e in PipelineNames]}"
        )

        # save parsed arguments
        self.args = parser.parse_args()

        # pipelines to launch
        if self.args.pipelines:
            for pipeline in self.args.pipelines:
                if pipeline == PipelineNames.FirstGCNPipeline.value:
                    print(" 🔷 Launching First GCN Pipeline")
                elif pipeline == PipelineNames.RandomForestPipeline.value:
                    print(" 🔷 Launching Random Forest Pipeline")
                else:
                    print(f"Unknown pipeline: {pipeline}")
                    exit(1)
        else:
            print(" 🔴 No pipelines specified. Stopping...")
            exit(1)

        # log parsed arguments
        print("Parsed program params:")
        for arg in vars(self.args):
            print("{0}: {1}".format(
                arg, getattr(self.args, arg)
            ))