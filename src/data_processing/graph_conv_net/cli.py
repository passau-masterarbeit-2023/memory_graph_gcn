# direct raw access to params
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

        Parse program arguments.
            -w max ml workers (threads for ML threads pool, -1 for illimited)
            -d debug
            -fad path to annotated DOT graph directory
            -fnd path to non-annotated DOT graph directory
            -fa load file containing annotated DOT graph
            -fn load file containing non-annotated DOT graph
        """
        parser = argparse.ArgumentParser(description='Program [ARGUMENTS]')
        parser.add_argument(
            '--debug', 
            action='store_true',
            help="debug, True or False"
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
            '--annotated_graph_dot_gv_dir_path', 
            type=str, 
            default=None,
            help="path to annotated DOT graph directory"
        )
        parser.add_argument(
            '-fnd',
            '--no_annotated_graph_dot_gv_dir_path',
            type=str,
            default=None,
            help="path to non-annotated DOT graph directory"
        )
        parser.add_argument(
            '-fa',
            '--annotated_graph_dot_gv_file_path',
            type=str,
            default=None,
            help="load file containing annotated DOT graph"
        )
        parser.add_argument(
            '-fn',
            '--no_annotated_graph_dot_gv_file_path',
            type=str,
            default=None,
            help="load file containing non-annotated DOT graph"
        )

        # save parsed arguments
        self.args = parser.parse_args()

        # log parsed arguments
        print("Parsed program params:")
        for arg in vars(self.args):
            print("{0}: {1}".format(
                arg, getattr(self.args, arg)
            ))