import os
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.results.result_writer import ResultsWriter
from research_base.results.base_result_writer import BaseResultWriter
from research_base.params.base_program_params import BaseProgramParams

from ..cli import CLIArguments

class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    cli_args: CLIArguments
    app_name : str = "GCN_ML"
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # ENV vars. WARN: Don't forget to add them to the .env file
    ANNOTATED_GRAPH_DOT_GV_DIR_PATH: str
    PICKLE_DATASET_DIR_PATH: str

    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        # determine dotenv path
        # NOTE: the .env file is in the same path as this current file, else, in the parent folder
        # Initialize dotenv_path to None
        dotenv_path = None

        # Start from the current directory
        current_dir = os.path.dirname(__file__)

        # Loop to walk upwards in the directory tree
        parent_dir_level = 0
        while current_dir != '/' and parent_dir_level < 3:
            potential_dotenv_path = os.path.join(current_dir, '.env')
            if os.path.exists(potential_dotenv_path):
                dotenv_path = potential_dotenv_path
                break
            # Move up to the parent directory
            current_dir = os.path.dirname(current_dir)
            parent_dir_level += 1

        if dotenv_path is None:
            raise Exception("ERROR: .env file not found.")

        super().__init__(
            app_name = self.app_name,
            pipeline_names_enum = PipelineNames,
            result_writer = ResultsWriter,
            load_program_argv = load_program_argv,
            debug = debug, 
            dotenv_path = dotenv_path
        )

        # to be done last
        self._log_program_params()
    
    
    def _load_program_argv(self):
        """
        Load given program arguments.
        """
        self.cli_args: CLIArguments = CLIArguments()


    def _consume_program_argv(self):
        """
        Consume given program arguments.
        """
        if self.cli_args.args.debug is not None:
            self.DEBUG = self.cli_args.args.debug
            assert isinstance(self.DEBUG, bool)

        if self.cli_args.args.max_ml_workers is not None:
            self.MAX_ML_WORKERS = int(self.cli_args.args.max_ml_workers)
            assert isinstance(self.MAX_ML_WORKERS, int)
        
        if self.cli_args.args.annotated_graph_dot_gv_dir_path is not None:
            self.ANNOTATED_GRAPH_DOT_GV_DIR_PATH = self.cli_args.args.annotated_graph_dot_gv_dir_path
            assert isinstance(self.ANNOTATED_GRAPH_DOT_GV_DIR_PATH, str)
        