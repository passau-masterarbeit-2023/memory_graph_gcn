import os
from copy import deepcopy

from graph_conv_net.embedding.node_to_vec_enums import NodeEmbeddingType

#from research_base.params.base_program_params import BaseProgramParams
from .base_program_params import BaseProgramParams

from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.results.result_writer import ResultWriter

from .cli import CLIArguments

class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    cli: CLIArguments
    app_name : str = "GCN_ML"

    nb_pipeline_runs: int = 1
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # ENV vars. WARN: Don't forget to add them to the .env file
    DRY_RUN: bool
    ALL_MEM2GRAPH_DATASET_DIR_PATH: str
    PICKLE_DATASET_DIR_PATH: str
    RESULT_SAVE_FILE_FORMAT: str
    HYPERPARAMS_JSON_FILE_PATH: str
    NB_RANDOM_FOREST_JOBS: int
    PARALLEL_PIPELINE_BATCH_SIZE: int
    PICKLE_CACHED_NODE2VEC_EMBEDDINGS: str

    def __init__(
            self, 
            load_program_argv : bool = True,
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
            result_writer = ResultWriter,
            load_program_argv = load_program_argv, 
            dotenv_path = dotenv_path
        )

        # to be done last
        self._log_program_params()
    
    
    def _load_program_argv(self):
        """
        Load given program arguments.
        """
        self.cli: CLIArguments = CLIArguments()


    def _consume_program_argv(self):
        """
        Consume given program arguments.
        """
        # NOTE: This is actually not needed, since the CLIArguments class already does this
        if self.cli.args.debug is not None:
            self.DEBUG = self.cli.args.debug
            assert isinstance(self.DEBUG, bool)

        if self.cli.args.max_ml_workers is not None:
            self.MAX_ML_WORKERS = int(self.cli.args.max_ml_workers)
            assert isinstance(self.MAX_ML_WORKERS, int)
        
        if self.cli.args.dry_run is not None:
            self.DRY_RUN = self.cli.args.dry_run
            assert isinstance(self.DRY_RUN, bool)
        else:
                self.DRY_RUN = False
        
        if self.cli.args.parallel_batch_size is not None:
            self.PARALLEL_PIPELINE_BATCH_SIZE = self.cli.args.parallel_batch_size
            assert isinstance(self.PARALLEL_PIPELINE_BATCH_SIZE, int)
        
