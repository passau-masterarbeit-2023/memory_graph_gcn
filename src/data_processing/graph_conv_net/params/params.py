from pathlib import Path
from commons.params.base_program_params import BaseProgramParams
from commons.params.app_params import AppName

from ..cli import CLIArguments

class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    cli_args: CLIArguments
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # data
    ANNOTATED_GRAPH_DOT_GV_DIR_PATH: str
    NO_ANNOTATION_GRAPH_DOT_GV_DIR_PATH: str

    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        # determine dotenv path
        # NOTE: the .env file is in the same path as this current file
        dotenv_path = Path(__file__).parent.joinpath('.env')
        assert(dotenv_path.exists() and dotenv_path.is_file(), "ERROR: .env file not found: {0}".format(dotenv_path))

        super().__init__(AppName.GCN_ML, load_program_argv, debug, dotenv_path=dotenv_path)

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
        
        if self.cli_args.args.no_annotated_graph_dot_gv_dir_path is not None:
            self.NO_ANNOTATION_GRAPH_DOT_GV_DIR_PATH = self.cli_args.args.no_annotated_graph_dot_gv_dir_path
            assert isinstance(self.NO_ANNOTATION_GRAPH_DOT_GV_DIR_PATH, str)
    