from abc import ABC, abstractmethod

from ..utils.utils import DATETIME_FORMAT, check_and_create_directory, datetime2str

import dotenv
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class BaseProgramParams(ABC):
    """
    Program parameters.
    This is a base class that contains the common parameters for all programs,
    like loggers, automatic path checking, etc.
    """
    app_name: str

    ### env vars
    # NOTE: all None values NEED to be overwritten by the .env file
    
    # default values
    MAX_ML_WORKERS: int
    DEBUG: bool
    RANDOM_SEED: int

    # logger
    COMMON_LOGGER_DIR_PATH: str

    RESULTS_LOGGER_DIR_PATH: str

    COMMON_LOGGER: logging.Logger 
    RESULTS_LOGGER: logging.Logger
    SAVE_RESULT_LOGS: bool

    def __init__(
            self, 
            app_name: str,
            load_program_argv : bool = True,
            dotenv_path: str | None = None,
            **kwargs
    ):
        """
        Program parameters. 
        If load_program_argv is True, the program parameters are loaded 
        from the command line arguments.
        Otherwise, the program parameters are loaded from the default values.
        if generate_important_log_file is True, the program will generate 
        an important log file with the current date and time.
        otherwise, the program will use the default log file.

        debug and generate_important_log_file parameter is used 
        only if load_program_argv is True.

        :param app_name: the name of the application
        :param result_writer: the result writer class
        :param load_program_argv: whether to load program arguments from argv
        :param dotenv_path: the path to the .env file
        """
        self.app_name = app_name

        self.__init_default_values()
        self.__load_env(dotenv_path)

        if load_program_argv:
            self.__parse_program_argv()
        else:
            self.SAVE_RESULT_LOGS = False
        
        print("🚧 DEBUG: {0}".format(self.DEBUG))

        self.__check_all_paths()
        self.__construct_log()

# ---------------------- initialise the values ----------------------

    def __init_default_values(self):
        """
        Init values as instance variables.
        """
        self.DEBUG = False
        self.MAX_ML_WORKERS = 10
        self.COMMON_LOGGER = logging.getLogger("common_logger")
        self.RESULTS_LOGGER = logging.getLogger("results_logger")

        self.data_origins_testing = None

# ---------------------- environnement loading ----------------------

    def __get_all_class_attribute_annotations(self):
        """
        Get all class attributes (annotations in Python).
        """
        # PYTHON MAGIC to get all the class variables
        all_annotations = {}

        # Iterate through the classes in the MRO in reverse order
        for cls in reversed(self.__class__.mro()):
            # Skip 'object', which has no annotations
            if cls is object:
                continue
            
            # Update the dictionary with the annotations from this class
            all_annotations.update(cls.__dict__.get('__annotations__', {}))

        # Now, 'all_annotations' should contain annotations from both the base and child classes.
        return all_annotations

    def __get_value_for_class_attribute(self, attribute_name: str):
        """
        Get the value for a given class attribute.
        NOTE: We need to define it since in python, the 'getattr' function
        only works for the last child class.
        """

        # Check if the attribute is in the instance dictionary (current class)
        if attribute_name in self.__dict__:
            return self.__dict__[attribute_name]

        # Iterate through the classes in the MRO in reverse order
        for cls in reversed(self.__class__.mro()):
            # Skip 'object', which has no attributes
            if cls is object:
                continue

            # Check if the class has the attribute
            if attribute_name in cls.__dict__:
                # Return the attribute value
                return cls.__dict__[attribute_name]

        # Raise an AttributeError if the attribute was not found
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{attribute_name}'")

    
    def __load_env(self, dotenv_path: str | None):
        """
        Load environment variables from .env file.
        Overwrite default values with values from .env file if they are defined there.
        """

        if dotenv_path is None:
            # determine project .env file, using the subclass python file location
            # and check recursively in parent directories for the first encountered .env file

            #  -------------------      python magic to get the path of the subclass python file- -------------------

            # 0 represents this line, 1 represents line at caller (init of this class), so the 3 represents the caller of the caller of the init of this class
            #import inspect
            #frame = inspect.stack()[3] 
            #module = inspect.getmodule(frame[0])
            #subclass_path = module.__file__

            # WARN : this not relialable, use the dotenv path instead
            # ------------------------------------------------------------
            

            tmp_folder = os.path.dirname(os.path.abspath(__file__))
            project_base_dir = tmp_folder
            while not os.path.exists(tmp_folder + "/.env"):
                tmp_folder = os.path.dirname(tmp_folder)
                if tmp_folder == "/":
                    print("BaseProgramParams ERROR: .env file not found, falling back to default values.")
                    dotenv.load_dotenv()
                    return
                    
                project_base_dir = tmp_folder

            # Load environment variables from .env file
            dotenv_path = os.path.join(project_base_dir, ".env")
        
        dotenv.load_dotenv(dotenv_path)

        # get all class attributes (annotations in Python)
        all_annotations = self.__get_all_class_attribute_annotations()
        
        # Overwrite default values with values from .env file if they are defined there
        # NOTE: cls.__annotations__ is a dictionary where the keys are the names of 
        #   the class variables and the values are their types
        for variable in all_annotations.keys():
            env_value = os.getenv(variable)
            if env_value is not None:
                # Convert to the appropriate type
                if all_annotations[variable] == bool:
                    env_value = env_value.lower() == 'true'
                elif all_annotations[variable] == int:
                    env_value = int(env_value)
                elif all_annotations[variable] == list:
                    env_value = env_value.split(',')
                elif all_annotations[variable] == str:
                    env_value = env_value
                setattr(self, variable, env_value)
                print("---> env var: %s, val: %s" % (variable, env_value))
        
        print("✅ Environment variables loaded.")


            
# ---------------------- program arg ----------------------

    def __parse_program_argv(self):
        """
        Parse program arguments.
        WARN: Do NOT parse program argv if running under pytest. 
        You can try, but it will fail. This is because argv will try to interpret 
        pytest arguments as program arguments, and give an unknown argument error.
        """
        if not self.__is_running_under_pytest():
            self._load_program_argv()
            self._consume_program_argv()
        else:
            self.cli_args = None # needed to avoid pytest error
    
    @abstractmethod # to implement in child
    def _load_program_argv(self):
        """
        Load given program arguments.
        """
        pass
    
    @abstractmethod # to implement in child
    def _consume_program_argv(self):
        """
        Consume given program arguments.
        """
        pass

# ---------------------- log ----------------------

    def __construct_log(self):
        """
        construct logger. Must be call after __load_program_argv()
        """
        # common formater
        common_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt=DATETIME_FORMAT.replace("_%f", "")
        )

        # ----------------- common logger -----------------
        # create common logger
        self.COMMON_LOGGER.setLevel(logging.DEBUG)

        # add RotatingFileHandler to common logger
        check_and_create_directory(self.COMMON_LOGGER_DIR_PATH + self.app_name)
        common_log_file_path = self.COMMON_LOGGER_DIR_PATH + self.app_name + "/common_log.log"
        common_log_file_handler = RotatingFileHandler(
            common_log_file_path, maxBytes=50000000, backupCount=5
        )
        logging_level = logging.DEBUG if self.DEBUG else logging.INFO
        common_log_file_handler.setLevel(logging_level)
        common_log_file_handler.setFormatter(common_formatter)
        self.COMMON_LOGGER.addHandler(common_log_file_handler)

        # add console handler to common logger
        common_log_console_handler = logging.StreamHandler(stream=sys.stdout)
        common_log_console_handler.setLevel(logging.INFO)
        common_log_console_handler.setFormatter(common_formatter)
        self.COMMON_LOGGER.addHandler(common_log_console_handler)

        # ----------------- results logger -----------------
        # create results logger
        self.RESULTS_LOGGER.setLevel(logging.DEBUG)

        # Result logger using file handler
        if self.SAVE_RESULT_LOGS:
            # when using results logger, do:
            check_and_create_directory(self.RESULTS_LOGGER_DIR_PATH + self.app_name)
            results_log_file_path = self.RESULTS_LOGGER_DIR_PATH + self.app_name + "/" + datetime2str(datetime.now()) + "_results.log"
        
            # inform user of the results logger path
            self.COMMON_LOGGER.info("⚓ Results logger path: %s" % results_log_file_path) 
        else:
            results_log_file_path = common_log_file_path
        results_log_file_handler = logging.FileHandler(results_log_file_path)
        results_log_file_handler.setLevel(logging.DEBUG)
        results_log_file_handler.setFormatter(common_formatter)
        self.RESULTS_LOGGER.addHandler(results_log_file_handler)

        # Result logger using console handler
        results_log_console_handler = logging.StreamHandler(stream=sys.stdout)
        results_log_console_handler.setLevel(logging.DEBUG)
        results_log_console_handler.setFormatter(common_formatter)
        self.RESULTS_LOGGER.addHandler(results_log_console_handler)

    
    def _log_program_params(self):
        """
        Log given program arguments.
        WARN: Must be CALLED LAST, manually in the child class, at the end of the __init__().
        """
        # get all class attributes (annotations in Python)
        all_annotations = self.__get_all_class_attribute_annotations()

        # log program params
        # NOTE: cls.__annotations__ is a dictionary where the keys are the names of 
        #   the class variables and the values are their types
        self.RESULTS_LOGGER.info("########## Program params:   [see below] ##########")
        for variable_name in all_annotations.keys():
            variable_value = self.__get_value_for_class_attribute(variable_name)
            self.RESULTS_LOGGER.info("%s: %s" % (variable_name, variable_value))
        self.RESULTS_LOGGER.info("########## Program params:   [see above] ##########")


# ---------------------- helper ----------------------

    def __is_running_under_pytest(self):
        """
        Check whether the code is running under pytest.
        """
        return 'pytest' in sys.modules

    def __check_all_paths(self):
        """
        Check if all paths are valid.
        """
        # get all parameters that are potentially paths
        __paths_vars_to_check = []
        for name, value in vars(self).items():
            if "PATH" in name:
                __paths_vars_to_check.append(name)

        # check if all paths exist
        for path_var_name in __paths_vars_to_check:
            path = getattr(self, path_var_name)
            if not os.path.exists(path):
                print("Program paths are NOT OK. Error in var: %s" % path_var_name)
                print("%s: %s" % (path_var_name, path))
                exit(1)
        
        print("✅ Program paths are OK.")