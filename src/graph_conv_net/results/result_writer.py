from datetime import datetime
import os

from graph_conv_net.utils.utils import DATETIME_FORMAT

from ..results.base_result_writer import BaseResultWriter

class ResultWriter(BaseResultWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """

    # NOTE: The result writer passed to the program params class should only have
    # the :pipeline_name: parameter in its init args.
    def __init__(self, pipeline_name: str):

        result_dir_path = os.environ.get("RESULTS_LOGGER_DIR_PATH")
        if result_dir_path is None:
            raise Exception("ERROR: RESULTS_LOGGER_DIR_PATH env var not set.")
        elif not os.path.exists(result_dir_path):
            raise Exception("ERROR: RESULTS_LOGGER_DIR_PATH env var does not point to a valid path.")

        super().__init__(
            result_dir_path,
            pipeline_name,
        )