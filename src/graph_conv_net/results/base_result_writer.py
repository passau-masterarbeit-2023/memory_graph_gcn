import csv
from datetime import datetime
from enum import Enum
import os
import json
import platform
from graph_conv_net.utils.utils import DATETIME_FORMAT
import psutil
from typing import Optional

class SaveFileFormat(Enum):
    CSV = "csv"
    JSON = "json"
    BOTH = "both"

class BaseResultWriter(object):
    """
    This class is used to write the results to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to the results.
    """
    save_dir_path: str
    results: dict[str, Optional[str]]
    __already_written_results: bool # Flag, only write results once

    def __init__(
            self, 
            save_dir_path: str, 
            pipeline_name: str = "unnamed-pipeline"
        ):
        self.save_dir_path = save_dir_path
        self.results = {}
        self.__save_system_info()

        self.__already_written_results = False

        # initialization
        self.set_result("pipeline_name", pipeline_name)
        self.set_result("start_time", datetime.now().strftime(DATETIME_FORMAT))

    def set_result(self, field: str, value: Optional[str]) -> None:
        """
        Set a result field with the given value.
        This field will be written to the CSV file when calling :save_results_to_csv:.
        """
        self.results[field] = value

    def save_results_to_file(self, save_file_format: SaveFileFormat = SaveFileFormat.CSV) -> None:
        """
        Write the results to the save file.
        It's possible to choose the format of the save file (CSV, JSON).
        WARNING: This function MUST only be called once.
        """
        if self.__already_written_results:
            raise ValueError("ClassificationResultsWriter: Results already written")
        
        # add end time and duration
        end_time = datetime.now()
        self.set_result("end_time", end_time.strftime(DATETIME_FORMAT))
        
        start_time_str = self.results["start_time"]
        if start_time_str is None:
            raise ValueError("ClassificationResultsWriter: start_time is None.")

        start_time = datetime.strptime(
            start_time_str, DATETIME_FORMAT
        )
        duration = end_time - start_time
        duration_in_seconds = duration.total_seconds()
        duration_hour_min_sec = "{} total sec ({:02d}h {:02d}m {:02d}s)".format(
            duration_in_seconds,
            int(duration_in_seconds // 3600),
            int((duration_in_seconds % 3600) // 60),
            int(duration_in_seconds % 60),
        )
        self.set_result("duration", duration_hour_min_sec)

        # save results to file
        if save_file_format == SaveFileFormat.CSV:
            self.__save_results_to_csv()
        elif save_file_format == SaveFileFormat.JSON:
            self.__save_results_to_json()
        elif save_file_format == SaveFileFormat.BOTH:
            self.__save_results_to_csv()
            self.__save_results_to_json()
        else:
            raise ValueError(f"Unknown save file format: {save_file_format}")
    
        self.__already_written_results = True
        
    def __save_results_to_csv(self) -> None:
        """
        Write the results to the CSV file.
        """
        csv_file_path = self.save_dir_path + "/all_results.csv"
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results.keys())
            # write header only if file is empty or does not exist
            if not file_exists or os.stat(csv_file_path).st_size == 0:
                writer.writeheader()
            writer.writerow(self.results)

    def __save_results_to_json(self) -> None:
        """
        Write the results to the JSON file.
        """
        current_time = datetime.now().strftime(DATETIME_FORMAT)
        pipeline_name = self.results["pipeline_name"]
        if self.results.get("pipeline_name") is None:
            pipeline_name = "unnamed-pipeline"
        assert pipeline_name is not None
        json_new_file_path = self.save_dir_path + "/" + pipeline_name + "_" + current_time + ".csv"
        json.dump(self.results, open(json_new_file_path, 'w'), indent=4)

    def print_results(self) -> None:
        for key, value in self.results.items():
            print(f"{key}: {value}")
    
    
    # utility functions
    def __save_system_info(self):
        """
        Save machine information to the result saver.
        """
        # Get system information
        uname = platform.uname()

        self.set_result("system", uname.system)
        self.set_result("node_name", uname.node)
        self.set_result("release", uname.release)
        self.set_result("version", uname.version)
        self.set_result("machine", uname.machine)
        self.set_result("processor", uname.processor)

        # Get CPU information
        self.set_result("physical_cores", str(psutil.cpu_count(logical=False)))
        self.set_result("total_cores", str(psutil.cpu_count(logical=True)))

        # Get memory information
        mem_info = psutil.virtual_memory()
        self.set_result("total_memory", str(mem_info.total))
        self.set_result("available_memory", str(mem_info.available))
