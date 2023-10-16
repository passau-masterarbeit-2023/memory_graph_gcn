from enum import Enum
from typing import Optional
from .base_result_writer import BaseResultWriter

from typing import TypeVar, Generic, Type

ResultWriter = TypeVar('ResultWriter', bound=BaseResultWriter)  # CustomResultWriter should be a subtype of BaseResultWriter
PipelineNamesEnum = TypeVar('PipelineNamesEnum', bound=Enum)  # PipelineNamesEnum should be a subtype of Enum

class BaseResultsManager(Generic[PipelineNamesEnum, ResultWriter]):
    """
    Manager of different result writers (one per pipeline).
    WARN : Only work with ResultWriter subclasses, with only one argument in the constructor: pipeline_name.
    """
    result_writer_dict: dict[PipelineNamesEnum, ResultWriter]
    csv_results_path: str

    def __init__(self, pipeline_names : Type[PipelineNamesEnum], ResultWriterType: Type[ResultWriter]):
        """
        WARN : Only work with ResultWriter subclasses, with only one argument in the constructor: pipeline_name.
        """
        # result keepers
        self.__create_results_keepers(pipeline_names, ResultWriterType)
    
    def __repr__(self):
        return f"ResultsManager instance of type: {type(self).__name__}"

    def __create_results_keepers(self, pipeline_names : Type[PipelineNamesEnum], ResultWriterType: Type[ResultWriter]):
        """
        Create results keepers.
        """
        self.result_writer_dict = {}
        for pipeline_name in pipeline_names:
            self.result_writer_dict[pipeline_name] = ResultWriterType(
                str(pipeline_name)
            ) # NOTE: Child classes should call BaseWriter constructor with the other arguments needed

    
    def set_result_forall(
        self, field: str, value: Optional[str]    
    ) -> None:
        """
        Set a result for all result keepers.
        """
        for classification_result_writer in self.result_writer_dict.values():
            classification_result_writer.set_result(field, value)
    
    def set_result_for(
        self, pipeline_name: PipelineNamesEnum, field: str, value: Optional[str]    
    ) -> None:
        """
        Set a result for a specific result keeper.
        """
        self.result_writer_dict[pipeline_name].set_result(field, value)

    def save_results_forall(self) -> None:
        """
        Write results for all result keepers.
        """
        for classification_result_writer in self.result_writer_dict.values():
            classification_result_writer.__save_results_to_csv()

    def save_results_for(self, pipeline_name: PipelineNamesEnum) -> None:
        """
        Write results for a specific result keeper.
        """
        self.result_writer_dict[pipeline_name].__save_results_to_csv()
    
    def get_result_writer_for(self, pipeline_name: PipelineNamesEnum) -> ResultWriter:
        """
        Get a specific result keeper.
        """
        return self.result_writer_dict[pipeline_name]
