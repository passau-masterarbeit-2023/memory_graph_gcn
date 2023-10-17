from enum import Enum
from dataclasses import dataclass

from graph_conv_net.results.base_result_writer import BaseResultWriter

class PipelineNames(Enum):
    FirstGCNPipeline = "first-gcn-pipeline"
    RandomForestPipeline = "random-forest-pipeline"

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class BaseHyperparams(object):
    index: int
    pipeline_name: PipelineNames

def add_hyperparams_to_result_writer(
    result_writer: BaseResultWriter,
    hyperparams: BaseHyperparams,
):
    """
    Add the hyperparams to the result writer.
    """
    for field in hyperparams.__dataclass_fields__.keys():
        result_writer.set_result(
            field, 
            getattr(hyperparams, field)
        )

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class Node2VecHyperparams(BaseHyperparams):
    """
    This class contains the hyperparameters for the first GCN pipeline.
    """

    # Node2Vec hyperparams
    node2vec_dimensions: int
    node2vec_walk_length: int
    node2vec_num_walks: int
    node2vec_p: float
    node2vec_q: float
    node2vec_window: int
    node2vec_batch_words: int
    node2vec_workers: int

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class FirstGCNPipelineHyperparams(Node2VecHyperparams):
    training_epochs: int

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class RandomForestPipeline(Node2VecHyperparams):
    random_forest_n_estimators: int # The number of trees in the forest.
    random_forest_n_jobs: int