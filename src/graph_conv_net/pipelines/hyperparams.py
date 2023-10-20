from dataclasses import dataclass
from itertools import product
import json

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.results.base_result_writer import BaseResultWriter


@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class BaseHyperparams(object):
    index: int
    pipeline_name: PipelineNames

def add_hyperparams_to_result_writer(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    result_writer: BaseResultWriter,
):
    """
    Add the hyperparams to the result writer.
    """
    for field in hyperparams.__dataclass_fields__.keys():
        value = getattr(hyperparams, field)

        if type(value) is PipelineNames:
            value = value.value
        elif "node2vec" in field and not params.USE_NODE2VEC_EMBEDDING:
            # hyperparams from Node2Vec are not used, replace them with None
            value = "None"
        
        result_writer.set_result(
            field, 
            value,
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
    gcn_training_epochs: int

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class RandomForestPipeline(Node2VecHyperparams):
    random_forest_n_estimators: int # The number of trees in the forest.
    random_forest_n_jobs: int


def load_hyperparams_from_json(
    hyperparams_json_file_path: str,
):
    """
    Load JSON containing ML hyperparameters.
    """
    json_of_hyperparams = json.load(
        open(hyperparams_json_file_path, "r")
    )

    expected_keys = [
        "node2vec_dimensions_range",
        "node2vec_walk_length_range",
        "node2vec_num_walks_range",
        "node2vec_p_range",
        "node2vec_q_range",
        "node2vec_window_range",
        "node2vec_batch_words_range",
        "node2vec_workers_range",
        "randomforest_trees_range",
        "gcn_training_epochs_range",
    ]

    # perform a sanity check on the keys
    for key in expected_keys:
        assert key in json_of_hyperparams.keys(), (
            "ERROR: expected json-key ({0}) is missing from the hyperparams JSON file.".format(key)
        )
    
    return json_of_hyperparams

def generate_hyperparams(
    params: ProgramParams,
):
    """
    Generate the hyperparameters.
    """
    hyperparams_list: list[RandomForestPipeline | FirstGCNPipelineHyperparams] = []

    # Generate the Cartesian product for node2vec parameters
    json_hyperparams = load_hyperparams_from_json(
        params.HYPERPARAMS_JSON_FILE_PATH,
    )
    node2vec_params_product = product(
        json_hyperparams["node2vec_dimensions_range"],
        json_hyperparams["node2vec_walk_length_range"],
        json_hyperparams["node2vec_num_walks_range"],
        json_hyperparams["node2vec_p_range"],
        json_hyperparams["node2vec_q_range"],
        json_hyperparams["node2vec_window_range"],
        json_hyperparams["node2vec_batch_words_range"],
        json_hyperparams["node2vec_workers_range"],
    )

    randomforest_trees_range = json_hyperparams["randomforest_trees_range"]
    gcn_training_epochs_range = json_hyperparams["gcn_training_epochs_range"]

    # Initialize hyperparams_list and hyperparam_index
    hyperparams_list = []
    hyperparam_index = 0

    # Iterate through the Cartesian product
    for node2vec_params in node2vec_params_product:
        (
            node2vec_dimensions,
            node2vec_walk_length,
            node2vec_num_walks,
            node2vec_p,
            node2vec_q,
            node2vec_window,
            node2vec_batch_words,
            node2vec_workers
        ) = node2vec_params
        
        if PipelineNames.RandomForestPipeline.value in params.cli_args.args.pipelines:
            for nb_trees in randomforest_trees_range:
                randforest_hyperparams = RandomForestPipeline(
                    pipeline_name=PipelineNames.RandomForestPipeline,
                    index=hyperparam_index,
                    node2vec_dimensions=node2vec_dimensions,
                    node2vec_walk_length=node2vec_walk_length,
                    node2vec_num_walks=node2vec_num_walks,
                    node2vec_p=node2vec_p,
                    node2vec_q=node2vec_q,
                    node2vec_window=node2vec_window,
                    node2vec_batch_words=node2vec_batch_words,
                    node2vec_workers=node2vec_workers,
                    random_forest_n_estimators=nb_trees,
                    random_forest_n_jobs=params.NB_RANDOM_FOREST_JOBS,
                )
                hyperparams_list.append(randforest_hyperparams)
                hyperparam_index += 1

        if PipelineNames.FirstGCNPipeline.value in params.cli_args.args.pipelines:
            for gcn_training_epochs in gcn_training_epochs_range:
                gcn_hyperparams = FirstGCNPipelineHyperparams(
                    pipeline_name=PipelineNames.FirstGCNPipeline,
                    index=hyperparam_index,
                    node2vec_dimensions=node2vec_dimensions,
                    node2vec_walk_length=node2vec_walk_length,
                    node2vec_num_walks=node2vec_num_walks,
                    node2vec_p=node2vec_p,
                    node2vec_q=node2vec_q,
                    node2vec_window=node2vec_window,
                    node2vec_batch_words=node2vec_batch_words,
                    node2vec_workers=node2vec_workers,
                    gcn_training_epochs=gcn_training_epochs,
                )
                hyperparams_list.append(gcn_hyperparams)
                hyperparam_index += 1
    
    return hyperparams_list