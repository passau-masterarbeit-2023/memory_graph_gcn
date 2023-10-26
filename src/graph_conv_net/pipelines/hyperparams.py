from dataclasses import dataclass
from itertools import product
import json
import os
from graph_conv_net.embedding.node_to_vec_enums import NodeEmbeddingType

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.results.base_result_writer import BaseResultWriter
from graph_conv_net.utils.utils import str2enum


@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class BaseHyperparams(object):
    index: int
    pipeline_name: PipelineNames
    input_mem2graph_dataset_dir_path: str
    node_embedding: NodeEmbeddingType

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
        if type(value) is NodeEmbeddingType:
            # this is a more complex case, because we want to replace "comment" by the real name of the embedding in the comment fields
            value = value.to_string(
                hyperparams.input_mem2graph_dataset_dir_path
            )
        elif "node2vec" in field and not hyperparams.node_embedding.is_using_node2vec():
            # hyperparams from Node2Vec are not used, replace them with None
            value = "None"
        elif "random_forest" in field and hyperparams.pipeline_name != PipelineNames.ClassicMLPipeline:
            # hyperparams from RandomForest are not used, replace them with None
            value = "None"
        elif "first_gcn" in field and hyperparams.pipeline_name != PipelineNames.GCNPipeline:
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
    first_gcn_training_epochs: int

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

def get_mem2graph_dataset_dir_paths(
    params: ProgramParams,
):
    """
    Determine the Mem2Graph dataset directory paths.
    """
    mem2graph_dataset_dir_paths = []
    if params.cli.args.all_mem2graph_datasets:
        # Using all Mem2Graph datasets
        params.COMMON_LOGGER.info("🔷 Looking for Mem2Graph dataset directories in {0}...".format(
            params.ALL_MEM2GRAPH_DATASET_DIR_PATH
        ))

        for dir_name in os.listdir(params.ALL_MEM2GRAPH_DATASET_DIR_PATH):
            if ".gitignore" not in dir_name:
                mem2graph_dataset_dir_paths.append(
                    os.path.join(params.ALL_MEM2GRAPH_DATASET_DIR_PATH, dir_name)
                )
    elif params.cli.args.input_dir_path:
        # Using input Mem2Graph dataset
        mem2graph_dataset_dir_paths.append(
            params.cli.args.input_dir_path
        )
    else:
        params.COMMON_LOGGER.error("🚩 PANIC: no Mem2Graph dataset directory path specified.")
        exit(1)

    params.COMMON_LOGGER.info("🗃 Found {0} Mem2Graph dataset directories.".format(
        str(len(mem2graph_dataset_dir_paths))
    ))
    
    return mem2graph_dataset_dir_paths

def generate_hyperparams(
    params: ProgramParams,
):
    """
    Generate the hyperparameters.
    """

    # get Mem2Graph dataset path list
    mem2graph_dataset_dir_paths = get_mem2graph_dataset_dir_paths(
        params,
    )

    # Generate the Cartesian product for node2vec parameters
    json_hyperparams = load_hyperparams_from_json(
        params.HYPERPARAMS_JSON_FILE_PATH,
    )

    # Determine which node embeddings
    node_embedding_types: list[NodeEmbeddingType] = []
    if params.cli.args.node_embedding is not None:
        selected_node_embedding = str2enum(
            params.cli.args.node_embedding,
            NodeEmbeddingType,
        )
        assert type(selected_node_embedding) is NodeEmbeddingType, (
            "ERROR: node embedding is not a NodeEmbeddingType: {0}".format(
                type(selected_node_embedding)
            )
        )
        node_embedding_types.append(
            selected_node_embedding
        )
    else:
        # Nothing specified, use all node embeddings
        _all_embeddings = NodeEmbeddingType.get_list_of_embeddings()
        print("len(_all_embeddings): {0}".format(len(_all_embeddings)))
        node_embedding_types.extend(
            _all_embeddings
        )
    assert len(node_embedding_types) > 0, (
        "ERROR: no node embedding specified."
    )
    print("🔷 node_embedding_types: {0}".format(node_embedding_types))

    node2vec_params_product = product(
        node_embedding_types,
        mem2graph_dataset_dir_paths,
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
    hyperparams_list: list[RandomForestPipeline | FirstGCNPipelineHyperparams | BaseHyperparams] = []
    hyperparam_index = 0

    # Iterate through the Cartesian product
    for node2vec_params in node2vec_params_product:
        (
            node_embedding,
            input_mem2graph_dataset_dir_path,
            node2vec_dimensions,
            node2vec_walk_length,
            node2vec_num_walks,
            node2vec_p,
            node2vec_q,
            node2vec_window,
            node2vec_batch_words,
            node2vec_workers
        ) = node2vec_params

        if PipelineNames.ClassicMLPipeline.value in params.cli.args.pipelines:
            for nb_trees in randomforest_trees_range:
                randforest_hyperparams = RandomForestPipeline(
                    index=hyperparam_index,
                    pipeline_name=PipelineNames.ClassicMLPipeline,
                    input_mem2graph_dataset_dir_path=input_mem2graph_dataset_dir_path,
                    node_embedding=node_embedding,
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

        if PipelineNames.GCNPipeline.value in params.cli.args.pipelines:
            for gcn_training_epochs in gcn_training_epochs_range:
                gcn_hyperparams = FirstGCNPipelineHyperparams(
                    index=hyperparam_index,
                    pipeline_name=PipelineNames.GCNPipeline,
                    input_mem2graph_dataset_dir_path=input_mem2graph_dataset_dir_path,
                    node_embedding=node_embedding,
                    node2vec_dimensions=node2vec_dimensions,
                    node2vec_walk_length=node2vec_walk_length,
                    node2vec_num_walks=node2vec_num_walks,
                    node2vec_p=node2vec_p,
                    node2vec_q=node2vec_q,
                    node2vec_window=node2vec_window,
                    node2vec_batch_words=node2vec_batch_words,
                    node2vec_workers=node2vec_workers,
                    first_gcn_training_epochs=gcn_training_epochs,
                )
                hyperparams_list.append(gcn_hyperparams)
                hyperparam_index += 1
    
    # feature evaluation pipelines
    if PipelineNames.FeatureEvaluationPipeline.value in params.cli.args.pipelines:
        for input_mem2graph_dataset_dir_path in mem2graph_dataset_dir_paths:
            for node_embedding in node_embedding_types:
                feature_eval_hyperparams = BaseHyperparams(
                    index=hyperparam_index,
                    pipeline_name=PipelineNames.FeatureEvaluationPipeline,
                    input_mem2graph_dataset_dir_path=input_mem2graph_dataset_dir_path,
                    node_embedding=node_embedding,
                )
                hyperparams_list.append(feature_eval_hyperparams)
                hyperparam_index += 1
    
    return hyperparams_list