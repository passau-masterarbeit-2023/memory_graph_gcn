from copy import deepcopy
import json
import os
from sklearn.preprocessing import StandardScaler
from graph_conv_net.embedding.node_to_vec import generate_node_embedding
from graph_conv_net.embedding.node_to_vec_enums import get_graph_comment
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.common.pipeline_common import common_load_labelled_graph, common_pipeline_end
from graph_conv_net.pipelines.hyperparams import BaseHyperparams, Node2VecHyperparams, add_hyperparams_to_result_writer
from graph_conv_net.pipelines.pipelines import FeatureEvaluationSubpipelineNames
from graph_conv_net.pipelines.random_forest.random_forest_pipeline import SamplesAndLabels
from graph_conv_net.results.base_result_writer import BaseResultWriter, SaveFileFormat
from graph_conv_net.utils.debugging import dp
from graph_conv_net.utils.utils import datetime_to_human_readable_str
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from enum import Enum
import networkx as nx


def __determine_feature_corr_matrix_save_file_path(
    hash_key: str,
) -> str:
    """
    Determine the path of the save file.
    """
    save_dir_path = os.environ.get("RESULTS_LOGGER_DIR_PATH")
    corr_matrix_dir_path = f"{save_dir_path}/feature_corr"

    # check if directory exists, else create it
    if not os.path.exists(corr_matrix_dir_path):
        os.makedirs(corr_matrix_dir_path)

    save_file_path = f"{corr_matrix_dir_path}/feature_correlation_matrix_-k_{hash_key}.png"
    return save_file_path

def __get_feature_names_from_comment(
        hyperparams: BaseHyperparams,
    ):
    """
    Get the feature names from the graph comment.
    """

    first_graph_file_path = os.path.join(
        hyperparams.input_mem2graph_dataset_dir_path,
        os.listdir(hyperparams.input_mem2graph_dataset_dir_path)[0],
    )

    comment_object = get_graph_comment(first_graph_file_path)
    dp("first_graph comment object: {0}".format(json.dumps(comment_object)))

    assert "embedding-fields" in comment_object.keys()
    embeddings_fields: list[str] = comment_object["embedding-fields"]
    assert isinstance(embeddings_fields, list)
    assert isinstance(embeddings_fields[0], str) 
    assert len(embeddings_fields) > 0

    return embeddings_fields

def feature_evaluation_pipeline(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    results_writer: BaseResultWriter,
):
    """
    A pipeline for feature evaluation.
    """

    # load data
    print(" 🔘 Loading data...")
    labelled_graphs = common_load_labelled_graph(
        params,
        hyperparams,
        results_writer,
    )

    # get graph comment (for custom comment feature names)
    embeddings_fields = __get_feature_names_from_comment(
        hyperparams
    )

    assert len(labelled_graphs) > 0, "ERROR: No graph was actually loaded."
    for graph in labelled_graphs:
        # check that each node has a 'comment' attribute
        for node, data in graph.nodes(data=True):
            assert 'comment' in data.keys(), (
                f"🚩 PANIC: Node {node} does not have a 'comment' attribute. "
                f"data keys: {data.keys()}"
            )

    start_total_embedding = datetime.now()

    all_samples_and_labels: list[SamplesAndLabels] = []
    for labelled_graph in labelled_graphs:

        start_embedding = datetime.now()

        # Generate Node2Vec embeddings
        embeddings: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = generate_node_embedding(
            params,
            labelled_graph,
            hyperparams,
        )
        print(f" ▶ [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}] embeddings len: {len(embeddings)}, features: {embeddings[0].shape}")
        
        # embeddings to numpy array
        samples = np.vstack(embeddings) # (2D array of float32)

        # labels from graph nodes 
        labels_in_list = [labelled_graph.nodes[node]['label'] for node in labelled_graph.nodes]
        labels = np.array(labels_in_list, dtype=np.int32) # (1D array of int32)
        all_samples_and_labels.append(
            SamplesAndLabels(samples, labels)
        )

        end_embedding = datetime.now()
        duration_embedding = end_embedding - start_embedding
        duration_embedding_human_readable = datetime_to_human_readable_str(duration_embedding)
        print("Generating embeddings took: {0}".format(duration_embedding_human_readable))
    
    end_total_embedding = datetime.now()
    duration_total_embedding = end_total_embedding - start_total_embedding
    duration_total_embedding_human_readable = datetime_to_human_readable_str(duration_total_embedding)
    print("Generating ALL embeddings took: {0}".format(duration_total_embedding_human_readable))
    results_writer.set_result("duration_embedding", duration_total_embedding_human_readable)

    # concat all samples
    samples = np.vstack([samples_and_labels.samples for samples_and_labels in all_samples_and_labels])

    # get column names
    column_names = []
    if hyperparams.node_embedding.is_using_node2vec():
        assert isinstance(hyperparams, Node2VecHyperparams), (
            f"ERROR: Expected hyperparams to be of type Node2VecHyperparams, but got {type(hyperparams)}"
        )
        column_names.extend(
            [f"node2vec_{i}" for i in range(hyperparams.node2vec_dimensions)]
        )
    if hyperparams.node_embedding.is_using_custom_comment_embedding():
        column_names.extend(
            embeddings_fields
        )
    assert len(column_names) == samples.shape[1], (
        f"ERROR: Expected column_names to have same length as samples.shape[1], but got {len(column_names)} != {samples.shape[1]}"
    )
    

    # Convert scaled_samples back to DataFrame
    scaled_samples_df = pd.DataFrame(
        samples,
        columns=column_names,
    )

    # perform feature evaluation
    print(" 🔘 Performing feature evaluations...")
    _evaluate_features(
        params,
        hyperparams,
        scaled_samples_df,
        FeatureEvaluationSubpipelineNames.PearsonCorrelationPipeline,
        deepcopy(results_writer),
    )
    _evaluate_features(
        params,
        hyperparams,
        scaled_samples_df,
        FeatureEvaluationSubpipelineNames.KendallCorrelationPipeline,
        deepcopy(results_writer),
    )
    _evaluate_features(
        params,
        hyperparams,
        scaled_samples_df,
        FeatureEvaluationSubpipelineNames.SpearmanCorrelationPipeline,
        deepcopy(results_writer),
    )

def _evaluate_features(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    scaled_samples_df, 
    subpipeline: FeatureEvaluationSubpipelineNames,
    results_writer: BaseResultWriter,
):
    
    start_time_train_test = datetime.now()

    # Get correlation algorithm
    correlation_algorithm = None
    if subpipeline == FeatureEvaluationSubpipelineNames.PearsonCorrelationPipeline:
        correlation_algorithm = "pearson"
    elif subpipeline == FeatureEvaluationSubpipelineNames.KendallCorrelationPipeline:
        correlation_algorithm = "kendall"
    elif subpipeline == FeatureEvaluationSubpipelineNames.SpearmanCorrelationPipeline:
        correlation_algorithm = "spearman"
    else:
        raise ValueError(f"Unknown Feature evaluation subpipeline: {subpipeline}")
    assert correlation_algorithm is not None

    results_writer.set_result(
        "subpipeline_name",
        subpipeline.value,
    )

    # Save the hyperparams to the results writer
    add_hyperparams_to_result_writer(
        params,
        hyperparams,
        results_writer,
    )

    # Calculate correlation matrix
    corr_matrix = scaled_samples_df.corr(correlation_algorithm)

    # Print and visualize the correlation matrix
    print(f"Correlation matrix (algorithm: {correlation_algorithm}): \n{corr_matrix}")

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True, cmap='coolwarm')
    plt.title(f"Feature Correlation Matrix (algorithm: {correlation_algorithm})")

    matrix_id = datetime.now().strftime("%Y%m%d%H%M%S")
    results_writer.set_result(
        "matrix_id",
        matrix_id,
    )
    corr_matrix_save_path = __determine_feature_corr_matrix_save_file_path(
        matrix_id
    )

    plt.savefig(corr_matrix_save_path)
    plt.close()

    # Keep best columns based on correlation sum
    corr_sums = corr_matrix.abs().sum()
    sorted_corr_sums = corr_sums.sort_values(ascending=False)

    feature_column_names_sorted = (
        f"descending_best_column_names (algorithm: {correlation_algorithm}): {sorted_corr_sums.index.tolist()}"
        #" ".join(sorted_corr_sums.index.tolist())
    )
    feature_column_values_sorted = (
        f"descending_best_column_values (algorithm: {correlation_algorithm}): {sorted_corr_sums.values.tolist()}"
        #" ".join(map(str, sorted_corr_sums.values.tolist()))
    )

    params.RESULTS_LOGGER.info(
        feature_column_names_sorted
    )
    params.RESULTS_LOGGER.info(
        feature_column_values_sorted
    )
    results_writer.set_result(
        "feature_column_names_sorted",
        feature_column_names_sorted,
    )
    results_writer.set_result(
        "feature_column_values_sorted",
        feature_column_values_sorted,
    )

    # conclude pipeline
    common_pipeline_end(
        params,
        subpipeline,
        start_time_train_test,
        results_writer,
        save_file_format=SaveFileFormat.FEATURE_CSV,
    )
