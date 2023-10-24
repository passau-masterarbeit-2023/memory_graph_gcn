from copy import deepcopy
import json
import os
from sklearn.preprocessing import StandardScaler
from graph_conv_net.embedding.node_to_vec import generate_node_embedding
from graph_conv_net.embedding.node_to_vec_enums import check_embedding_feature_len_consistency, get_feature_names_from_comment, get_graph_comment
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
    coorelation_algorithm: str,
) -> str:
    """
    Determine the path of the save file.
    """
    save_dir_path = os.environ.get("RESULTS_LOGGER_DIR_PATH")
    corr_matrix_dir_path = f"{save_dir_path}/feature_corr"

    # check if directory exists, else create it
    if not os.path.exists(corr_matrix_dir_path):
        os.makedirs(corr_matrix_dir_path)

    save_file_path = f"{corr_matrix_dir_path}/feature_{coorelation_algorithm}_correlation_matrix_{hash_key}.png"
    return save_file_path



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
    custom_comment_embedding_len = check_embedding_feature_len_consistency(
        hyperparams.input_mem2graph_dataset_dir_path
    )

    # get graph comment (for custom comment feature names)
    embeddings_fields = get_feature_names_from_comment(
        hyperparams.input_mem2graph_dataset_dir_path
    )

    assert len(labelled_graphs) > 0, "ERROR: No graph was actually loaded."

    start_total_embedding = datetime.now()

    all_samples_and_labels: list[SamplesAndLabels] = []
    for labelled_graph in labelled_graphs:

        start_embedding = datetime.now()

        # Generate Node2Vec embeddings
        embeddings: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = generate_node_embedding(
            params,
            labelled_graph,
            hyperparams,
            custom_comment_embedding_len,
        )
        print(f" ▶ [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}] embeddings len: {len(embeddings)}, features: {embeddings[0].shape}")
        
        # embeddings to numpy array
        samples = np.vstack(embeddings) # (2D array of float32)

        # labels from graph nodes 
        labels_in_list = [labelled_graph.graph.nodes[node]['label'] for node in labelled_graph.graph.nodes]
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

    # # WARN: Some embeddings (the statistical embedding) is not always consistent in feature length
    # # In that case, some 0-padding is necessary on the samples
    # max_feature_length = max([samples_and_labels.samples.shape[1] for samples_and_labels in all_samples_and_labels])

    # all_padded_samples = []
    # for i in range(len(all_samples_and_labels)):
    #     samples_and_labels = all_samples_and_labels[i]
    #     sample_shape = samples_and_labels.samples.shape
    #     if sample_shape[1] != max_feature_length:
    #         # pad with zeros to have a consistent number of features
    #         padding_shape = (sample_shape[0], max_feature_length - sample_shape[1])
    #         padded_sample = np.hstack([samples_and_labels.samples, np.zeros(padding_shape)])
    #         all_padded_samples.append(padded_sample)

    #     else:
    #         all_padded_samples.append(samples_and_labels.samples)
    # assert len(all_padded_samples) == len(all_samples_and_labels)
    # del all_samples_and_labels

    # # check that all samples have same number of features
    # assert len(set([padded_samples.shape[1] for padded_samples in all_padded_samples])) == 1, (
    #     "ERROR: Not all samples have the same number of features."
    # )

    nb_features = all_samples_and_labels[0].samples.shape[1]
    for i in range(len(all_samples_and_labels)):
        samples_and_labels = all_samples_and_labels[i]
        if samples_and_labels.samples.shape[1] != nb_features:
            raise ValueError(
                f"Error: Samples have different number of features: {samples_and_labels.samples.shape[1]} (index: {i}) != {nb_features} (index: 0), "
                f"on dir: {hyperparams.input_mem2graph_dataset_dir_path}"
            )

    # concat all samples
    samples = np.vstack([samples_and_labels.samples for samples_and_labels in all_samples_and_labels])
    #samples = np.vstack([samples for samples in all_padded_samples])

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
        f"ERROR: Expected column_names to have same length as samples.shape[1], but got {len(column_names)} != {samples.shape[1]} "
        f"on dir: {hyperparams.input_mem2graph_dataset_dir_path}. "
        f"column_names: {column_names}, of length: {len(column_names)}"
        f"stacked samples.shape: {samples.shape}"
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

    last_dir_component = os.path.basename(hyperparams.input_mem2graph_dataset_dir_path)

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
    plt.title(f"Feature Correlation Matrix (algorithm: {correlation_algorithm}) on {last_dir_component}")

    corr_matrix_save_path = __determine_feature_corr_matrix_save_file_path(
        last_dir_component, correlation_algorithm
    )
    results_writer.set_result(
        "corr_matrix_img_save_path",
        corr_matrix_save_path,
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
