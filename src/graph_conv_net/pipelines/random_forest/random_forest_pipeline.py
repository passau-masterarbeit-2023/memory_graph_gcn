from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from graph_conv_net.pipelines.hyperparams import RandomForestPipeline, add_hyperparams_to_result_writer
from graph_conv_net.pipelines.pipelines import ClassicMLSubpipelineNames
import numpy as np

from graph_conv_net.embedding.node_to_vec import generate_node_embedding
from graph_conv_net.ml.evaluation import evaluate_metrics
from graph_conv_net.pipelines.common.pipeline_common import common_embedding_loop_end, common_load_labelled_graph, common_pipeline_end
from graph_conv_net.results.base_result_writer import BaseResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str
from graph_conv_net.params.params import ProgramParams

@dataclass(init=True, repr=True, eq=True, unsafe_hash=True, frozen=True)
class SamplesAndLabels:
    samples: np.ndarray
    labels: np.ndarray

def random_forest_pipeline(
    params: ProgramParams,
    hyperparams: RandomForestPipeline,
    results_writer: BaseResultWriter,
):
    """
    A pipeline to test the Random Forest model.
    """

    # load data
    print(" 🔘 Loading data...")
    labelled_graphs = common_load_labelled_graph(
        params,
        hyperparams,
        results_writer,
    )
    custom_comment_embedding_len = len(labelled_graphs[0].custom_embedding_fields)
    
    # perform embedding of graph nodes
    start_total_embedding = datetime.now()

    all_samples_and_labels: list[SamplesAndLabels] = []
    length_of_labelled_graphs = len(labelled_graphs)
    for i in range(length_of_labelled_graphs):
        labelled_graph = labelled_graphs[i]

        start_embedding = datetime.now()

        # Generate Node2Vec embeddings
        embeddings: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = generate_node_embedding(
            params,
            labelled_graph,
            hyperparams,
            custom_comment_embedding_len
        )

        # Node2Vec embeddings to numpy array
        samples = np.vstack(embeddings) # (2D array of float32)

        # labels from graph nodes 
        labels_in_list = [labelled_graph.graph.nodes[node]['label'] for node in labelled_graph.graph.nodes]
        labels = np.array(labels_in_list, dtype=np.int32) # (1D array of int32)

        all_samples_and_labels.append(
            SamplesAndLabels(samples, labels)
        )

        common_embedding_loop_end(
            i,
            params,
            hyperparams,
            length_of_labelled_graphs,
            start_embedding,
            len(embeddings),
            embeddings[0].shape[0],
        )
    
    end_total_embedding = datetime.now()
    duration_total_embedding = end_total_embedding - start_total_embedding
    duration_total_embedding_human_readable = datetime_to_human_readable_str(duration_total_embedding)
    print("Generating ALL embeddings took: {0}".format(duration_total_embedding_human_readable))
    results_writer.set_result("duration_embedding", duration_total_embedding_human_readable)

    # split data into train and test sets
    print(" 🔘 Splitting data into train and test sets...")
    PERCENTAGE_OF_DATA_FOR_TRAINING = 0.8
    separator_index = int(len(all_samples_and_labels) * PERCENTAGE_OF_DATA_FOR_TRAINING)
    train_data = all_samples_and_labels[:separator_index]
    test_data = all_samples_and_labels[separator_index:]
    assert len(train_data) > 0
    assert len(test_data) > 0
    print("len(train_data): {0}".format(len(train_data)))
    print("len(test_data): {0}".format(len(test_data)))

    # Create arrays for training and testing
    max_length = max([samples_and_labels.samples.shape[0] for samples_and_labels in all_samples_and_labels])
    max_label_length = max([len(samples_and_labels.labels) for samples_and_labels in all_samples_and_labels])
    print("max_length for samples: {0}".format(max_length))
    print("max_label_length for labels: {0}".format(max_label_length))
    assert max_length == max_label_length, (
        "ERROR: Label and sample max lengths should be equal, "
        f"but sample max length is {max_length} and label max length is {max_label_length}. "
    )

    # we need padding, since Random Forest requires all samples to have the same length
    # This means that each 2D sample array will be padded with zeros to match the max length
    def pad_array(arr, max_length):
        pad_len = max_length - arr.shape[0]
        return np.pad(arr, [(0, pad_len), (0, 0)], mode='constant')

    def pad_labels(arr, max_label_length):
        pad_len = max_label_length - len(arr)
        return np.pad(arr, (0, pad_len), mode='constant', constant_values=0)

    # data for training
    padded_arrays = [pad_array(samples_and_labels.samples, max_length) for samples_and_labels in train_data]
    print("len(padded_arrays): {0}".format(len(padded_arrays)))
    print("padded_arrays[0] type: {0}".format(type(padded_arrays[0])))
    print("padded_arrays[0].shape: {0}".format(padded_arrays[0].shape))
    X_train = np.vstack(padded_arrays)

    y_trained_in_list = [pad_labels(samples_and_labels.labels, max_label_length) for samples_and_labels in train_data]
    print("len(y_trained_in_list): {0}".format(len(y_trained_in_list)))
    print("y_trained_in_list[0] type: {0}".format(type(y_trained_in_list[0])))
    print("y_trained_in_list[0].shape: {0}".format(y_trained_in_list[0].shape))
    y_train = np.concatenate(y_trained_in_list)

    X_test = np.vstack([pad_array(samples_and_labels.samples, max_length) for samples_and_labels in test_data])
    y_test = np.concatenate([pad_labels(samples_and_labels.labels, max_label_length) for samples_and_labels in test_data])

    print("X_train.shape: {0}".format(X_train.shape))
    print("y_train.shape: {0}".format(y_train.shape))
    print("X_test.shape: {0}".format(X_test.shape))
    print("y_test.shape: {0}".format(y_test.shape))

    # train and test classical ML models
    print(" 🔘 Training and testing classical ML models...")
    train_and_eval_classical_ml(
        params,
        hyperparams,
        deepcopy(results_writer),
        RandomForestClassifier(
            n_estimators=hyperparams.random_forest_n_estimators, 
            random_state=params.RANDOM_SEED,
            n_jobs=hyperparams.random_forest_n_jobs,
        ),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    train_and_eval_classical_ml(
        params,
        hyperparams,
        deepcopy(results_writer),
        SGDClassifier(random_state=42, n_jobs = params.MAX_ML_WORKERS),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    train_and_eval_classical_ml(
        params,
        hyperparams,
        deepcopy(results_writer),
        LogisticRegression(n_jobs = params.MAX_ML_WORKERS),
        X_train,
        y_train,
        X_test,
        y_test,
    )

def train_and_eval_classical_ml(
    params: ProgramParams,
    hyperparams: RandomForestPipeline,
    results_writer: BaseResultWriter,
    clf: RandomForestClassifier | SGDClassifier | LogisticRegression,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Train and evaluate a classical ML model.
    """

    start_time_train_test = datetime.now()

    # subpipeline
    subpipeline = None
    if type(clf) == RandomForestClassifier:
        subpipeline = ClassicMLSubpipelineNames.RandomForestPipeline
    elif type(clf) == SGDClassifier:
        subpipeline = ClassicMLSubpipelineNames.SGDClassifierPipeline
    elif type(clf) == LogisticRegression:
        subpipeline = ClassicMLSubpipelineNames.LogisticRegressionPipeline
    else:
        raise ValueError("ERROR: Unknown subpipeline: {0}".format(subpipeline))
    
    assert subpipeline is not None
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

    # Training
    try:
        print(" 🔘 Training...")
        clf.fit(X_train, y_train)

        # Prediction
        y_pred = clf.predict(X_test)

        # Evaluation metrics
        print(" 🔘 Evaluating...")
        _ = evaluate_metrics(
            y_test, 
            y_pred,
            results_writer,
            params,
        )
        
        # conclude pipeline
        common_pipeline_end(
            params,
            subpipeline,
            start_time_train_test,
            results_writer,
        )
    except Exception as e:
        print(
            f"ERROR: Exception occurred during train and eval step: {e} "    
            f"for subpipeline: {subpipeline} "
            f"for dir path: {hyperparams.input_mem2graph_dataset_dir_path}"
        )
        raise e
