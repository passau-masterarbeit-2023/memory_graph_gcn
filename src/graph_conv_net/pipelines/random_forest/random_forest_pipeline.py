from dataclasses import dataclass
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from graph_conv_net.data_loading.data_loading import dev_load_training_graphs
from graph_conv_net.embedding.node_to_vec import generate_node2vec_graph_embedding
from graph_conv_net.ml.evaluation import evaluate_metrics
from graph_conv_net.results.base_result_writer import BaseResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str
from graph_conv_net.pipelines.pipelines import PipelineNames, RandomForestPipeline, add_hyperparams_to_result_writer
from graph_conv_net.params.params import ProgramParams
import numpy as np

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
    CURRENT_PIPELINE_NAME = PipelineNames.RandomForestPipeline
    
    add_hyperparams_to_result_writer(
        results_writer,
        hyperparams,
    )

    # load data
    print(" 🔘 Loading data...")
    print("Annotated graph from: {0}".format(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH))

    start = datetime.now()
    
    labelled_graphs = dev_load_training_graphs(
        params,
        hyperparams,
        params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH
    )
    
    end = datetime.now()
    duration = end - start
    duration_human_readable = datetime_to_human_readable_str(duration)
    print("Loading data took: {0}".format(duration_human_readable))
    print("type(labelled_graphs): {0}".format(type(labelled_graphs)))
    print("type of a labelled_graphs element: {0}".format(type(labelled_graphs[0])))
    print("len(labelled_graphs): {0}".format(len(labelled_graphs)))
    
    # filter out None values
    labelled_graphs = [graph for graph in labelled_graphs if graph is not None]

    # print a graph to see what it looks like
    #t_graph = labelled_graphs[0]
    #print("t_graph.nodes.data(): {0}".format(t_graph.nodes.data()))

    # convert graphs to PyTorch Geometric data objects
    start_total_embedding = datetime.now()

    all_samples_and_labels: list[SamplesAndLabels] = []
    for labelled_graph in labelled_graphs:

        start_embedding = datetime.now()

        # Generate Node2Vec embeddings
        embeddings = generate_node2vec_graph_embedding(
            params,
            labelled_graph,
            hyperparams
        )
        print(f"embeddings len: {len(embeddings)} [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}]".format())
        #print("embeddings[0]: {0}".format(embeddings[0]))
        
        # Replace node features with Node2Vec embeddings
        samples = np.array(embeddings) 

        # Prepare edge connectivity (from adjacency matrix or edge list)
        labels = np.ndarray([labelled_graph.nodes[node]['label'] for node in labelled_graph.nodes])
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

    def pad_array(arr, max_length):
        pad_len = max_length - arr.shape[0]
        return np.pad(arr, [(0, pad_len), (0, 0)], mode='constant')

    X_train = np.array([pad_array(samples_and_labels.samples, max_length) for samples_and_labels in train_data])
    y_train = np.concatenate([samples_and_labels.labels for samples_and_labels in train_data])

    X_test = np.array([pad_array(samples_and_labels.samples, max_length) for samples_and_labels in test_data])
    y_test = np.concatenate([samples_and_labels.labels for samples_and_labels in test_data])

    # Initialize Random Forest model
    clf = RandomForestClassifier(
        n_estimators=hyperparams.random_forest_n_estimators, 
        random_state=params.RANDOM_SEED,
        n_jobs=hyperparams.random_forest_n_jobs,
    )

    # Training
    print(" 🔘 Training...")
    clf.fit(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Evaluation metrics
    print(" 🔘 Evaluating...")
    metrics = evaluate_metrics(
        y_test, 
        y_pred,
        results_writer,
        params,
    )
    return metrics
