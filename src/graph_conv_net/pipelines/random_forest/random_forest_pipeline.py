from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch_geometric.data
from graph_conv_net.data_loading.data_loading import dev_load_training_graphs
from graph_conv_net.embedding.node_to_vec import generate_node2vec_graph_embedding
from graph_conv_net.ml.evaluation import evaluate_metrics
from graph_conv_net.results.base_result_writer import BaseResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str
from torch_geometric.utils import from_networkx, convert
from graph_conv_net.pipelines.pipelines import PipelineNames, RandomForestPipeline, add_hyperparams_to_result_writer
from graph_conv_net.params.params import ProgramParams
import torch
import numpy as np

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
    data_from_graphs: list[torch_geometric.data.Data] = []
    for labelled_graph in labelled_graphs:

        start_embedding = datetime.now()
        
        # Generate Node2Vec embeddings
        embeddings = generate_node2vec_graph_embedding(
            params,
            labelled_graph,
            hyperparams
        )
        print("embeddings len: {0}".format(len(embeddings)))
        print("embeddings[0]: {0}".format(embeddings[0]))
        
        # Convert the graph to a PyTorch Geometric data object
        data: torch_geometric.data.Data = from_networkx(labelled_graph)

        # Replace node features with Node2Vec embeddings
        data.x = torch.tensor(np.array(embeddings), dtype=torch.float) # type: ignore

        # Prepare edge connectivity (from adjacency matrix or edge list)
        data.edge_index = convert.from_networkx(labelled_graph).edge_index # type: ignore
        data.y = torch.tensor([labelled_graph.nodes[node]['label'] for node in labelled_graph.nodes()], dtype=torch.float).unsqueeze(1) # type: ignore
        data_from_graphs.append(data)

        end_embedding = datetime.now()
        duration_embedding = end_embedding - start_embedding
        duration_embedding_human_readable = datetime_to_human_readable_str(duration_embedding)
        print("Generating embeddings took: {0}".format(duration_embedding_human_readable))
    
    # split data into train and test sets
    print(" 🔘 Splitting data into train and test sets...")
    PERCENTAGE_OF_DATA_FOR_TRAINING = 0.7
    train_data = data_from_graphs[:int(len(data_from_graphs) * PERCENTAGE_OF_DATA_FOR_TRAINING)]
    test_data = data_from_graphs[int(len(data_from_graphs) * PERCENTAGE_OF_DATA_FOR_TRAINING):]
    assert len(train_data) > 0
    assert len(test_data) > 0
    print("len(train_data): {0}".format(len(train_data)))
    print("len(test_data): {0}".format(len(test_data)))

    labelled_graphs = dev_load_training_graphs(
        params,
        params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH
    )

    # Create arrays for training and testing
    X_train = [data.x.numpy() for data in train_data]
    y_train = [data.y.numpy() for data in train_data]
    X_test = [data.x.numpy() for data in test_data]
    y_test = [data.y.numpy() for data in test_data]

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
