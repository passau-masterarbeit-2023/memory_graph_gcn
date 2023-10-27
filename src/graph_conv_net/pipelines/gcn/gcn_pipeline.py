from copy import deepcopy
from datetime import datetime
import torch_geometric.data
from graph_conv_net.ml.advanced_gcn import AdvancedGCN
from graph_conv_net.ml.gcn_with_dropout import ImprovedGNN
from graph_conv_net.ml.simple_gcn import LessSimplifiedGNN
from graph_conv_net.ml.very_simple_gcn import VerySimplifiedGNN
from graph_conv_net.pipelines.pipelines import GCNSubpipelineNames
import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.ml.first_model import GNN
from graph_conv_net.ml.evaluation import evaluate_metrics
from graph_conv_net.embedding.node_to_vec import generate_node_embedding
from graph_conv_net.pipelines.common.pipeline_common import common_load_labelled_graph, common_pipeline_end
from graph_conv_net.pipelines.hyperparams import FirstGCNPipelineHyperparams, add_hyperparams_to_result_writer
from graph_conv_net.results.base_result_writer import BaseResultWriter, SaveFileFormat
from graph_conv_net.utils.debugging import dp
from graph_conv_net.utils.utils import datetime_to_human_readable_str, str2enum

def gcn_pipeline(
    params: ProgramParams,
    hyperparams: FirstGCNPipelineHyperparams,
    results_writer: BaseResultWriter,
):
    """
    A first pipeline to test the GCN model.
    """

    # load data
    print(" 🔘 Loading data...")
    labelled_graphs = common_load_labelled_graph(
        params,
        hyperparams,
        results_writer,
    )
    custom_comment_embedding_len = len(labelled_graphs[0].custom_embedding_fields)
    
    # convert graphs to PyTorch Geometric data objects
    start_total_embedding = datetime.now()
    
    print(" 🔘 Generating embeddings...")
    data_from_graphs = []
    length_of_labelled_graphs = len(labelled_graphs)
    for i in range(length_of_labelled_graphs):
        labelled_graph = labelled_graphs[i]
        dp(
            "Graph contains: nb nodes: {0}".format(len(labelled_graph.graph.nodes)), 
            "nb edges: {0}".format(len(labelled_graph.graph.edges))
        )

        start_embedding = datetime.now()
        
        # Generate Node2Vec embeddings
        embeddings = generate_node_embedding(
            params,
            labelled_graph,
            hyperparams,
            custom_comment_embedding_len,
        )
        print(
            f" ▶ [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}]",
            f"[graph: {i}/{length_of_labelled_graphs}]]",
            f"embeddings len: {len(embeddings)}, features: {embeddings[0].shape}",
        )
        
        # Node features using custom Node2Vec based embeddings
        node_feature_matrix = torch.tensor(np.vstack(embeddings), dtype=torch.float)

        # Edge features using edge weights
        labelled_graph = nx.convert_node_labels_to_integers(labelled_graph.graph)
        edge_features = []
        for u, v, edge_data in list(labelled_graph.edges(data=True)):
            edge_features.append(edge_data['weight'])
        edge_feature_matrix = torch.tensor(edge_features, dtype=torch.float)

        # Prepare edge connectivity (from adjacency matrix or edge list)
        edges = list(labelled_graph.edges)
        dp("type(edges): {0}".format(type(edges)))
        dp("len(edges): {0}".format(len(edges)))
        dp("type(edges[0]): {0}".format(type(edges[0])))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graph_connectivity = edge_index.view(2, -1)
        
        node_labels = torch.tensor([labelled_graph.nodes[node]['label'] for node in labelled_graph.nodes()], dtype=torch.float).unsqueeze(1) 
        
        data = torch_geometric.data.Data(
            x=node_feature_matrix,
            edge_index=graph_connectivity,
            edge_attr=edge_feature_matrix,
            y=node_labels,
        )
        data.validate(raise_on_error=True)
        data_from_graphs.append(data)

        end_embedding = datetime.now()
        duration_embedding = end_embedding - start_embedding
        duration_embedding_human_readable = datetime_to_human_readable_str(duration_embedding)
        print(
            f" ▶ [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}]",
            f"[graph: {i}/{length_of_labelled_graphs}]]. ",
            f"Embeddings loop took: {0}".format(duration_embedding_human_readable),
        )
    
    end_total_embedding = datetime.now()
    duration_total_embedding = end_total_embedding - start_total_embedding
    duration_total_embedding_human_readable = datetime_to_human_readable_str(duration_total_embedding)
    print("Generating ALL embeddings took: {0}".format(duration_total_embedding_human_readable))
    results_writer.set_result("duration_embedding", duration_total_embedding_human_readable)

    # split data into train and test sets
    print(" 🔘 Splitting data into train and test sets...")
    PERCENTAGE_OF_DATA_FOR_TRAINING = 0.7
    train_data = data_from_graphs[:int(len(data_from_graphs) * PERCENTAGE_OF_DATA_FOR_TRAINING)]
    test_data = data_from_graphs[int(len(data_from_graphs) * PERCENTAGE_OF_DATA_FOR_TRAINING):]
    assert len(train_data) > 0
    assert len(test_data) > 0
    print("len(train_data): {0}".format(len(train_data)))
    print("len(test_data): {0}".format(len(test_data)))

    # Define and initialize GCN model
    num_features = train_data[0].num_node_features
    num_classes = 2 # label 0 or 1
    print("num_features: {0}".format(num_features))
    print("num_classes: {0}".format(num_classes))

    results_writer.set_result(
        "nb_node_features",
        str(num_features),
    )
    
    # train and evaluate models
    print(" 🔘 Training and evaluating models...")

    train_and_eval_gcn(
        params = params,
        hyperparams = hyperparams,
        results_writer = deepcopy(results_writer),
        model = VerySimplifiedGNN(num_features, num_classes),
        train_data = deepcopy(train_data),
        test_data = deepcopy(test_data),
    )
    train_and_eval_gcn(
        params = params,
        hyperparams = hyperparams,
        results_writer = deepcopy(results_writer),
        model = LessSimplifiedGNN(num_features, num_classes),
        train_data = deepcopy(train_data),
        test_data = deepcopy(test_data),
    )
    train_and_eval_gcn(
        params = params,
        hyperparams = hyperparams,
        results_writer = deepcopy(results_writer),
        model = GNN(num_features, num_classes),
        train_data = deepcopy(train_data),
        test_data = deepcopy(test_data),
    )
    train_and_eval_gcn(
        params = params,
        hyperparams = hyperparams,
        results_writer = deepcopy(results_writer),
        model = ImprovedGNN(num_features, num_classes),
        train_data = deepcopy(train_data),
        test_data = deepcopy(test_data),
    )
    train_and_eval_gcn(
        params = params,
        hyperparams = hyperparams,
        results_writer = deepcopy(results_writer),
        model = AdvancedGCN(num_features, num_classes),
        train_data = deepcopy(train_data),
        test_data = deepcopy(test_data),
    )
    
    

def train_and_eval_gcn(
        params: ProgramParams,
        hyperparams: FirstGCNPipelineHyperparams,
        results_writer: BaseResultWriter,
        model: torch.nn.Module,
        train_data: list[torch_geometric.data.Data],
        test_data: list[torch_geometric.data.Data],
):
    """
    Train and evaluate the GCN model.
    """

    start_time_train_test = datetime.now()

    # Set the name of the subpipeline
    subpipeline = None
    if type(model) == VerySimplifiedGNN:
        subpipeline = GCNSubpipelineNames.VerySimpleGCNPipeline
    elif type(model) == LessSimplifiedGNN:
        subpipeline = GCNSubpipelineNames.SimpleGCNPipeline
    elif type(model) == GNN:
        subpipeline = GCNSubpipelineNames.FirstGCNPipeline
    elif type(model) == ImprovedGNN:
        subpipeline = GCNSubpipelineNames.GCNWithDropoutPipeline
    elif type(model) == AdvancedGCN:
        subpipeline = GCNSubpipelineNames.AdvancedGCNPipeline
    else:
        raise Exception("Unknown model type: {0}".format(type(model)))

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

    # Define loss function and optimizer
    pos_weight = torch.tensor([1.0, 50.0])  # Adjust the weight for the positive class
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # Training loop
    print(" 🔘 Training...")
    model.train()
    epochs = hyperparams.first_gcn_training_epochs  # Replace with a sensible number of epochs

    for _ in range(epochs):
        for data in train_data:
            optimizer.zero_grad()
            output = model(data)
            target = F.one_hot(data.y.view(-1).long(), num_classes=2).float()
            loss = criterion(output, target)    
            #loss = criterion(output, data.y.view(-1, 2).float())  # Assuming data.y is [batch_size, 2] and has labels for each class
            loss.backward()
            optimizer.step()
    
    # Evaluation of trained model
    print(" 🔘 Evaluating...")
    model.eval()
    
    # Lists to store true labels and predictions for all test graphs
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for data in test_data:
            output = model(data)
            predicted = output.argmax(dim=1)
            
            all_true_labels.extend(data.y.view(-1).long().tolist())
            all_pred_labels.extend(predicted.tolist())

    # Convert lists to numpy arrays for evaluation
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    # Compute the metrics
    _ = evaluate_metrics(
        all_true_labels, 
        all_pred_labels,
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