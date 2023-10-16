import time
import torch_geometric.data
from torch_geometric.utils import from_networkx, convert
from graph_conv_net.embedding.node_to_vec import generate_node2vec_graph_embedding
from graph_conv_net.ml.evaluation import evaluate_metrics
import torch
import numpy as np

from graph_conv_net.data_loading.data_loading import dev_load_training_graphs, load_annotated_graph
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.ml.first_model import GNN


def main(params: ProgramParams):
    
    # load data
    print(" |> Loading data...")
    print("Annotated graph from: {0}".format(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH))

    start = time.time()
    labelled_graphs = dev_load_training_graphs(
        params,
        params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH
    )
    end = time.time()
    print("Loading data took: {0} seconds".format(end - start))
    print("type(datas): {0}".format(type(labelled_graphs)))
    print("type of a data element: {0}".format(type(labelled_graphs[0])))
    print("len(datas): {0}".format(len(labelled_graphs)))
    
    # filter out None values
    labelled_graphs = [graph for graph in labelled_graphs if graph is not None]

    # print a graph to see what it looks like
    t_graph = labelled_graphs[0]
    print("t_graph.nodes.data(): {0}".format(t_graph.nodes.data()))

    # convert graphs to PyTorch Geometric data objects
    data_from_graphs = []
    for labelled_graph in labelled_graphs:
        # TODO: move out
        # Make sure all edge weights are numerical
        # for u, v, data in labelled_graph.edges(data=True):
        #     data['weight'] = float(data.get('weight', 1.0))

        # Generate Node2Vec embeddings
        embeddings = generate_node2vec_graph_embedding(
            params,
            labelled_graph
        )
        
        # Convert the graph to a PyTorch Geometric data object
        data: torch_geometric.data.Data = from_networkx(labelled_graph)

        # Replace node features with Node2Vec embeddings
        data.x = torch.tensor(embeddings, dtype=torch.float)

        # Prepare node features (identity matrix as placeholder) (you may need to adjust features, edges, etc.)
        #data.x = torch.eye(labelled_graph.number_of_nodes(), dtype=torch.float)  
        
        # Prepare edge connectivity (from adjacency matrix or edge list)
        data.edge_index = convert.from_networkx(labelled_graph).edge_index
        data.y = torch.tensor([labelled_graph.nodes[node]['label'] for node in labelled_graph.nodes()], dtype=torch.float).unsqueeze(1)
        data_from_graphs.append(data)
    
    # split data into train, validation and test sets
    train_data = data_from_graphs[:int(len(data_from_graphs) * 0.8)]
    test_data = data_from_graphs[int(len(data_from_graphs) * 0.8):]
    assert len(train_data) > 0
    assert len(test_data) > 0

    # CONTINUE HERE. NEED TO TRAIN THE MODEL ON THE TRAINING SET, THEN EVALUATE IT ON THE TEST SET.
    # Define and initialize your GCN model
    num_features = train_data[0].num_node_features
    num_classes = 2 # label 0 or 1
    print("num_features: {0}".format(num_features))
    print("num_classes: {0}".format(num_classes))
    model = GNN(num_features, num_classes)  # Replace with your model class and appropriate input/output sizes
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()

    epochs = 5 # Replace with a sensible number of epochs
    for _ in range(epochs):
        for data in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1).long())
            loss.backward()
            optimizer.step()
    
    # Evaluation of trained model
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
    metrics = evaluate_metrics(
        all_true_labels, all_pred_labels
    )

    # Display the metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")





if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)