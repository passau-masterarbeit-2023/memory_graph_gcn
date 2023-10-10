import time
from torch_geometric.utils import from_networkx, convert
import torch
import networkx as nx

import os
import pickle

from graph_conv_net.data_loading.data_loading import dev_load_training_graphs, load_annotated_graph
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.ml.first_model import GNN


def main(params: ProgramParams):
    
    # load data
    print("Loading data...")
    print("Annotated graph from: {0}".format(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH))

    start = time.time()
    datas = dev_load_training_graphs(
        params,
        params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH
    )
    end = time.time()
    print("Loading data took: {0} seconds".format(end - start))
    print("type(datas): {0}".format(type(datas)))
    print("len(datas): {0}".format(len(datas)))

    # TODO: start working from here...

    # # Convert to PyTorch Geometric data object
    # print("Converting to PyTorch Geometric data object...")
    # start = time.time()
    
    # data = from_networkx(nx_graph)

    # end = time.time()
    # print("Converting to PyTorch Geometric data object took: {0} seconds".format(end - start))
    # print("type(data): {0}".format(type(data)))

    # # training model
    # print("Preparing node features, edge connectivity and labels...")
    # start = time.time()
    # # Prepare node features (identity matrix as placeholder) (you may need to adjust features, edges, etc.)
    # data.x = torch.eye(nx_graph.number_of_nodes(), dtype=torch.float)  
    # # Prepare edge connectivity (from adjacency matrix or edge list)
    # #A = nx.to_scipy_sparse_array(nx_graph)  # Adjacency matrix
    # data.edge_index = convert.from_networkx(nx_graph).edge_index
    # data.y = torch.tensor([nx_graph.nodes[node]['label'] for node in nx_graph.nodes()], dtype=torch.float).unsqueeze(1)
    # end = time.time()
    # print("Preparing node features, edge connectivity and labels took: {0} seconds".format(end - start))

    # # Model, optimizer, and loss
    # model = GNN(data)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # loss_func = torch.nn.BCELoss()

    # # Training loop
    # print("Training loop...")
    # start = time.time()
    # for epoch in range(1000):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = loss_func(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()

    #     # Print progress
    #     if epoch % 100 == 0:
    #         print(f'Epoch: {epoch}, Loss: {loss.item()}')
    # end = time.time()
    # print("Training loop took: {0} seconds".format(end - start))

    # #### Evaluation
    # # load annotated graph, and evaluate the model on it.
    # print("### Evaluating model ###")

    # # load annotated graph
    # start = time.time()
    # eval_nx_graph = load_annotated_graph(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH + "/Training_27572-1644383600-heap.raw_dot.gv")
    # end = time.time()
    # print("Loading annotated graph took: {0} seconds".format(end - start))

    # # Set labels for the evaluation graph
    # print("Setting labels for the evaluation graph...")
    # start = time.time()
    # for node in eval_nx_graph.nodes():
    #     eval_nx_graph.nodes[node]['label'] = 1 if 'KN_KEY' in node else 0
    # end = time.time()
    # print("Setting labels for the evaluation graph took: {0} seconds".format(end - start))

    # # Convert to PyTorch Geometric data object
    # print("Converting to PyTorch Geometric data object...")
    # start = time.time()
    # eval_data = from_networkx(eval_nx_graph)
    # eval_data.x = torch.eye(eval_nx_graph.number_of_nodes(), dtype=torch.float)
    # A_eval = nx.to_scipy_sparse_matrix(eval_nx_graph)
    # eval_data.edge_index = convert.from_scipy_sparse_matrix(A_eval)
    # eval_data.y = torch.tensor([eval_nx_graph.nodes[node]['label'] for node in eval_nx_graph.nodes()], dtype=torch.float).unsqueeze(1)
    # end = time.time()
    # print("Converting to PyTorch Geometric data object took: {0} seconds".format(end - start))

    # # Evaluate the model (assuming binary classification)
    # print("Evaluating the model...")
    # start = time.time()
    # model.eval()
    # with torch.no_grad():
    #     eval_out = model(eval_data)
    #     predictions = (eval_out > 0.5).float()
    #     correct = (predictions == eval_data.y).sum().item()
    #     total = eval_data.y.size(0)
    #     accuracy = correct / total
    #     print(f'Evaluation Accuracy: {accuracy * 100:.2f}%')
    # end = time.time()
    # print("Evaluating the model took: {0} seconds".format(end - start))


if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)