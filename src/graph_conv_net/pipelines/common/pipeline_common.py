from datetime import datetime
from graph_conv_net.data_loading.data_loading import dev_load_training_graphs
from graph_conv_net.embedding.node_to_vec_enums import NodeEmbeddingType
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.hyperparams import BaseHyperparams, add_hyperparams_to_result_writer
from graph_conv_net.results.base_result_writer import BaseResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str

def common_load_labelled_graph(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    results_writer: BaseResultWriter,
):
    """
    Load labelled graph from given path.
    """
    print("Annotated graph from: {0}".format(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH))

    start = datetime.now()
    
    labelled_graphs = dev_load_training_graphs(
        params,
        hyperparams,
        hyperparams.input_mem2graph_dataset_dir_path,
    )
    
    end = datetime.now()
    duration = end - start
    duration_human_readable = datetime_to_human_readable_str(duration)
    print("Loading data took: {0}".format(duration_human_readable))
    print("type(labelled_graphs): {0}".format(type(labelled_graphs)))
    print("type of a labelled_graphs element: {0}".format(type(labelled_graphs[0])))
    print("len(labelled_graphs): {0}".format(len(labelled_graphs)))

    results_writer.set_result(
        "nb_input_graphs",
        str(len(labelled_graphs)),
    )
    
    # filter out None values
    labelled_graphs = [graph for graph in labelled_graphs if graph is not None]

    # print a graph to see what it looks like
    #t_graph = labelled_graphs[0]
    #print("t_graph.nodes.data(): {0}".format(t_graph.nodes.data()))
    return labelled_graphs

def common_init_result_writer_additional_results(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    results_writer: BaseResultWriter,
):
    """
    Common initialization of the result writer,
    with some additional results not specified
    as hyperparameters.
    """
    add_hyperparams_to_result_writer(
        params,
        hyperparams,
        results_writer,
    )