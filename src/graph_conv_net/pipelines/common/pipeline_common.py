from datetime import datetime
from enum import Enum
from graph_conv_net.data_loading.data_loading import dev_load_training_graphs
from graph_conv_net.embedding.node_to_vec_enums import NodeEmbeddingType
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.hyperparams import BaseHyperparams, add_hyperparams_to_result_writer
from graph_conv_net.results.base_result_writer import BaseResultWriter, SaveFileFormat
from graph_conv_net.utils.debugging import dp
from graph_conv_net.utils.utils import datetime_to_human_readable_str, str2enum

def common_load_labelled_graph(
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    results_writer: BaseResultWriter,
):
    """
    Load labelled graph from given path.
    """

    start = datetime.now()
    
    labelled_graphs = dev_load_training_graphs(
        params,
        hyperparams.input_mem2graph_dataset_dir_path,
    )
    assert len(labelled_graphs) > 0, "ERROR: No graph was actually loaded."
    
    end = datetime.now()
    duration = end - start
    duration_human_readable = datetime_to_human_readable_str(duration)
    print("Loading data took: {0}".format(duration_human_readable))
    dp("type(labelled_graphs): {0}".format(type(labelled_graphs)))
    dp("type of a labelled_graphs element: {0}".format(type(labelled_graphs[0])))
    dp("len(labelled_graphs): {0}".format(len(labelled_graphs)))

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

def common_pipeline_end(
    params: ProgramParams,
    subpipeline: Enum,
    start_time_train_test: datetime,
    results_writer: BaseResultWriter,
    save_file_format: SaveFileFormat | None = None,
):
    """
    A common pipeline end.
    """

    # time
    end_time_train_test = datetime.now()
    duration_train_test = end_time_train_test - start_time_train_test
    duration_train_test_human_readable = datetime_to_human_readable_str(duration_train_test)
    print("Training and testing took: {0}, for subpipeline {1}".format(
        duration_train_test_human_readable,
        subpipeline.value,        
    ))
    results_writer.set_result("duration_train_test", duration_train_test_human_readable)

    # save results
    if save_file_format is not None:
        results_writer.save_results_to_file(save_file_format)
    else:
        results_writer.save_results_to_file(
            str2enum(params.RESULT_SAVE_FILE_FORMAT, SaveFileFormat)
        )

def common_embedding_loop_end(
    i: int,
    params: ProgramParams,
    hyperparams: BaseHyperparams,
    length_of_labelled_graphs: int,
    start_embedding: datetime,
    nb_embedding_samples_len: int,
    nb_embedding_features: int,
):
    end_embedding = datetime.now()
    duration_embedding = end_embedding - start_embedding
    duration_embedding_human_readable = datetime_to_human_readable_str(duration_embedding)
    print(
        f" ▶ [pipeline index: {hyperparams.index}/{params.nb_pipeline_runs}]",
        f"[pipeline name: {hyperparams.pipeline_name.value}]",
        f"[graph: {i+1}/{length_of_labelled_graphs}]",
        f"Embeddings loop took: {duration_embedding_human_readable}",
        f"(embeddings len: {nb_embedding_samples_len}, features: {nb_embedding_features})",
    )