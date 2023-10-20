from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.gcn.first_gcn_pipeline import first_gcn_pipeline
from graph_conv_net.pipelines.hyperparams import FirstGCNPipelineHyperparams, RandomForestPipeline
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.pipelines.random_forest.random_forest_pipeline import random_forest_pipeline
from graph_conv_net.results.result_writer import ResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str

def run_pipeline(
        i: int, 
        params: ProgramParams,
        hyperparams: FirstGCNPipelineHyperparams | RandomForestPipeline
    ):
    params.RESULTS_LOGGER.info(f"Running pipeline [index:{i}] {hyperparams.pipeline_name}...")
    params.RESULTS_LOGGER.info(f"Current Hyperparams: {hyperparams}")

    result_writer = ResultWriter()
    
    if hyperparams.pipeline_name == PipelineNames.FirstGCNPipeline:
        assert isinstance(hyperparams, FirstGCNPipelineHyperparams), (
            "ERROR: hyperparams is not of type FirstGCNPipelineHyperparams, but of type {0}".format(type(hyperparams))
        )
        _ = first_gcn_pipeline(params,  hyperparams, result_writer)
    elif hyperparams.pipeline_name == PipelineNames.RandomForestPipeline:
        assert isinstance(hyperparams, RandomForestPipeline), (
            "ERROR: hyperparams is not of type RandomForestPipeline, but of type {0}".format(type(hyperparams))
        )
        _ = random_forest_pipeline(params, hyperparams, result_writer)
    else:
        raise ValueError("ERROR: Unknown pipeline name: {0}".format(hyperparams.pipeline_name))

def main(params: ProgramParams):

    start_time = datetime.now()

    hyperparams_list: list[RandomForestPipeline | FirstGCNPipelineHyperparams] = []

    node2vec_dimensions_range = [128]
    node2vec_walk_length_range = [16, 32]
    node2vec_num_walks_range = [50]
    node2vec_p_range = [0.5, 1.0, 1.5]
    node2vec_q_range = [0.5, 1.0, 1.5]
    node2vec_window_range = [10]
    node2vec_batch_words_range = [8]
    node2vec_workers_range = [6]

    randomforest_trees_range = [100, 500, 1000]

    TRAINING_EPOCHS = 20
    NB_INPUT_GRAPHS = 600
    NB_RANDOM_FOREST_JOBS = 5

    hyperparam_index = 0
    for node2vec_dimensions in node2vec_dimensions_range:
        for node2vec_walk_length in node2vec_walk_length_range:
            for node2vec_num_walks in node2vec_num_walks_range:
                for node2vec_p in node2vec_p_range:
                    for node2vec_q in node2vec_q_range:
                        for node2vec_window in node2vec_window_range:
                            for node2vec_batch_words in node2vec_batch_words_range:
                                for node2vec_workers in node2vec_workers_range:
                                        
                                        if PipelineNames.RandomForestPipeline.value in params.cli_args.args.pipelines:
                                            for nb_trees in randomforest_trees_range:
                                                randforest_hyperparams = RandomForestPipeline(
                                                    pipeline_name=PipelineNames.RandomForestPipeline,
                                                    index=hyperparam_index,
                                                    nb_input_graphs=NB_INPUT_GRAPHS,
                                                    node2vec_dimensions=node2vec_dimensions,
                                                    node2vec_walk_length=node2vec_walk_length,
                                                    node2vec_num_walks=node2vec_num_walks,
                                                    node2vec_p=node2vec_p,
                                                    node2vec_q=node2vec_q,
                                                    node2vec_window=node2vec_window,
                                                    node2vec_batch_words=node2vec_batch_words,
                                                    node2vec_workers=node2vec_workers,
                                                    random_forest_n_estimators=nb_trees,
                                                    random_forest_n_jobs=NB_RANDOM_FOREST_JOBS,
                                                )
                                                hyperparams_list.append(randforest_hyperparams)
                                                hyperparam_index += 1

                                        if PipelineNames.FirstGCNPipeline.value in params.cli_args.args.pipelines:
                                            gcn_hyperparams = FirstGCNPipelineHyperparams(
                                                pipeline_name=PipelineNames.FirstGCNPipeline,
                                                index=hyperparam_index,
                                                nb_input_graphs=NB_INPUT_GRAPHS,
                                                node2vec_dimensions=node2vec_dimensions,
                                                node2vec_walk_length=node2vec_walk_length,
                                                node2vec_num_walks=node2vec_num_walks,
                                                node2vec_p=node2vec_p,
                                                node2vec_q=node2vec_q,
                                                node2vec_window=node2vec_window,
                                                node2vec_batch_words=node2vec_batch_words,
                                                node2vec_workers=node2vec_workers,
                                                training_epochs=TRAINING_EPOCHS,
                                            )
                                            hyperparams_list.append(gcn_hyperparams)
                                            hyperparam_index += 1

    # log the hyperparams
    params.COMMON_LOGGER.info("📝 Logging hyperparams...")
    for i in range(len(hyperparams_list)):
        params.COMMON_LOGGER.info("Hyperparams [{0}]: {1}".format(i, hyperparams_list[i]))

    if not params.DEBUG:
        print("🚀 Running pipelines...")
        params.nb_pipeline_runs = len(hyperparams_list)

        # Set the batch size
        BATCH = 6
        print(">>> BATCH: {0}".format(BATCH))

        # Main code
        with ProcessPoolExecutor(max_workers=BATCH) as executor:
            for i in range(0, len(hyperparams_list), BATCH):
                batch_hyperparams = hyperparams_list[i:i+BATCH]
                executor.map(
                    run_pipeline, 
                    range(i, i + len(batch_hyperparams)), 
                    [params] * len(batch_hyperparams), 
                    batch_hyperparams
                )
    else:
        print("🚧 Running in DEBUG mode, so not in parallel...")
        params.nb_pipeline_runs = 1
        
        print("len(hyperparams_list): {0}".format(len(hyperparams_list)))
        first_hyperparams = hyperparams_list[0]

        run_pipeline(0, params, first_hyperparams)
    
    end_time = datetime.now()
    duration = end_time - start_time
    duration_hour_min_sec = datetime_to_human_readable_str(
        duration
    )
    print("🏁 Program finished in {0}".format(
        duration_hour_min_sec
    ))

if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)