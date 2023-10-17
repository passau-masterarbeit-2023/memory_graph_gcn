from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.gcn_pipelines import FirstGCNPipelineHyperparams, first_gcn_pipeline
from graph_conv_net.utils.utils import datetime_to_human_readable_str

def run_pipeline(i, hyperparams):
    new_params = ProgramParams()
    new_params.RESULTS_LOGGER.info(f"Running pipeline [{i}]...")
    new_params.RESULTS_LOGGER.info(f"Current Hyperparams: {hyperparams}")
    _ = first_gcn_pipeline(new_params, hyperparams)

def main(params: ProgramParams):

    start_time = datetime.now()

    hyperparams_list = []

    node2vec_dimensions_range = [16, 128]
    node2vec_walk_length_range = [16, 64]
    node2vec_num_walks_range = [20, 50]
    node2vec_p_range = [0.5, 1.0, 1.5]
    node2vec_q_range = [0.5, 1.0, 1.5]
    node2vec_window_range = [10, 20]
    node2vec_batch_words_range = [4, 16]

    hyperparam_index = 0
    for node2vec_dimensions in node2vec_dimensions_range:
        for node2vec_walk_length in node2vec_walk_length_range:
            for node2vec_num_walks in node2vec_num_walks_range:
                for node2vec_p in node2vec_p_range:
                    for node2vec_q in node2vec_q_range:
                        for node2vec_window in node2vec_window_range:
                            for node2vec_batch_words in node2vec_batch_words_range:
                                hyperparams_list.append(
                                    FirstGCNPipelineHyperparams(
                                        index=hyperparam_index,
                                        node2vec_dimensions=node2vec_dimensions,
                                        node2vec_walk_length=node2vec_walk_length,
                                        node2vec_num_walks=node2vec_num_walks,
                                        node2vec_p=node2vec_p,
                                        node2vec_q=node2vec_q,
                                        node2vec_window=node2vec_window,
                                        node2vec_batch_words=node2vec_batch_words,
                                    )
                                )
                                hyperparam_index += 1

    # log the hyperparams
    params.COMMON_LOGGER.info("📝 Logging hyperparams...")
    for i in range(len(hyperparams_list)):
        params.COMMON_LOGGER.info("Hyperparams [{0}]: {1}".format(i, hyperparams_list[i]))

    # Set the batch size
    BATCH = 6

    # Main code
    print("🚀 Running pipeline...")
    with ProcessPoolExecutor(max_workers=BATCH) as executor:
        for i in range(0, len(hyperparams_list), BATCH):
            batch_hyperparams = hyperparams_list[i:i+BATCH]
            executor.map(run_pipeline, range(i, i + len(batch_hyperparams)), batch_hyperparams)
    
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