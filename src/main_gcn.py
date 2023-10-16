from datetime import datetime
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.gcn_pipelines import FirstGCNPipelineHyperparams, first_gcn_pipeline
from graph_conv_net.utils.utils import datetime_to_human_readable_str

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

    for node2vec_dimensions in node2vec_dimensions_range:
        for node2vec_walk_length in node2vec_walk_length_range:
            for node2vec_num_walks in node2vec_num_walks_range:
                for node2vec_p in node2vec_p_range:
                    for node2vec_q in node2vec_q_range:
                        for node2vec_window in node2vec_window_range:
                            for node2vec_batch_words in node2vec_batch_words_range:
                                hyperparams_list.append(
                                    FirstGCNPipelineHyperparams(
                                        node2vec_dimensions=node2vec_dimensions,
                                        node2vec_walk_length=node2vec_walk_length,
                                        node2vec_num_walks=node2vec_num_walks,
                                        node2vec_p=node2vec_p,
                                        node2vec_q=node2vec_q,
                                        node2vec_window=node2vec_window,
                                        node2vec_batch_words=node2vec_batch_words,
                                    )
                                )

    # log the hyperparams
    params.COMMON_LOGGER.info("📝 Logging hyperparams...")
    for i in range(len(hyperparams_list)):
        params.COMMON_LOGGER.info("Hyperparams [{0}]: {1}".format(i, hyperparams_list[i]))

    # run the pipeline with each hyperparams
    print("🚀 Running pipeline...")
    for i in range(len(hyperparams_list)):
        new_params = ProgramParams()
        new_params.RESULTS_LOGGER.info("Running pipeline [{0}]...".format(i))
        new_params.RESULTS_LOGGER.info("Current Hyperparams: {0}".format(hyperparams_list[i]))
        first_gcn_pipeline(
            new_params,
            hyperparams_list[i]
        )
    
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