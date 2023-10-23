from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.feature_eval.feature_eval_pipeline import feature_evaluation_pipeline
from graph_conv_net.pipelines.gcn.gcn_pipeline import gcn_pipeline
from graph_conv_net.pipelines.hyperparams import BaseHyperparams, FirstGCNPipelineHyperparams, RandomForestPipeline, generate_hyperparams
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.pipelines.random_forest.random_forest_pipeline import random_forest_pipeline
from graph_conv_net.results.result_writer import ResultWriter
from graph_conv_net.utils.utils import datetime_to_human_readable_str

def run_pipeline(
        i: int, 
        params: ProgramParams,
        hyperparams: FirstGCNPipelineHyperparams | RandomForestPipeline | BaseHyperparams,
    ):
    params.RESULTS_LOGGER.info(f"Running pipeline [index:{i}] {hyperparams.pipeline_name}...")
    params.RESULTS_LOGGER.info(f"Current Hyperparams: {hyperparams}")

    result_writer = ResultWriter()
    
    if hyperparams.pipeline_name == PipelineNames.GCNPipeline:
        assert isinstance(hyperparams, FirstGCNPipelineHyperparams), (
            "ERROR: hyperparams is not of type FirstGCNPipelineHyperparams, but of type {0}".format(type(hyperparams))
        )
        gcn_pipeline(params,  hyperparams, result_writer)
    elif hyperparams.pipeline_name == PipelineNames.RandomForestPipeline:
        assert isinstance(hyperparams, RandomForestPipeline), (
            "ERROR: hyperparams is not of type RandomForestPipeline, but of type {0}".format(type(hyperparams))
        )
        random_forest_pipeline(params, hyperparams, result_writer)
    elif hyperparams.pipeline_name == PipelineNames.FeatureEvaluationPipeline:
        assert isinstance(hyperparams, BaseHyperparams), (
            "ERROR: hyperparams is not of type BaseHyperparams, but of type {0}".format(type(hyperparams))
        )
        feature_evaluation_pipeline(params, hyperparams, result_writer)
    else:
        raise ValueError("ERROR: Unknown pipeline name: {0}".format(hyperparams.pipeline_name))

def main(params: ProgramParams):

    start_time = datetime.now()

    hyperparams_list = generate_hyperparams(params)
                                        
    # log the hyperparams
    params.COMMON_LOGGER.info("📝 Logging hyperparams...")
    for i in range(len(hyperparams_list)):
        params.COMMON_LOGGER.info("Hyperparams [{0}]: {1}".format(i, hyperparams_list[i]))

    if params.DRY_RUN:
        print("🔶 DRY_RUN mode, no compute instances will be launched...")
    elif not params.DEBUG:
        print("🚀 Running pipelines...")
        params.nb_pipeline_runs = len(hyperparams_list)

        # Set the batch size
        batch = params.PARALLEL_PIPELINE_BATCH_SIZE
        print("🔷 Parallel computing BATCH size: {0}".format(batch))

        # Main code
        with ProcessPoolExecutor(max_workers=batch) as executor:
            for i in range(0, len(hyperparams_list), batch):
                batch_hyperparams = hyperparams_list[i:i+batch]
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