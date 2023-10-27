from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import traceback
import resource

from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.feature_eval.feature_eval_pipeline import feature_evaluation_pipeline
from graph_conv_net.pipelines.gcn.gcn_pipeline import gcn_pipeline
from graph_conv_net.pipelines.hyperparams import BaseHyperparams, FirstGCNPipelineHyperparams, RandomForestPipeline, generate_hyperparams
from graph_conv_net.pipelines.pipelines import PipelineNames
from graph_conv_net.pipelines.random_forest.random_forest_pipeline import random_forest_pipeline
from graph_conv_net.results.result_writer import ResultWriter
from graph_conv_net.utils.cpu_gpu_torch import print_device_info
from graph_conv_net.utils.utils import check_memory, datetime_to_human_readable_str

# -------------------- Memory limit -------------------- #
MAX_MEMORY_GB = 250  # 250 GB
MAX_MEMORY_IN_BYTES = MAX_MEMORY_GB * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, 
    (MAX_MEMORY_IN_BYTES, MAX_MEMORY_IN_BYTES)
)

def run_pipeline(
        i: int, 
        params: ProgramParams,
        hyperparams: FirstGCNPipelineHyperparams | RandomForestPipeline | BaseHyperparams,
    ):
    params.RESULTS_LOGGER.info(f"Running pipeline [index:{i}] {hyperparams.pipeline_name}...")
    params.RESULTS_LOGGER.info(f"Current Hyperparams: {hyperparams}")

    result_writer = ResultWriter()
    
    try:
        if hyperparams.pipeline_name == PipelineNames.GCNPipeline:
            assert isinstance(hyperparams, FirstGCNPipelineHyperparams), (
                "ERROR: hyperparams is not of type FirstGCNPipelineHyperparams, but of type {0}".format(type(hyperparams))
            )
            gcn_pipeline(params,  hyperparams, result_writer)
        elif hyperparams.pipeline_name == PipelineNames.ClassicMLPipeline:
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
        
        return None  # Return None if there is no error
    except Exception as e:
        return (i, e, traceback.format_exc())  # Return the index, exception, and traceback

def display_pipeline_result_if_error(
        start_time: datetime,
        hyperparams_list: list[FirstGCNPipelineHyperparams | RandomForestPipeline | BaseHyperparams],
        res: tuple[int, Exception, str]
    ):
    """
    Display the pipeline result if there is an error.
    In that case, exit the program.
    """

    idx, exception, tb = res
    end_time: datetime = datetime.now()
    duration = end_time - start_time
    duration_hour_min_sec = datetime_to_human_readable_str(
        duration
    )
    print(
        f"❌ ERROR: in pipeline [index: {idx}], "
        f"after {duration_hour_min_sec}, "
        f"for pipeline {hyperparams_list[idx].pipeline_name}, "
        f"with hyperparams {hyperparams_list[idx]}. \n"
        f"Mem2Graph GV dataset dir: {hyperparams_list[idx].input_mem2graph_dataset_dir_path}. \n"
        f"Exception: {exception} "
    )
    print(tb)
    exit(1)
    
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

                # Get the results, with error handling
                futures = list(executor.map(
                    run_pipeline, 
                    range(i, i + len(batch_hyperparams)), 
                    [params] * len(batch_hyperparams), 
                    batch_hyperparams)
                )

                # Check for exceptions and print/log them
                for future in futures:
                    memory_used_gb = check_memory()
                    print(f" | [󱙌 Program Memory: {memory_used_gb} GB] | ", end="")
                    if future:
                        display_pipeline_result_if_error(
                            start_time,
                            hyperparams_list,
                            future
                        )
                    else:
                        print(f"✅ Pipeline ran successfully")

    else:
        print("🚧 Running in DEBUG mode, so not in parallel...")
        params.nb_pipeline_runs = 1
        
        print("len(hyperparams_list): {0}".format(len(hyperparams_list)))
        first_hyperparams = hyperparams_list[243]
        assert first_hyperparams.index == 243

        res = run_pipeline(0, params, first_hyperparams)
        if res is not None:
            display_pipeline_result_if_error(
                start_time,
                hyperparams_list,
                res
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

    # -------------------- GPU -------------------- # 
    print_device_info()
    
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)