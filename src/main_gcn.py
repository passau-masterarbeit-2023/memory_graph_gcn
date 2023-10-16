from graph_conv_net.params.params import ProgramParams
from graph_conv_net.pipelines.gcn_pipelines import first_gcn_pipeline

def main(params: ProgramParams):
    first_gcn_pipeline(params)

if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)