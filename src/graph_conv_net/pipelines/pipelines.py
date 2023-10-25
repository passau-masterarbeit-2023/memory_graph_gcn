from enum import Enum

class PipelineNames(Enum):
    # GCN models
    GCNPipeline = "gcn-pipeline"

    # Classic ML models
    ClassicMLPipeline = "classic-ml-pipeline"

    # Feature evaluation
    FeatureEvaluationPipeline = "feature-evaluation-pipeline"

class GCNSubpipelineNames(Enum):
    VerySimpleGCNPipeline = "very-simple-gcn-pipeline"
    SimpleGCNPipeline = "simple-gcn-pipeline"
    FirstGCNPipeline = "first-gcn-pipeline"
    GCNWithDropoutPipeline = "gcn-with-dropout-pipeline"
    AdvancedGCNPipeline = "advanced-gcn-pipeline"

class ClassicMLSubpipelineNames(Enum):
    RandomForestPipeline = "random-forest-pipeline"
    LogisticRegressionPipeline = "logistic-regression-pipeline"
    SGDClassifierPipeline = "sgd-classifier-pipeline"

class FeatureEvaluationSubpipelineNames(Enum):
    PearsonCorrelationPipeline = "pearson-correlation-pipeline"
    KendallCorrelationPipeline = "kendall-correlation-pipeline"
    SpearmanCorrelationPipeline = "spearman-correlation-pipeline"