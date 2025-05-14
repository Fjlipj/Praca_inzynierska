"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_random_forest, train_random_forest_with_BPM, evaluate_model, evaluate_model_with_BPM, autoML


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_random_forest,
            inputs=["X_train", "y_train"],
            outputs="trained_model",
            name="train_random_forest_node"
        ),
        node(
            func=train_random_forest_with_BPM,
            inputs=["XX_train", "yy_train"],
            outputs="trained_model_with_BPM",
            name="train_random_forest_with_BPM_node"
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test", "features"],
            outputs=["model_metrics", "predictions"],
            name="evaluate_model_node"
        ),
        node(
            func=evaluate_model_with_BPM,
            inputs=["trained_model_with_BPM", "XX_test", "yy_test", "features_for_refined_model"],
            outputs=["model_metrics_with_BPM", "predictions_with_BPM"],
            name="evaluate_model_with_BPM_node"
        ),
        node(
            func=autoML,
            inputs=["preprocessed_data"],
            outputs="autoML_model",
            name="autoML"
            ),
    ])
