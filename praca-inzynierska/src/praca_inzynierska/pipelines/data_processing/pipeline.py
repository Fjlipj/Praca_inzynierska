"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import load_and_preprocess_data, prepare_features_and_target, prepare_features_and_target_for_refined_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_preprocess_data,
            inputs="dataset",
            outputs="preprocessed_data",
            name="load_and_preprocess_data_node"
        ),
        node(
            func=prepare_features_and_target,
            inputs="preprocessed_data",
            outputs=["X_train", "X_test", "y_train", "y_test", "scaler", "features"],
            name="prepare_features_node"
        ),
        node(
            func=prepare_features_and_target_for_refined_model,
            inputs="preprocessed_data",
            outputs=["XX_train", "XX_test", "yy_train", "yy_test", "scaler_for_refined_model", "features_for_refined_model"],
            name="prepare_features_node_and_target_for_refined_model"
        ),
    ])