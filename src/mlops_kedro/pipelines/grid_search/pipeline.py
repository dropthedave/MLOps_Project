
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  run_grid_search


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [

            node(
                func= run_grid_search,
                inputs=["x_train_preprocessed","y_train_preprocessed","x_test_preprocessed","y_test"],
                outputs= "best_params",
                name="grid_search",
            ),
        ]
    )
