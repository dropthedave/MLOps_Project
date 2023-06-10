
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  fit_and_predict


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=fit_and_predict,
                inputs=dict(
                    best_params="best_params",
                    x="x_train_preprocessed",
                    y="y_train_preprocessed",
                    xtest="x_test_preprocessed"
                ),
                outputs="y_pred",
                name="predict",
            ),
        ]
    )
