from kedro.pipeline import Pipeline, node
from .nodes import  preprocessing

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs=["x_train" , "y_train" , "x_test"],
                outputs=["x_train_preprocessed" , "y_train_preprocessed" , "x_test_preprocessed"],
                name="data_preprocessing"
            )
        ]
    )
