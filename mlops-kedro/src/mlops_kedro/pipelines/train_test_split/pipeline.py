from kedro.pipeline import Pipeline, node
from .nodes import train_test_split_df

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_test_split_df,
                inputs="data_split1",
                outputs= ["x_train" , "x_test" , "y_train" , "y_test"],
                name="train_test_split"
            )
        ]
    )