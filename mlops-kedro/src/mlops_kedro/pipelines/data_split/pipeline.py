from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_dataframe,
                inputs="raw_data",
                outputs=["data_split1", "data_split2", "data_split3", "data_split1_described", "data_split2_described", "data_split3_described"],
                name="split_data",
            ),
            # ... other nodes in the pipeline
        ]
    )