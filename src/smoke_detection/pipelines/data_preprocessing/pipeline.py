
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  clean_data, feature_engineer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["dataset1", "parameters"],
                outputs=["cleaned_dataset1", "raw_describe_dataset1", "cleaned_describe_dataset1"],
                name="clean",
            ),

            node(
                func= feature_engineer,
                inputs="cleaned_dataset1",
                outputs= "engineered_dataset1",
                name="engineering",
            ),

        ]
    )
