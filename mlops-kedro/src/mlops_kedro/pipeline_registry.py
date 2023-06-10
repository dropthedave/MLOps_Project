"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_kedro.pipelines import data_split ,  data_preprocessing, grid_search , train_test_split , model_predict

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_split_stage = data_split.create_pipeline()
    train_test_split_Stage = train_test_split.create_pipeline()
    preprocessing_stage = data_preprocessing.create_pipeline()
    grid_search_stage = grid_search.create_pipeline()
    prediction_stage = model_predict.create_pipeline()
    
    return {
        "data_split": data_split_stage,
        "train_test_split": train_test_split_Stage,
        "preprocessing": preprocessing_stage,
        "grid_search": grid_search_stage,
        "prediction": prediction_stage,
        "__default__": data_split_stage + train_test_split_Stage + preprocessing_stage + grid_search_stage + prediction_stage
    }
