import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

############ Decision Tree ############
def feature_selection_dt(data: pd.DataFrame, X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):
    """
    This node selects the best features for the model
    Input: Engineered data, X_train, y_train, parameters
    Output: pickle file with the best features
    """

    # check in parameters.yml if feature selection is set to "rfe"
    if parameters["feature_selection"] == "rfe":
        # initialize and fit the model
        model = DecisionTreeClassifier(max_depth=parameters["dt"]["max_depth"], min_samples_split=parameters["dt"]["min_samples_split"])
        model.fit(X_train, y_train.iloc[:,0].values)
        # select the best features
        rfe = RFE(model) 
        rfe = rfe.fit(X_train, y_train.iloc[:,0].values)
        f = rfe.get_support(1)
        # get the names of the best features
        X_cols = X_train.columns[f].tolist()

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns is: {len(X_cols)}")
    return X_cols

############ Random Forest ############
def feature_selection_rf(data: pd.DataFrame, X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):
    """
    This node selects the best features for the model
    Input: Engineered data, X_train, y_train, parameters
    Output: pickle file with the best features
    """

    # check in parameters.yml if feature selection is set to "rfe"
    if parameters["feature_selection"] == "rfe":
        # initialize and fit the model
        model = RandomForestClassifier(n_estimators=parameters["rf"]["n_estimators"], max_depth=parameters["rf"]["max_depth"], max_features=parameters["rf"]["max_features"])
        model.fit(X_train, y_train.iloc[:,0].values)
        # select the best features
        rfe = RFE(model) 
        rfe = rfe.fit(X_train, y_train.iloc[:,0].values)
        f = rfe.get_support(1)
        # get the names of the best features
        X_cols = X_train.columns[f].tolist()

    log = logging.getLogger(__name__)
    log.info(f"Number of best columns is: {len(X_cols)}")
    return X_cols