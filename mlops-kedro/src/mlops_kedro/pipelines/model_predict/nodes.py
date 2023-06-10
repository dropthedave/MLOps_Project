import logging
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

def fit_and_predict(best_params, x, y, xtest) -> pd.DataFrame:
    """
    Fits the classifier on the given training data (x, y) and predicts using xtest.

    Args:
        best_params: The best parameters obtained from the grid search.
        x: The training data features.
        y: The training data labels.
        xtest: The test data features to predict on.

    Returns:
        ypred: The predicted values for xtest.
    """
    log = logging.getLogger(__name__)
    log.info("Fitting and predicting...")

    classifier = RandomForestClassifier(random_state=420)
    # Set the classifier's parameters
    classifier.set_params(**best_params)
    # Fit the classifier on the training data
    classifier.fit(x, y)
    
    log.info("Predicting on test data...")
    # Predict on the test data
    y_pred = classifier.predict(xtest)

    y_pred_df = pd.DataFrame(y_pred, columns=["prediction"])
    log.info("Prediction completed.")
    return y_pred_df


