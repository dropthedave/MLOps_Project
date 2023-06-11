import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def run_grid_search(x: pd.DataFrame,
                    y: pd.DataFrame,
                    xtest: pd.DataFrame = None,
                    ytest: pd.DataFrame = None, clf=RandomForestClassifier(random_state=420),
                    param_grid: dict = {'n_estimators': [100 
                                                         #, 200, 300, 400, 500
                                                         ],
                                       'max_depth': [3
                                                     #, 4, 5, 6, 7, 8, 9, 10
                                                     ]}):
    # run name
    clf_name = str(clf).split("(")[0]
    run_name = f"Grid Search - {clf_name}"
    
    log = logging.getLogger(__name__)
    
    # Initialize grid search
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    
    # Fit grid search
    log.info("Starting grid search...")
    grid.fit(x, y.values.ravel())
    log.info("Grid search completed.")
    
    # Log best parameters
    log.info("Best Parameters: %s", grid.best_params_)
    best_params = grid.best_params_
    
    # Log best score
    log.info("Best Score: %f", grid.best_score_)
    
    # Get the best model
    best_model = grid.best_estimator_
    
    return best_params

