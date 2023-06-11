import logging
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def train_test_split_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    log = logging.getLogger(__name__)
    log.info("Splitting dataframe into train and test sets...")
    
    x, y = dataframe.drop(columns=['Fire Alarm']), dataframe['Fire Alarm']
    log.debug("Features (x): %s", x.head())
    log.debug("Target (y): %s", y.head())
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=420)
    log.info("Train and test sets created.")
    
    return xtrain, xtest, ytrain, ytest









