import logging
import warnings
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from kedro.pipeline import node
warnings.filterwarnings("ignore")

def preprocessing(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, reset: bool = False) -> pd.DataFrame:
    log = logging.getLogger(__name__)
    log.info("Starting preprocessing...")
    
    # preprocessing pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # ('power_transformer', PowerTransformer())
    ])
    
    log.info("Dropping 'CNT' column...")
    x_train = x_train.drop(columns=['CNT'])
    x_test = x_test.drop(columns=['CNT'])
    
    log.info("Converting 'UTC' column to datetime...")
    x_train['Timestamp'] = pd.to_datetime(x_train['UTC'], unit='s')
    x_test['Timestamp'] = pd.to_datetime(x_test['UTC'], unit='s')
    
    log.info("Dropping 'UTC' column...")
    x_train = x_train.drop(columns=['UTC'])
    x_test = x_test.drop(columns=['UTC'])
    
    log.info("Creating new columns 'Hour' and 'WeekDay'...")
    x_train['Hour'] = x_train['Timestamp'].dt.hour
    x_train['WeekDay'] = x_train['Timestamp'].dt.dayofweek
    
    x_test['Hour'] = x_test['Timestamp'].dt.hour
    x_test['WeekDay'] = x_test['Timestamp'].dt.dayofweek
    
    log.info("Dropping 'Timestamp' column...")
    x_train = x_train.drop(columns=['Timestamp'])
    x_test = x_test.drop(columns=['Timestamp'])
    
    log.info("Fitting and transforming the preprocessing pipeline...")
    x_train_preprocessed = pipeline.fit_transform(x_train)
    x_test_preprocessed = pipeline.transform(x_test)
    
    log.info("Converting ndarray to DataFrame...")
    x_train_preprocessed = pd.DataFrame(x_train_preprocessed, columns=x_train.columns)
    x_test_preprocessed = pd.DataFrame(x_test_preprocessed, columns=x_test.columns)
    
    if y_train is not None:
        log.info("Applying SMOTE...")
        smote = SMOTE(random_state=420)
        x_train_preprocessed, y_train_preprocessed = smote.fit_resample(x_train_preprocessed, y_train)
        return x_train_preprocessed, y_train_preprocessed, x_test_preprocessed
    
    log.info("Preprocessing completed.")
    return x_train_preprocessed, x_test_preprocessed






