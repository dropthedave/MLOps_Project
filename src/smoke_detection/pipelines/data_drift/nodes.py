"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import shap 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn
import mlflow
import nannyml as nml

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# define a function to calculate univariate drift
def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):
    

    #define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=0.3, upper=0.7)
    constant_threshold.thresholds(data_reference)
    
    # define column names
    column_names = data_reference.columns.tolist()
    column_names.remove('y_true')
    column_names.remove('UTC')


    # Let's initialize the object that will perform the Univariate Drift calculations
    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names= column_names, #["Temperature[C]", "Humidity[%]"],
        treat_as_categorical=['y_pred'],
        # timestamp_column_name='UTC',
        chunk_number = 10, #chunk_size=50,
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
        #categorical_methods=['jensen_shannon'],
        thresholds={"jensen_shannon":constant_threshold},
        )

    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).to_df()

    #generate a report for some numeric features using KS test and evidely ai
    data_drift_report = Report(metrics=[
    DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])

    data_drift_report.run(current_data=data_analysis[column_names], 
                          reference_data=data_reference[column_names], 
                          column_mapping=None)
    data_drift_report.save_html("data/08_reporting/univariate_data_drift_report.html")
    return results




# define a function to calculate multivariate drift
def data_drift_multivariate(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):
    #define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=0.3, upper=0.7)
    constant_threshold.thresholds(data_reference)
    
    # define column names
    column_names = data_reference.columns.tolist()
    column_names.remove('y_true')
    column_names.remove('UTC')


    # Let's initialize the object that will perform the Univariate Drift calculations
    multivariate_calculator = nml.DataReconstructionDriftCalculator(
        column_names= column_names, #["Temperature[C]", "Humidity[%]"],
        # timestamp_column_name='UTC',
        chunk_number = 10, #chunk_size=50,
        thresholds={"jensen_shannon":constant_threshold},
        imputer_categorical=sklearn.impute.SimpleImputer(strategy='constant', fill_value='missing'),
        imputer_continuous=sklearn.impute.SimpleImputer(strategy='median'))

    multivariate_calculator.fit(data_reference)
    results = multivariate_calculator.calculate(data_analysis).to_df()

    #generate a report for some numeric features using KS test and evidely ai
    data_drift_report = Report(metrics=[
    DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])

    data_drift_report.run(current_data=data_analysis[column_names], 
                          reference_data=data_reference[column_names], 
                          column_mapping=None)
    data_drift_report.save_html("data/08_reporting/multivariate_data_drift_report.html")
    return results


# def plot_univariate_drift(reference_df: pd.DataFrame, analysis_df: pd.DataFrame):
#     # define column names
#     column_names = reference_df.columns.tolist()
#     column_names.remove('y_true')
#     column_names.remove('UTC')
#     # define drift calculator
#     calc = nml.UnivariateDriftCalculator(
#         column_names=column_names,
#         treat_as_categorical=['y_pred'],
#         # timestamp_column_name='UTC',
#         continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
#         categorical_methods=['chi2', 'jensen_shannon'],
#         #chunker = SizeBasedChunker(chunk_size=1000 , incomplete='append'),
#         chunk_number=10
#         )
#     # fit reference data and calculate drift
#     calc.fit(reference_df)
#     results = calc.calculate(analysis_df)

#     # display(results.filter(period='analysis', column_names=['debt_to_income_ratio']).to_df())
#     # plot drift and distribution


#     figure = results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='drift')
#     figure.show()
#     figure = results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='drift')
#     figure.show()
#     figure = results.filter(column_names=results.continuous_column_names, methods=['jensen_shannon']).plot(kind='distribution')
#     figure.show()
#     figure = results.filter(column_names=results.categorical_column_names, methods=['chi2']).plot(kind='distribution')
#     figure.show()



# def plot_multivariate_drift(reference_df: pd.DataFrame, analysis_df: pd.DataFrame, lower_threshold=0.1, upper_threshold=2):
#     constant_threshold = nml.thresholds.ConstantThreshold(lower=lower_threshold, upper=upper_threshold)
#     # define feature column names
#     feature_column_names = reference_data.columns.tolist()
#     feature_column_names.remove('y_true')
#     feature_column_names.remove('UTC')
#     feature_column_names.remove('y_pred_proba')
#     feature_column_names.remove('y_pred')
#     #define drift calculator
#     calc = nml.DataReconstructionDriftCalculator(
#         column_names=feature_column_names,
#         # timestamp_column_name='timestamp',
#         # chunk_size=1000,
#         #chunker = nml.SizeBasedChunker(chunk_size=1000 , incomplete='append'),
#         chunk_number=10,
#         #thresholds = constant_threshold,
#         imputer_categorical=SimpleImputer(strategy='constant', fill_value='missing'),
#         imputer_continuous=SimpleImputer(strategy='median'),
#     )
#     calc.fit(reference_df)
#     results = calc.calculate(analysis_df)
#     # display results
#     # display(results.filter(period='analysis').to_df())
#     # display(results.filter(period='reference').to_df())
#     figure = results.plot()
#     figure.show()