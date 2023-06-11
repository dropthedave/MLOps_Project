import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")

def split_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    log = logging.getLogger(__name__)
    log.info("Splitting the dataframe...")
    # order dataframe by month
    dataframe = dataframe.sort_values(by='UTC').reset_index(drop=True)

    # split dataframe into 3 equal sized datasets
    dataframe_1 = dataframe.iloc[:int(len(dataframe) / 3)]
    dataframe_2 = dataframe.iloc[int(len(dataframe) / 3):int(len(dataframe) / 3) * 2]
    dataframe_3 = dataframe.iloc[int(len(dataframe) / 3) * 2:]

    describe_to_dict_verified1 = dataframe_1.describe().to_dict()
    describe_to_dict_verified2 = dataframe_2.describe().to_dict()
    describe_to_dict_verified3 = dataframe_3.describe().to_dict()

    log.info("Splitting complete.")

    return dataframe_1, dataframe_2, dataframe_3 , describe_to_dict_verified1 , describe_to_dict_verified2, describe_to_dict_verified3



