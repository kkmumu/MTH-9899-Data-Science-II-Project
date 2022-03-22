# load libraries and prepare for main.py
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

import glob

import os

import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from statistics import median
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

import zipfile as zf

#import eli5
#from eli5.sklearn import PermutationImportance
#import warnings
#warnings.filterwarnings("ignore")

my_random_state = 2022


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input", help = "input directory")
    parser.add_argument("-o", "--output", help = "output directory")
    parser.add_argument("-p", "--provide", help = "directory containing the learned models (only for mode 2)")
    parser.add_argument("-s", "--start", help = "start date in YYYYMMDD format", type = int)
    parser.add_argument("-e", "--end", help = "end date in YYYYMMDD format", type = int)
    parser.add_argument("-m","--mode", help = "mode to choose 1 or 2", type = int)
    
    args = parser.parse_args()

    return args



def unzip(path):
    files = zf.ZipFile(path,'r')
    files.extractall()
    files.close()
    

def input_data(name, path, start_date, end_date, Id_range = None, add_Date = False):
    
    # empty list to hold files that are within certain date range
    files = []
    
    # select filed within date range
    for filename in glob.iglob(path, recursive=True):
        if (filename[-12:-4] >= str(start_date)) and (filename[-12:-4] <= str(end_date)):
            files.append(filename)


    if add_Date:
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=",")
            df['Date'] = f[-12:-4]
            df = df[df.Id.isin(Id_range)]
            dfs.append(df)
    else:
        dfs = [pd.read_csv(f, sep=",") for f in files]    
    data = pd.concat(dfs,ignore_index=True)
    data.Date = data.Date.apply(str)
    return data


def load_data(path, start_date, end_date):
    ret = input_data("return", path+"return/*.csv", start_date, end_date)

    Id_range = ret.Id.unique()
    risk = input_data("risk", path+"risk/*.csv", start_date, end_date, Id_range = Id_range, add_Date = True)
    
    ret = pd.merge(ret, risk, how="left", on=['Date', 'Id'])
    
    mdv = input_data("mdv",  path+"mdv/*.csv", start_date, end_date)
    volume = input_data("price_volume",  path+"price_volume/*.csv", start_date, end_date)    
    shout = input_data("shout",  path+"shout/*.csv", start_date, end_date)

    ret = pd.merge(ret,shout,how="left",on=['Date', 'Id'])    
    ret = pd.merge(ret,mdv,how="left",on=['Date','Id'])    
    ret = pd.merge(ret,volume,how="left",on=['Date','Time', 'Id'])
    
    print("ret shape: ", ret.shape)
    
    return ret


def compare_ffill(row):
    if row.Time[:2]!='10':
        if row.ResidualCumReturn_ffill==float("nan") or row.ResidualCumReturn_ffill<row.ResidualNoWinsorCumReturn:
            row.ResidualCumReturn_ffill = row.ResidualNoWinsorCumReturn
        if row.RawCumReturn_ffill==float("nan") or row.RawCumReturn_ffill<row.RawNoWinsorCumReturn:
            row.RawCumReturn_ffill = row.RawNoWinsorCumReturn
        if row.CumVol_ffill==float("nan") or row.CumVol_ffill<row.CumVolume:
            row.CumVol_ffill = row.CumVolume
    return row


def missing_value(ret, na_type = 1):
    # get the number of missing data points per column
    missing_values_count = ret.isnull().sum()
    print("missing_values_count: \n", missing_values_count)
    
    # percent of data that is missing
    total_cells = np.product(ret.shape)
    total_missing = missing_values_count.sum()

    # percent of data that is missing
    percent_missing = (total_missing/total_cells) * 100
    print("percent_missing: ", percent_missing)

    
    if na_type == 2:
        # fillna
        ret = ret.groupby(['Id']).apply(lambda x: x.ffill())  
    elif na_type == 3:
        # fillna_cumulative
        ret2 = pd.DataFrame(index=ret.index)
        ret2[['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill']] = ret.groupby(['Time', 'Id'])['ResidualNoWinsorCumReturn', 
                                                                                                            'RawNoWinsorCumReturn', 
                                                                                                            'CumVolume'].apply(lambda x: x.ffill())
        ret = ret.groupby(['Id']).apply(lambda x: x.ffill())  
        ret[['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill']] = ret2[['ResidualCumReturn_ffill',
                                                                                      'RawCumReturn_ffill',
                                                                                      'CumVol_ffill']]
        ret = ret.apply(lambda row: compare_ffill(row), axis = 'columns')  
        ret = ret.drop(columns=['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill'])
        new_names = ['ResidualNoWinsorCumReturn','RawNoWinsorCumReturn','CumVolume']
        old_names = ['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill']
        ret.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    # dropna    
    ret = ret.dropna()
    
    return ret


def winsorization(ret):
    # extract numeric columns
    numeric_columns = list(ret.dtypes[ret.dtypes != "object"].index)
    numeric_columns.remove('IsOpen')
    
    for feature in numeric_columns:   
        ret[str(feature+' winsorized')] = np.zeros(ret.shape[0])
        date_range = ret.Date.unique()

        for date in date_range:  
            # compute limits
            mad = ret[ret.Date==date][feature].mad()
            med = median(ret[ret.Date==date][feature])        
            llimit, ulimit = med - 5*mad, med + 5*mad

            # Create winsorized versions of the features with new column names        
            ret[str(feature+' winsorized')][ret.Date==date] = np.clip(ret[ret.Date==date][feature],llimit, ulimit)
        
    ret.to_csv('merged_data.csv',index=False)  
    return ret


def data_preprocess(input_dir, start_date, end_date, unzip_type = False):
    if unzip_type:
        unzip(input_dir)
    ret = load_data(input_dir, start_date, end_date)
    ret = missing_value(ret, na_type = 3)
    ret = winsorization(ret)
    return ret
    



