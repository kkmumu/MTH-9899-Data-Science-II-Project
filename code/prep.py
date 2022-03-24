### @brief: load libraries and preprocess data
### @author: Ming Fu, Shangwen Sun

import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
import time

import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib

import glob
import os
import bz2
import pickle
import zipfile as zf
import argparse

import pandas as pd
import numpy as np
from math import *

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import *
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import *
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from statistics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs


from sklearn.ensemble import ExtraTreesRegressor
# from lightgbm import LGBMRegressor
# from xgboost import XGBRegressor
import itertools


# import eli5
# from eli5.sklearn import PermutationImportance
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
            df['Date'] = int(f[-12:-4])
            df = df[df.Id.isin(Id_range)]
            dfs.append(df)
    else:
        dfs = [pd.read_csv(f, sep=",") for f in files]    
    data = pd.concat(dfs, ignore_index=True)
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



def missing_value(ret):
    # get the number of missing data points per column
    missing_values_count = ret.isnull().sum()
    print("missing_values_count:\n", missing_values_count)
    
    # percent of data that is missing
    total_cells = np.product(ret.shape)
    total_missing = missing_values_count.sum()

    # percent of data that is missing
    percent_missing = (total_missing/total_cells) * 100
    print("percent_missing: ", percent_missing)

    # fillna_cumulative forward fill
    ret[['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill']] = ret.groupby(['Time', 'Id'])[['ResidualNoWinsorCumReturn', 'RawNoWinsorCumReturn', 'CumVolume']].fillna(method = "ffill", axis = 1)
    
    # make sure the cumulative return at 17:30 is larger than 10:00, 16:00
    ret[['ResidualNoWinsorCumReturn', 'RawNoWinsorCumReturn', 'CumVolume']] = ret.groupby(['Id'])[['ResidualNoWinsorCumReturn', 'RawNoWinsorCumReturn', 'CumVolume']].fillna(method = "ffill", axis = 1)
    
    ret['ResidualNoWinsorCumReturn'] = ret[['ResidualNoWinsorCumReturn','ResidualCumReturn_ffill']].max(axis = 1)
    ret['RawNoWinsorCumReturn'] = ret[['RawNoWinsorCumReturn', 'RawCumReturn_ffill']].max(axis = 1)
    ret['CumVolume'] = ret[['CumVolume', 'CumVol_ffill']].max(axis = 1)
    
    ret = ret.drop(['ResidualCumReturn_ffill','RawCumReturn_ffill','CumVol_ffill'], axis = 1)
    
    # dropna    
    ret = ret.dropna(axis = 0)
    
    return ret


def clip_MAD(x):
    m = x.median()
    mad = x.mad()
    return x.clip(m - 5 * mad, m + 5 * mad)


def cross_sectional_winsorization(ret):
    
    # extract numeric columns
    numeric_columns = list(ret.dtypes[ret.dtypes != "object"].index)
    numeric_columns.remove('IsOpen')
    
    # clip using 5 MAD
    for feature in numeric_columns:
        ret[str(feature+' winsorized')] = ret.groupby(["Date", "Time"])[feature].transform(clip_MAD)
    
    return ret


def data_preprocess(input_dir, output_dir, start_date, end_date, unzip_type = False):
    if unzip_type:
        unzip(input_dir)
        
    t0 = time.time()
    ret = load_data(input_dir, start_date, end_date)
    t1 = time.time()
    print("Load data......elapsed time: ", t1 - t0, "s......")
    
    ret = missing_value(ret)
    t2 = time.time()
    print("Fill missing data......elapsed time: ", t2 - t1, "s......")
    
    ret = cross_sectional_winsorization(ret)
    t3 = time.time()
    print("Winsorize data......elapsed time: ", t3 - t2, "s......")
       
    # save preprocessed data to directory
    ret.to_csv(output_dir + 'winsorized_data' + str(start_date) + "_" + str(end_date) + '.csv', index=False) 
    t4 = time.time()
    print("Save data......Finished......elapsed time: ", t4 - t3, "s......")
    
    return ret
    



