# load libraries and prepare for main.py
import numpy as np
import pandas as pd
import glob
import os

import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, make_scorer
from sklearn.base import clone
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import *
from sklearn.model_selection import *

import warnings
warnings.filterwarnings("ignore")

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


