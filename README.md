# MTH-9899-Data-Science-II-Project

by ***Group 7: Ming Fu, Shangwen Sun, Yizhou Wang***

This is the readme file for implementing MTH 9899 final project.

Our handin contains two folders, one is code, the other is models.



```bash
> MTH9899_Group7 .
.
code
├── main.py
├── prep.py
├── features.py
├── light_gbm.py
├── xg_boost.py
├── extra_trees.py
├── regmodel.py
├── cross_validation.py
└── plot_helper.py
.
models
├── xgboost.sav
├── light_gbm.sav
├── extratrees.sav
├── ridge.sav
└── regmodel.sav
.
white_paper
.
others (hyper-parameter tuning notebooks)
```


To make predictions, please run main.py specifying mode.

examples:

## MODE I
python3 main.py -i ../train/ -o ../out/ -s 20140101 -e 20180101 -m 1

## MODE II
python3 main.py -i ../out/ -o ../out/ -p ../models/ -s 20140101 -e 20180101 -m 2


