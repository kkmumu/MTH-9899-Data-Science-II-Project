### MTH-9899-Data-Science-II-Project

This is the code to implement MTH 9899 final project.

To make predictions, please run main.py specifying mode.

## test merged data

test merged data (google drive link):
+ forward fill(compare two ffill values): https://drive.google.com/file/d/1RB4kmc1iW9CVGjhCfZ21ADZRKuv-GGsp/view?usp=sharing

## training data

white paper (overleaf link):
https://www.overleaf.com/9378534412nympgyccjsyk

features:
+ dropna: https://drive.google.com/file/d/1itPLOQ_lIDoo_5rA0-7qyJf7yPRqPXIp/view?usp=sharing
+ forward fill: https://drive.google.com/file/d/18K58wxYPV-nl4OYQ_xmng0-Hz9y6uDiX/view?usp=sharing 
+ forward fill(compare two ffill values): https://drive.google.com/file/d/11tT2xFVULG882XrpQB_cMDR-BjZJmq73/view?usp=sharing
+ 2014-2018 processed data features with target https://drive.google.com/file/d/1AuBQmS2qQnYJtfUxjJZikVjzgTZ_uU3P/view?usp=sharing
+ 2014-2018 merged data with dates https://drive.google.com/file/d/19sGtlbsTyhnuJap5ogIU73roeoC7HJEU/view?usp=sharing

split train/test data code:
----------------------------------------------------------------------------
path1 = "/Users/sunshangwen/Dropbox (Personal)/Mac/Desktop/Git Uploads/MTH-9899-Data-Science-II-Project/processed data/features20140101_20190101.csv"

path2 = "/Users/sunshangwen/Dropbox (Personal)/Mac/Desktop/Git Uploads/MTH-9899-Data-Science-II-Project/processed data/merged_data20140101_20190101.csv"

m1 = pd.read_csv(path1)

m2 = pd.read_csv(path2)

train = m1.loc[m1["Unnamed: 0"].isin(m2[m2["Date"] < 20180101].index)]

test = m1.loc[m1["Unnamed: 0"].isin(m2[m2["Date"] >= 20180101].index)]


---------------------------------------------------------------------------
merged data (google drive link):
+ dropna: https://drive.google.com/file/d/15of74YpmZKAxG9MDSk51Ckj-n2KDfTeD/view?usp=sharing
+ forward fill: https://drive.google.com/file/d/1CNiqLjHiAj2ej_5K9QohaDGQbJow3_dq/view?usp=sharing
+ forward fill(compare two ffill values): https://drive.google.com/file/d/1MD-dOma42A6_3DNUHcUL-cEWN0U_ri0D/view?usp=sharing


## reference
white paper 2017 reference: https://github.com/XinluXiao/MTH9899-Machine-Learning/blob/master/final%20project/machine%20learning%20paper.pdf

pickle reference:https://github.com/XinluXiao/MTH9899-Machine-Learning/blob/master/final%20project/ML_final_code_Untitiled_Group.ipynb



