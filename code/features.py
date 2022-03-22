### Author: Shangwen Sun

from prep import *


def create_features(args):
    """
    Create features between startdate and enddate and saved to the output directory specified.

    Parameters
    ----------
    start : int
        start date to create features (YYYYMMDD format)
    end : int
        end date to create features (YYYYMMDD format)
    input_dir : str
        directory to read the raw input data
    output_dir : str
        directory to save the output features files

    Returns
    -------
    None.

    """
    start_date = args.start
    end_date = args.end
    input_dir = args.input
    output_dir = args.output
    
    # load data from start_date to end_date
    df = data_preprocess(input_dir, start_date, end_date)
    
    # intraday return features
    temp = df[df['Time']=='10:00:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id') - df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '10/17res'
    features = temp.reset_index()

    temp = df[df['Time']=='10:00:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id') - df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '10/17raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='16:00:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id') - df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '16/17res'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='16:00:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id') - df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '16/17raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    # 17:30 ma raw return features
    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=1).mean()
    temp = temp.unstack()
    temp.name = '17ma1raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=3).mean()
    temp = temp.unstack()
    temp.name = '17ma3raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=5).mean()
    temp = temp.unstack()
    temp.name = '17ma5raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='RawNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=20).mean()
    temp = temp.unstack()
    temp.name = '17ma20raw'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    features['17ma1-3raw'] = features['17ma1raw'] - features['17ma3raw']
    features['17ma1-5raw'] = features['17ma1raw'] - features['17ma5raw']
    features['17ma1-20raw'] = features['17ma1raw'] - features['17ma20raw']

    # 17:30 ma residual return features
    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=1).mean()
    temp = temp.unstack()
    temp.name = '17ma1res'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=3).mean()
    temp = temp.unstack()
    temp.name = '17ma3res'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=5).mean()
    temp = temp.unstack()
    temp.name = '17ma5res'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='ResidualNoWinsorCumReturn winsorized',index='Date',columns='Id').rolling(window=20).mean()
    temp = temp.unstack()
    temp.name = '17ma20res'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    features['17ma1-3res'] = features['17ma1res'] - features['17ma3res']
    features['17ma1-5res'] = features['17ma1res'] - features['17ma5res']
    features['17ma1-20res'] = features['17ma1res'] - features['17ma20res']

    # market value
    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='CleanMid winsorized',index='Date',columns='Id') * df[df['Time']=='17:30:00.000'].pivot_table(values='SharesOutstanding winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'market_value'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    # intraday volume features
    temp = df[df['Time']=='10:00:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') / df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '10/17vol'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = (df[df['Time']=='16:00:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') - df[df['Time']=='10:00:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id')) / df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = '16/17vol'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    # turnover ma
    temp = (df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') / df[df['Time']=='17:30:00.000'].pivot_table(values='SharesOutstanding winsorized',index='Date',columns='Id')).rolling(window=1).mean()
    temp = temp.unstack()
    temp.name = 'turnover_ma1'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = (df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') / df[df['Time']=='17:30:00.000'].pivot_table(values='SharesOutstanding winsorized',index='Date',columns='Id')).rolling(window=3).mean()
    temp = temp.unstack()
    temp.name = 'turnover_ma3'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = (df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') / df[df['Time']=='17:30:00.000'].pivot_table(values='SharesOutstanding winsorized',index='Date',columns='Id')).rolling(window=5).mean()
    temp = temp.unstack()
    temp.name = 'turnover_ma5'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = (df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id') / df[df['Time']=='17:30:00.000'].pivot_table(values='SharesOutstanding winsorized',index='Date',columns='Id')).rolling(window=20).mean()
    temp = temp.unstack()
    temp.name = 'turnover_ma20'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    # original features 
    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='estVol winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'estVol'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='CleanMid winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'cleanMid17'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='16:00:00.000'].pivot_table(values='CleanMid winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'cleanMid16'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='10:00:00.000'].pivot_table(values='CleanMid winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'cleanMid10'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    temp = df[df['Time']=='17:30:00.000'].pivot_table(values='CumVolume winsorized',index='Date',columns='Id')
    temp = temp.unstack()
    temp.name = 'vol17'
    temp = temp.reset_index()
    features = features.merge(temp,on=['Id','Date'])

    # Merging with target
    target = df[df["Time"] == "17:30:00.000"][['Date','Id','ResidualNoWinsorCumReturn winsorized']].copy()
    merged = features.merge(target ,on =['Date','Id']).drop_duplicates()
    merged = merged.sort_values(['Id','Date'],ascending = [True, True])
    merged["y"] = merged.groupby(['Id'])["ResidualNoWinsorCumReturn winsorized"].shift(-1)
    merged = merged.dropna(how = 'any')
    merged = merged.sort_values('Date',ascending=True)
    merged = merged.drop(["Id", "Date","ResidualNoWinsorCumReturn winsorized"], axis = 1)
    
    # save to the output directory specified
    merged.to_csv(output_dir + '/' + str(start_date) + "_" + str(end_date))
        
    return merged
