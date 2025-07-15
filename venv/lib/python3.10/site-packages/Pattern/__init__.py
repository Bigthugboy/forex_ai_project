import pandas as pd
import numpy as np

def doji(df):
    '''doji (df) will return Doji column'''
    df['Doji'] = np.where(np.multiply((df.High - df.Low) , 0.05) > np.absolute(df.Open-df.Close), True, False)
    return df

def gravestone_doji(df):
    df['Gravestone'] = np.where( (np.multiply((df.High - df.Low) , 0.05) > np.absolute(df.Open-df.Close)) & \
        (df.Open < (df.Low + np.multiply((df.High - df.Low), 0.05))), True, False) 
    return df

def dragonfly_doji(df):
    df['Dragonfly'] = np.where( (np.multiply((df.High - df.Low) , 0.05) > np.absolute(df.Open-df.Close)) & \
        (df.Open > (df.High - np.multiply((df.High - df.Low), 0.05))), True, False) 
    return df
    
def longleg_doji(df):
    var_df = df.copy(deep=True)
    var_df['l10'] = np.multiply((var_df.High - var_df.Low), 0.1)
    var_df['l30'] = np.multiply((var_df.High - var_df.Low), 0.3)
    var_df['isupper'] = np.where(( ((var_df.High - var_df.l10) > var_df.Open) & ( (var_df.High - var_df.l30) < var_df.Open )  ), True, False)
    var_df['islower'] = np.where(( ((var_df.Low + var_df.l10) < var_df.Open) & ( (var_df.Low + var_df.l30) > var_df.Open )  ), True, False)
        
    df['LongLeg'] = np.where( var_df.Doji & (var_df.isupper | var_df.islower), True, False)
    del var_df
    
    return df
    
def Hammer_Hanging_Man(df):
    df['Hammer'] = np.where( ((((df.Close - df.Low) > (df.High - df.Open) * 2) & (df.Close >= df.Open)) |\
        (((df.Open - df.Low) > (df.High - df.Close) * 2) & (df.Open >= df.Close))  ) , True, False)
    return df

def Inv_Hammer(df):
    df['Inv_Hammer'] = np.where( (( ((df.High - df.Close) > (df.Close - df.Low) * 2) & (df.Close > df.Open)) |\
        (((df.High - df.Open) > (df.Open - df.Low) * 2) & (df.Open > df.Close))  ) , True, False)
        
    return df
    
def Spinning_Top(df):
    df['Spinning'] = np.where(( (df.Close >= ( df.Low + ((df.High - df.Low) / 3 ))) & \
        (df.Open >= ( df.Low + ((df.High - df.Low) / 3 )))  & \
            (df.Close <= ( df.High - ((df.High - df.Low) / 3 ))) &\
                (df.Open <= ( df.High - ((df.High - df.Low) / 3 ))) ) , True, False)
    return df

def Marubozu(df):
    df['Bull_Marubozu'] = np.where(((df.Open < df.Close) & (df.Open == df.Low) & (df.Close == df.High)) , True, False)
    df['Bear_Marubouzu'] = np.where(((df.Open > df.Close) & (df.Open == df.High) & (df.Close == df.Low)), True, False)
    return df

def Engulf(df):
    df['BullEngulf'] = np.where(( (df.Close.shift(1) > df.Open.shift(1)) & \
        (df.High.shift(1) < df.High) & (df.Low.shift(1) > df.Low) & (df.Open.shift(1) < df.Close) &\
            (df.Close.shift(1) > df.Open) ), True, False)
    df['BearEngulf'] = np.where(( (df.Open.shift(1) > df.Close.shift(1)) &\
        (df.High.shift(1) < df.High) & (df.Low.shift(1) > df.Low) & (df.Close.shift(1) < df.Open) &\
            (df.Open.shift(1)) > df.Close  ), True, False)
    return df

def Engulfing(df):
    df['SBullEngulf'] = np.where(( (df.Close.shift(1) > df.Open.shift(1)) & \
        (df.High.shift(1) < df.Close) & (df.Low.shift(1) > df.Open)  ), True, False)
    df['SBearEngulf'] = np.where(( (df.Open.shift(1) > df.Close.shift(1)) &\
        (df.High.shift(1) < df.Open) & (df.Low.shift(1) > df.Open)   ), True, False)
    return df

def Harami(df):
    df['BullHarami'] = np.where( ((df.High <= df.Open.shift(1)) &\
        (df.Low >= df.Close.shift(1)) & (df.Close > df.Open) ) , True, False)
    df['BearHarami'] = np.where( ((df.High <= df.Close.shift(1)) & \
        (df.Low >= df.Open.shift(1)) & (df.Close < df.Open) ) , True, False)
    return df

def DarkCloudCover(df):
    df['DarkCloud'] = np.where( ((df.Close.shift(1) > df.Open.shift(1)) &\
        (((df.Close.shift(1) + df.Open.shift(1)) / 2 ) > df.Close) & (df.Open > df.Close) &\
            (df.Open > df.Close.shift(1)) & (df.Close > df.Open.shift(1)) &
            ((df.Open - df.Close)/ (0.001 + (df.High - df.Low)) > 0.6 ) ) , True, False)
    return df

def PiercingPattern(df):
    df['Piercing'] = np.where( ((df.Close.shift(1) < df.Open.shift(1)) &\
        (((df.Close.shift(1) + df.Open.shift(1)) / 2 ) < df.Close) & (df.Open < df.Close) &\
            (df.Open < df.Close.shift(1)) & (df.Close < df.Open.shift(1)) &
            ((df.Open - df.Close)/ (0.001 + (df.High - df.Low)) < 0.6 ) ) , True, False)
    return df




