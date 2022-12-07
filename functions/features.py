import pandas as pd
import numpy as np

class features:

    def __init__(self):
        pass

    def SMA(self, df, feature, window_size):
        new_col = 'MA' + feature + str(window_size)
        df[new_col] = df[feature].rolling(window=window_size).mean()
        return df

    def Volitility(self, df, feature, window_size):
        new_col = 'VOLITILITY' + feature + str(window_size)
        returns = np.log(df[feature]/df[feature].shift())
        returns.fillna(0, inplace=True)
        df[new_col] = returns.rolling(window=window_size).std()*np.sqrt(window_size)
        return df

    def RSI(self, df, feature, window_size):
        new_col = 'RSI' + feature + str(window_size)
        delta = df[feature].diff()
        delta = delta[1:]
        up, down = delta.clip(lower=0), delta.clip(upper=0)
        roll_up = up.rolling(window_size).mean()
        roll_down = down.abs().rolling(window_size).mean()
        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        df[new_col] = RSI
        return df

    def LogVolume(self, df, feature):
         df[feature + 'log_volume'] = np.log10(df[feature + 'Volume'])
         return df

    def RangePercent(self, df, feature):
        df[feature + 'daily_range_percent'] = (df[feature+ 'High'] - df[feature+ 'Low']) / df[feature +'Close']
        return df