from functools import reduce
import pandas as pd
import yfinance as yf
from features import features

class yfinanceDAO:

    def __init__(self):
        pass

    def drop_extra_data(self, full_df):
        full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Open')))]
        full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='High')))]
        full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Low')))]
        full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Adj Close')))]
        full_df = full_df.fillna(0)
        for col in full_df.columns:
            if (full_df[col] == 0).all():
                full_df = full_df.drop(col, axis=1)
        return full_df

    def load_historical_data(self):

        prefixes = [
            'SPY_', 
            'Energy_', 
            'Materials_', 
            'Industrial_', 
            'Utilities_', 
            'Health_', 
            'Financial_', 
            'Consumer_Discretionary_', 
            'Consumer_Staples_', 
            'Technology_', 
            'Real_Estate_', 
            'TYBonds_', 
            'VIX_'
        ]

        SPY_daily = yf.download('SPY')
        energy_daily = yf.download('XLE')
        materials_daily = yf.download('XLB')
        industrial_daily = yf.download('XLI')
        utilities_daily = yf.download('XLU')
        health_daily = yf.download('XLV')
        financial_daily = yf.download('XLF')
        consumer_discretionary_daily = yf.download('XLY')
        consumer_staples_daily = yf.download('XLP')
        technology_daily = yf.download('XLK')
        real_estate_daily = yf.download('VGSIX')
        TYBonds_daily = yf.download('^TNX')
        VIX_daily = yf.download('^VIX')

        data_frames = [ 
            SPY_daily, 
            energy_daily, 
            materials_daily, 
            industrial_daily, 
            utilities_daily, 
            health_daily, 
            financial_daily, 
            consumer_discretionary_daily, 
            consumer_staples_daily, 
            technology_daily, 
            real_estate_daily, 
            TYBonds_daily, VIX_daily
        ]

        for i, df in enumerate(data_frames):
            data_frames[i] = df.rename(columns=lambda x: prefixes[i] + x)

        full_df = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True), data_frames)

        for prefix in prefixes:
            for suffix in ['Open', 'High', 'Low','Close', 'Volume']:
                full_df[prefix + suffix].interpolate(method='linear', inplace=True)

        full_df.index = pd.to_datetime(full_df.index)
        full_df = full_df.loc[full_df.index > '1999-01-01']
        
        #feature engineering
        f = features()
        for prefix in prefixes:
            full_df = f.LogVolume(full_df, prefix)
            full_df = f.RangePercent(full_df, prefix)
            for window in [7,20,50,200]:
                full_df = f.SMA(full_df, prefix + 'Close', window)
                full_df = f.Volitility(full_df, prefix + 'Close', window)
                full_df = f.RSI(full_df, prefix + 'Close', window)
        
        full_df = self.drop_extra_data(full_df)

        return full_df

    def load_recent_data(self):

        prefixes = [
            'SPY_', 
            'Energy_', 
            'Materials_', 
            'Industrial_', 
            'Utilities_', 
            'Health_', 
            'Financial_', 
            'Consumer_Discretionary_', 
            'Consumer_Staples_', 
            'Technology_', 
            'Real_Estate_', 
            'TYBonds_', 
            'VIX_'
        ]

        SPY_daily = yf.download('SPY', period = "1y")
        energy_daily = yf.download('XLE', period = "1y")
        materials_daily = yf.download('XLB', period = "1y")
        industrial_daily = yf.download('XLI', period = "1y")
        utilities_daily = yf.download('XLU', period = "1y")
        health_daily = yf.download('XLV', period = "1y")
        financial_daily = yf.download('XLF', period = "1y")
        consumer_discretionary_daily = yf.download('XLY', period = "1y")
        consumer_staples_daily = yf.download('XLP', period = "1y")
        technology_daily = yf.download('XLK', period = "1y")
        real_estate_daily = yf.download('VGSIX', period = "1y")
        TYBonds_daily = yf.download('^TNX', period = "1y")
        VIX_daily = yf.download('^VIX', period = "1y")

        data_frames = [ 
            SPY_daily, 
            energy_daily, 
            materials_daily, 
            industrial_daily, 
            utilities_daily, 
            health_daily, 
            financial_daily, 
            consumer_discretionary_daily, 
            consumer_staples_daily, 
            technology_daily, 
            real_estate_daily, 
            TYBonds_daily, VIX_daily
        ]

        for i, df in enumerate(data_frames):
            data_frames[i] = df.rename(columns=lambda x: prefixes[i] + x)

        full_df = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True), data_frames)

        for prefix in prefixes:
            for suffix in ['Open', 'High', 'Low','Close', 'Volume']:
                full_df[prefix + suffix].interpolate(method='linear', inplace=True)

        full_df.index = pd.to_datetime(full_df.index)
        
        #feature engineering
        f = features()
        for prefix in prefixes:
            full_df = f.LogVolume(full_df, prefix)
            full_df = f.RangePercent(full_df, prefix)
            for window in [7,20,50,200]:
                full_df = f.SMA(full_df, prefix + 'Close', window)
                full_df = f.Volitility(full_df, prefix + 'Close', window)
                full_df = f.RSI(full_df, prefix + 'Close', window)
        
        full_df = self.drop_extra_data(full_df)

        return full_df
