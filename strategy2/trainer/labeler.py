import pandas as pd
import numpy as np

class labeler:

    def __init__(self, t_final, volitility_multiplier):
        # how many days we hold the stock which set the vertical barrier
        self.T_FINAL = t_final
        #the up and low boundary multipliers
        self.UPPER_LOWER_MULTIPLIERS = [volitility_multiplier, volitility_multiplier/3]

    def get_Daily_Volatility(self, close,span0=10):
        # simple percentage returns
        df0=close.pct_change()
        # 20 days, a month EWM's std as boundary
        df0=df0.ewm(span=span0).std()
        df0.dropna(inplace=True)
        return df0

    def get_3_barriers(self, daily_volatility, prices):
        #create a container
        barriers = pd.DataFrame(columns=['days_passed', 
              'price', 'vert_barrier', \
              'top_barrier', 'bottom_barrier'], \
               index = daily_volatility.index)

        for day, vol in daily_volatility.iteritems():
            days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
            #set the vertical barrier 
            if (days_passed + self.T_FINAL < len(daily_volatility.index) and self.T_FINAL != 0):
                vert_barrier = daily_volatility.index[days_passed + self.T_FINAL]
            else:
                vert_barrier = np.nan
            #set the top barrier
            if self.UPPER_LOWER_MULTIPLIERS[0] > 0:
                top_barrier = prices.loc[day] + prices.loc[day] * self.UPPER_LOWER_MULTIPLIERS[0] * vol
            else:
                #set it to NaNs
                top_barrier = pd.Series(index=prices.index)
            #set the bottom barrier
            if self.UPPER_LOWER_MULTIPLIERS[1] > 0:
                bottom_barrier = prices.loc[day] - prices.loc[day] * self.UPPER_LOWER_MULTIPLIERS[1] * vol
            else: 
                #set it to NaNs
                bottom_barrier = pd.Series(index=prices.index)
            barriers.loc[day, ['days_passed', 'price', 'vert_barrier','top_barrier', 'bottom_barrier']] = \
                days_passed, prices.loc[day], vert_barrier, \
                top_barrier, bottom_barrier

        return barriers

    def get_labels(self, barriers):

        labels = []
        size = [] # percent gained or lossed 

        for i in range(len(barriers.index)):
            start = barriers.index[i]
            end = barriers.vert_barrier[i]
            if pd.notna(end):
                # assign the initial and final price
                price_initial = barriers.price[start]
                price_final = barriers.price[end]
                # assign the top and bottom barriers
                top_barrier = barriers.top_barrier[i]
                bottom_barrier = barriers.bottom_barrier[i]
                #set the profit taking and stop loss conditons
                condition_pt = (barriers.price[start: end] >= top_barrier).any()
                condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
                #assign the labels
                if condition_pt: 
                    labels.append(1)
                else: 
                    labels.append(0)
                    #labels.append(max([(price_final - price_initial) / (top_barrier - price_initial), (price_final - price_initial)/(price_initial - bottom_barrier)], key=abs))
                size.append((price_final - price_initial) / price_initial)
            else:
                labels.append(np.nan)
                size.append(np.nan)

        return labels, size

    def add_labels(self, df):
        vol_df = self.get_Daily_Volatility(df.SPY_Close)
        prices = df.SPY_Close[vol_df.index]
        barriers = self.get_3_barriers(vol_df, prices)
        barriers.index = pd.to_datetime(barriers.index)
        labs, size = self.get_labels(barriers)
        df = df[df.index.isin(barriers.index)]
        df['label_side'] = labs
        df['label_size'] = size
        df['label_volitility'] = vol_df.to_numpy()
        df['label_target'] = df['SPY_Close'] + df['SPY_Close'] * df['label_volitility'].multiply(self.UPPER_LOWER_MULTIPLIERS[0])
        df['label_stop'] = df['SPY_Close'] - df['SPY_Close'] * df['label_volitility'].multiply(self.UPPER_LOWER_MULTIPLIERS[1])
        df['label_SPY_Close'] = df['SPY_Close']

        return df 