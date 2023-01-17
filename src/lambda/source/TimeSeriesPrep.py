import pandas as pd

class TimeSeriesPrep():

    def __init__(self, window):
        self.WINDOW = window

    def percentage_change(self,initial,final):
        return ((final - initial) / initial)

    def expand_features(self,df):
        new_df = pd.DataFrame()
        for col in df.columns:
            if not col.startswith('label'):
                column = df[col]
                for i in range(1, self.WINDOW):
                    shifted = column.shift(i)
                    new_df['Shifted' + str(i) + col] = self.percentage_change(shifted, column)
            else:
                new_df[col] = df[col]

        return new_df
        