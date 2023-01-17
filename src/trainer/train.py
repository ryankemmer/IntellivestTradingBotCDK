import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from functools import reduce
from yfinanceDAO import yfinanceDAO 
from labeler import labeler
from TimeSeriesPrep import TimeSeriesPrep
from datetime import datetime
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, make_scorer, accuracy_score
import warnings
import math
warnings.filterwarnings('ignore')
import shap

import logging
logging.basicConfig()
logger = logging.getLogger()

T_FINAL = 10 #days to look in the future
VOLITILITY_MULTIPLIER = 2 #for labelling
LOOK_BACK_DAYS = 100 #days to look back for timeseries
SEED = 1

#dynamodb = boto3.resource('dynamodb')
#resultstable = dynamodb.Table('resultstable')

#s3 = boto3.resource('s3')
#BUCKET = "modelbucketintellivest"

class trainer:

    def __init__(self):
        self.train_time = str(datetime.utcnow())[:-7]
        self.dir_name = self.train_time
        self.artifacts_dir_name = self.dir_name + '/artifacts'
        self.size_models = []
        self.side_models = []

    def split_data(self, df):
        #train test split at 10%
        split_index = int(len(df)*(10/100))
        holdout = df.tail(split_index)
        train = df.iloc[:-split_index]

        return train, holdout

    def plot_trades(self, holdout, Performace_Graph, Trade_Log, title):
        #Performance over time
        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=pd.to_datetime(holdout.index), y=Performace_Graph , name='Performance'))
        for trade in Trade_Log:
            fig.add_vline(x=pd.to_datetime(trade['date']), line_color='green' if trade['profit'] > 0 else 'red')
        fig.show()

    def plot_test_size(self, holdout, title):
        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=np.arange(holdout.label_size.size), y=holdout.label_size , name='Returns'))
        fig.add_trace(go.Scatter(x=np.arange(holdout.size_probabilities.size), y=holdout.size_probabilities , name='Probabilities'))
        fig.show()

    def plot_roc(self, fpr_train, tpr_train, fpr_test, tpr_test, title):
        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train , name="ROC curve Train (area = %0.2f)" % auc(fpr_test, tpr_test),))
        fig.add_trace(go.Scatter(x=fpr_test, y=tpr_test , name="ROC curve Test (area = %0.2f)" % auc(fpr_test, tpr_test),))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Base"))
        fig.show()

    def plot_thresholds(self, holdout, thresholds, title):
        performance = []
        for t in thresholds:
            performance.append(self.backtestStrategy(holdout, t, 10000)['Performance'])

        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=thresholds, y=performance[:] , name='Performance'))
        fig.show()

    def plot_opt_threshold(self, houldout, optimal_threshold, title):

        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=pd.to_datetime(houldout.index), y=houldout.side_probabilities , name='Threshold Graph'))
        fig.add_trace(go.Scatter(x=[pd.to_datetime(houldout.index[0]),pd.to_datetime(houldout.index[-1])], y=[optimal_threshold,optimal_threshold], name="Opt Thresh"))
        fig.show()

    def plot_pr_curve(self, precision, recall, title):

        fig = go.Figure()
        fig.update_layout(title_text=title, title_x=0.5)
        fig.add_trace(go.Scatter(x=recall, y=precision , name="PR curve Test (area = %0.2f)" % auc(recall, precision),))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,0], name="Base"))
        fig.show()

    def plot_roc_and_save(self, fpr_train, tpr_train, fpr_test, tpr_test, name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train , name="ROC curve Train (area = %0.2f)" % auc(fpr_train, tpr_train),))
        fig.add_trace(go.Scatter(x=fpr_test, y=tpr_test , name="ROC curve Test (area = %0.2f)" % auc(fpr_test, tpr_test),))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Base"))
        #fig.write_image(self.artifacts_dir_name + "/" + name)

    def hyperopt(X_train, Y_train):

        test_params = {
            'loss_function':['Logloss'],
            'eval_metric':['F1'],
            'early_stopping_rounds': [200],
            'verbose': [True],
            'random_seed': [SEED],
            'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'iterations': [100],
            'l2_leaf_reg': [1, 5, 9]
        }

        model = CatBoostClassifier()
        scorer = make_scorer(f1_score)
        grid = GridSearchCV(estimator=model, param_grid=test_params, scoring=scorer, cv=5, n_jobs=-1)

        #Hyperparamter tuning
        grid.fit(X_train, Y_train)
        best_param = grid.best_params_

        print("Found Params")
        print(best_param)

        return best_param

    def train_models(self, df_train, df_test, hyperopt= False):

        train_ind = df_train.index.to_numpy()
        test_ind = df_test.index.to_numpy()

        train_start = str(train_ind[0])[:10]
        train_end = str(train_ind[-1])[:10]

        X_train = df_train.loc[:, ~df_train.columns.str.startswith('label')]
        X_test = df_test.loc[:, ~df_test.columns.str.startswith('label')]
        Y_train = df_train[['label_side']]
        Y_test = df_test[['label_side']]

        Y_train_size = df_train['label_size'].abs()
        Y_test_size = df_test['label_size'].abs()

        classification_params = {
            'loss_function':'Logloss',
            'eval_metric':'F1',
            'early_stopping_rounds': 200,
            'verbose': False,
            'random_seed': SEED,
            'learning_rate':  0.1,
            'depth': 6,
            'l2_leaf_reg': 1
        }

        if hyperopt is True:
            best_params = hyperopt(X_train, Y_train)
            classification_params = {
                'loss_function':'Logloss',
                'eval_metric':'F1',
                'early_stopping_rounds': 200,
                'verbose': False,
                'random_seed': SEED,
                'learning_rate':  best_params['learning_rate'],
                'depth': best_params['depth'],
                'l2_leaf_reg': best_params['l2_leaf_reg']
            }


        side_model = CatBoostClassifier(**classification_params)
        side_model.fit(X_train, Y_train, 
            eval_set=(X_test, Y_test), 
            use_best_model=True, 
            plot=True)

        regression_params = {'loss_function':'RMSE',
            'eval_metric':'RMSE',
            'early_stopping_rounds': 200,
            'verbose': False,
            'random_seed': SEED
        }

        size_model = CatBoostRegressor(**regression_params)
        size_model.fit(X_train, Y_train_size,
                    eval_set=(X_test, Y_test_size), 
                    use_best_model=True, 
                    plot=True)

        side_model_name = 'side-' + train_start + '~' + train_end
        size_model_name = 'size-' + train_start + '~' + train_end

        #TODO: Save Models
        #s3.Bucket(BUCKET).upload_file("your/local/file", "dump/file")
        #model.save_model(self.dir_name + '/' + side_model_name)
        #size_model.save_model(self.dir_name + '/' + size_model_name)

        #
        # Results
        #

        #shap explainer
        shap.initjs()  
        shap_values = side_model.get_feature_importance(Pool(X_train, label=Y_train), type="ShapValues")
        shap.summary_plot(shap_values[:,:-1], X_train, plot_type="bar", show=False)

        #get prediction probabilities
        side_probabilities_train = side_model.predict_proba(X_train)[:,1]
        side_probabilities_test = side_model.predict_proba(X_test)[:,1]
        size_probabilities_train = size_model.predict(X_train)
        size_probabilities_test = size_model.predict(X_test)
        df_train['side_probabilities'] = side_probabilities_train
        df_test['side_probabilities'] = side_probabilities_test
        df_train['size_probabilities'] = size_probabilities_train
        df_test['size_probabilities'] = size_probabilities_test

        #ROC Curve
        fpr_train, tpr_train, _ = roc_curve(Y_train, side_probabilities_train)
        fpr_test, tpr_test, roc_thresholds = roc_curve(Y_test, side_probabilities_test)

        #PR Curve
        precision, recall, pr_thresholds = precision_recall_curve(Y_test, side_probabilities_test)
        fscore = 2 * (precision * recall) / (precision + recall)

        #PR Curve Train 
        precision_train, recall_train, pr_thresholds_train = precision_recall_curve(Y_train, side_probabilities_train)
        fscore_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

        #Find optimal thresh on PR curve
        ix = np.argmax(fscore)
        optimal_threshold = pr_thresholds[ix]
        expectedFscore = fscore[ix]

        #Find optimal thresh on PR curve train
        ix = np.argmax(fscore_train)
        optimal_threshold = pr_thresholds_train[ix]
        expectedFscore_train = fscore_train[ix]

        #Backtest on holdout
        backtest_results = self.backtestStrategy(df_test, optimal_threshold)

        #Plot Model Results
        self.plot_roc(fpr_train, tpr_train, fpr_test, tpr_test, side_model_name + "Train + Test ROC") 
        self.plot_pr_curve(precision, recall, "Test PR")
        self.plot_test_size(df_test, "Size Results")

        #Plot PR Threshold Performances
        self.plot_thresholds(df_test, pr_thresholds, 'Thresholds vs Profit on Holdout')
        self.plot_opt_threshold(df_test, optimal_threshold, 'Probabilities On Holdout Group')

        #Plot Trades with Optimal Thresh
        self.plot_trades(df_test, backtest_results['Performace_Graph'], backtest_results['Log'], "Trades On Holdout With Optimal Thresh")
        self.print_results(backtest_results)
        print("Optimal Threshold: " + str(optimal_threshold))

        return side_model, size_model, backtest_results

    def maxDD(self, balances):
        wealth_index = pd.Series(balances)
        previous_peaks = wealth_index.cummax()
        drawdown = wealth_index - previous_peaks
        mdd = drawdown.min()
        return mdd 

    def backtestStrategy(self, holdout, optimal_threshold, startingBalance=10000):
        side_probabilities = holdout['side_probabilities'].to_numpy()
        size_predictions = holdout['size_probabilities'].to_numpy()
        prices = holdout['label_SPY_Close'].to_numpy()
        vol = holdout['label_volitility'].to_numpy()
        dates = holdout.index.to_numpy()

        performance_graph = []
        balance_graph = []
        cash_graph = []
        shares_graph = []
        money_in_market_graph = []

        open_trades = []
        tradelog = []
        cash = startingBalance
        for i, date in enumerate(dates):

            #check existing trades
            for opentrade in open_trades:
                #check if stopped out
                if opentrade['stop'] >= prices[i]:
                    #calculate gains
                    cash += (opentrade['stop'] * opentrade['shares'])
                    #delete from trades
                    open_trades.remove(opentrade)
                    opentrade['sell_price'] = opentrade['stop']
                    opentrade['profit'] = (opentrade['stop'] - opentrade['bought_price']) * opentrade['shares']
                    tradelog.append(opentrade)
                #check if take profit
                elif opentrade['target'] <= prices[i]:
                    #calculate gains
                    cash += (opentrade['target'] * opentrade['shares'])
                    #delete from trades
                    open_trades.remove(opentrade)
                    opentrade['sell_price'] = opentrade['target']
                    opentrade['profit'] = (opentrade['target'] - opentrade['bought_price']) * opentrade['shares']
                    tradelog.append(opentrade)
                #check if expired
                elif i >=10:
                    if opentrade['date'] == dates[i-10]:
                        #calc win or loss
                        cash += (prices[i] * opentrade['shares'])
                        #delete from trades
                        open_trades.remove(opentrade)
                        opentrade['sell_price'] = prices[i]
                        opentrade['profit'] = (prices[i] - opentrade['bought_price']) * opentrade['shares']
                        tradelog.append(opentrade)

            #Performance
            shares_in_market = sum([opentrade['shares'] for opentrade in open_trades])
            money_in_market = prices[i] * shares_in_market
            total_balance = cash + money_in_market
            performance = ((total_balance - startingBalance) / startingBalance) * 100
            performance_graph.append(performance)
            balance_graph.append(total_balance)
            cash_graph.append(cash)
            shares_graph.append(shares_in_market)
            money_in_market_graph.append(money_in_market)

            #Shares to buy
            #Shares to buy is calibrated to maximize money in market
            #Max money in market (around 5 trades) should equal portfolio balance
            Shares = (total_balance / (prices[i] * 5))
            
            #Make new trades
            if side_probabilities[i] >= optimal_threshold and cash >= prices[i]:
                trade = { "date": dates[i], 
                            "bought_price": prices[i], 
                            "target": prices[i] + (prices[i] * vol[i] * VOLITILITY_MULTIPLIER),  
                            "stop": prices[i] - .3 * (prices[i] * vol[i] * VOLITILITY_MULTIPLIER), 
                            "shares": Shares,
                            "sell_price": 0,
                            "profit": 0}
                cash -= (prices[i] * Shares)
                open_trades.append(trade)

        #cleanup caluculations

        #sell current positions
        for opentrade in open_trades:
            cash += prices[i] * opentrade['shares']
            open_trades.remove(opentrade)
            opentrade['sell_price'] = prices[i]
            opentrade['profit'] = (prices[i] - opentrade['bought_price']) * opentrade['shares']
            tradelog.append(opentrade)

        total_trades = len(tradelog)
        net_loss = sum([abs(trade["profit"]) for trade in tradelog if trade["profit"] <= 0])
        net_profit = sum([trade["profit"] for trade in tradelog if trade["profit"] > 0])
        wins = len([trade["profit"] for trade in tradelog if trade["profit"] > 0])

        Total_Net_Profit = net_profit - net_loss
        Profit_Factor = 100 if net_loss == 0 else (net_profit / net_loss)
        Average_Trade_Net_Profit = 0 if total_trades == 0 else sum([trade["profit"] for trade in tradelog]) / total_trades
        Percent_Profitable =  100 if total_trades == 0 else (wins / total_trades) * 100

        return {
                'Performance': performance_graph[-1], 
                'Total_Trades': total_trades,
                'Total_Net_Profit': Total_Net_Profit,
                'Profit_Factor': Profit_Factor,
                'Percent_Profitable': Percent_Profitable,
                'Average_Trade_Net_Profit': Average_Trade_Net_Profit,
                'Max_Drawdown': self.maxDD(balance_graph),
                'Performace_Graph': performance_graph, 
                'Balanace_Graph': balance_graph,
                'Cash_Graph': cash_graph,
                'Shares_Graph': shares_graph,
                'Money_In_Market_Graph': money_in_market_graph,
                'Log': tradelog
                }

    def print_results(self, results):
        print('Performance: ' + str(results['Performance']))
        print('Total_Trades: ' + str(results['Total_Trades'])) 
        print('Total_Net_Profit: ' + str(results['Total_Net_Profit'])) 
        print('Profit_Factor: ' + str(results['Profit_Factor'])) 
        print('Percent_Profitable: ' + str(results['Percent_Profitable'])) 
        print('Average_Trade_Net_Profit: ' + str(results['Average_Trade_Net_Profit'])) 
        print('Max_Drawdown: ' + str(results['Max_Drawdown']))

    def train(self):

        yfDAOObj = yfinanceDAO()
        labelerObj = labeler(T_FINAL, VOLITILITY_MULTIPLIER)
        TimeSeriesPrepObj =  TimeSeriesPrep(LOOK_BACK_DAYS)

        historical_data = yfDAOObj.load_historical_data()
        historical_data = labelerObj.add_labels(historical_data)
        historical_data = TimeSeriesPrepObj.expand_features(historical_data)

        train, holdout = self.split_data(historical_data)

        delete_period = T_FINAL+1
        holdout = holdout.iloc[:-delete_period]

        #os.mkdir(self.dir_name)
        #os.mkdir(self.artifacts_dir_name)
        side_model, size_model, backtest_results = self.train_models(train, holdout)

        #Store results to db


def main():
    t = trainer()
    t.train()

if __name__ == "__main__":
    main()