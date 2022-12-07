import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from functools import reduce
import json
import boto3
from datetime import date
from ModelLoaderS3 import modelObject
from TimeSeriesPrep import TimeSeriesPrep
from yfinanceDAO import yfinanceDAO
from decimal import Decimal

LOOK_BACK_DAYS = 100
THRESH = .47

total_balance = 2000
cash = 2000

def fetch_data():

    yfinanceDAOObj = yfinanceDAO()
    data = yfinanceDAOObj.load_recent_data()
    return data

def make_predictions(data):

    model = modelObject()
    TimeSeriesPrepObj =  TimeSeriesPrep(LOOK_BACK_DAYS)
    data = TimeSeriesPrepObj.expand_features(data)

    side_predictions = model.predict_side(data)
    size_predictions = model.predict_size(data)

    return side_predictions[-1], size_predictions[-1]

def handler(event, context):
    logger.info(event)

    dynamodb = boto3.resource('dynamodb')
    tradestable = dynamodb.Table('tradestable')
    historictrades = dynamodb.Table('historicaltradestable')

    data = fetch_data()

    current_price = data['SPY_Close'].tail(1).to_numpy()[0]
    current_date = data.index.tolist()[-1].isoformat()
    current_expiration_date = data.index.tolist()[-11].isoformat()

    side_prediction, size_prediction = make_predictions(data)

    open_trades = tradestable.scan()['Items']
    #check existing trades
    for opentrade in open_trades:
        #check if stopped out
        if opentrade['stop'] >= current_price:
            #delete from trades
            tradestable.delete_item(Key = opentrade)
            #manipulate JSON 
            opentrade['sell_price'] = opentrade['stop']
            opentrade['profit'] = (opentrade['stop'] - opentrade['bought_price']) * opentrade['shares']
            #add to historictrades
            historictrades.put_item(
                Item = opentrade
            )
        #check if take profit
        elif opentrade['target'] <= current_price:
            #delete from trades
            tradestable.delete_item(Key = opentrade)
            #manipulate JSON 
            opentrade['sell_price'] = opentrade['target']
            opentrade['profit'] = (opentrade['target'] - opentrade['bought_price']) * opentrade['shares']
            #add to historictrades
            historictrades.put_item(
                Item = opentrade
            )
        #check if expired:
        elif opentrade['date'] == current_expiration_date:
            #delete from trades
            tradestable.delete_item(Key = opentrade)
            #manipulate JSON 
            opentrade['sell_price'] = Decimal(current_price)
            opentrade['profit'] = (current_price - opentrade['bought_price']) * opentrade['shares']
            #add to historictrades
            historictrades.put_item(
                Item = opentrade
            )

    #Shares to buy
    #Shares to buy is calibrated to maximize money in market
    #Max money in market (around 5 trades) should equal portfolio balance
    Shares = (total_balance / (current_price * 5))

    #Make new trades
    if side_prediction >= THRESH and cash >= current_price:
        Target = current_price + (current_price * size_prediction)
        Stop = current_price - .3 * (current_price * size_prediction)
        trade = { "date": current_date, 
                    "bought_price": Decimal(current_price), 
                    "target": Decimal(Target),  
                    "stop": Decimal(Stop), 
                    "shares": Decimal(Shares),
                    "sell_price": 0,
                    "profit": 0}
        # add to open trades
        tradestable.put_item(
                Item = trade
            )

    return {
        'statusCode': 200,
        'body': json.dumps('Success'),
        'size_prediction' : size_prediction,
        'side_prediction' : size_prediction
    }