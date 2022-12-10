import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from functools import reduce
import json
import boto3
from datetime import date, timedelta
from ModelLoaderS3 import modelObject
from TimeSeriesPrep import TimeSeriesPrep
from yfinanceDAO import yfinanceDAO
from decimal import Decimal
import alpaca_trade_api as tradeapi
from auth_params import DEV_API_KEY, DEV_SECRET_KEY, DEV_URL
from auth_params import PROD_API_KEY, PROD_SECRET_KEY, PROD_URL

LOOK_BACK_DAYS = 100
THRESH = .46

current_date = date.today()

#alpaca API
api = tradeapi.REST(DEV_API_KEY,DEV_SECRET_KEY,DEV_URL)
account = api.get_account()
current_price = float(api.get_snapshot('SPY').latest_quote.ap)
cash = float(account.cash)
total_balance = float(account.equity)

dynamodb = boto3.resource('dynamodb')
tradestable = dynamodb.Table('tradestable')

def fetch_data():

    yfinanceDAOObj = yfinanceDAO()
    data = yfinanceDAOObj.load_recent_data()
    return data

def date_by_adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    cur_date = from_date
    while business_days_to_add > 0:
        cur_date += timedelta(days=1)
        weekday = cur_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return cur_date

def make_predictions(data):

    model = modelObject()
    TimeSeriesPrepObj =  TimeSeriesPrep(LOOK_BACK_DAYS)
    data = TimeSeriesPrepObj.expand_features(data)

    side_predictions = model.predict_side(data)
    size_predictions = model.predict_size(data)

    return side_predictions[-1], size_predictions[-1]

def handler(event, context):
    logger.info(event)

    data = fetch_data()
    current_expiration_date = date_by_adding_business_days(current_date, 10)

    current_date_iso = current_date.isoformat()
    current_expiration_date_iso = current_expiration_date.isoformat()

    side_prediction, size_prediction = make_predictions(data)

    #Shares to buy
    #Shares to buy is calibrated to maximize money in market
    #Max money in market (around 5 trades) should equal portfolio balance
    Shares = (total_balance / (current_price * 5))

    #Make new trades
    if side_prediction >= THRESH and cash >= current_price:
        Target = current_price + (current_price * size_prediction)
        Stop = current_price - .3 * (current_price * size_prediction)

        #make trade on alpaca
        api.submit_order(
            symbol='SPY',
            qty=round(Shares, 2),
            side='buy',
            type='market',
            time_in_force='day',
        )

        trade = { "date": current_date_iso, 
            "bought_price": Decimal(str(round(current_price, 2))), 
            "target": Decimal(str(round(Target,2))),  
            "stop": Decimal(str(round(Stop, 2))), 
            "shares": Decimal(str(round(Shares, 2))),
            "expiration_date": current_expiration_date_iso
        }

        # add to open trades
        tradestable.put_item(
                Item = trade
            )


    return {
        'statusCode': 200,
        'body': json.dumps('Success'),
        'size_prediction' : size_prediction,
        'side_prediction' : side_prediction
    }