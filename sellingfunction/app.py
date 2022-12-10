import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import date, datetime
import boto3
import json
from decimal import Decimal
from auth_params import DEV_API_KEY, DEV_SECRET_KEY, DEV_URL
from auth_params import PROD_API_KEY, PROD_SECRET_KEY, PROD_URL

current_date = date.today().isoformat()
now = datetime.now().isoformat()

#alpaca API
api = tradeapi.REST(DEV_API_KEY,DEV_SECRET_KEY,DEV_URL)
account = api.get_account()

dynamodb = boto3.resource('dynamodb')
tradestable = dynamodb.Table('tradestable')
historictrades = dynamodb.Table('historicaltradestable')

current_price = float(api.get_snapshot('SPY').latest_quote.ap)

def close_trade(opentrade, outcome):
    #close trade with alpaca
    api.submit_order(
        symbol='SPY',
        qty=opentrade['shares'],
        side='sell',
        type='market',
        time_in_force='day'
    )
    #calculate profit
    profit = (current_price - opentrade['bought_price']) * opentrade['shares']
    #delete from trades
    tradestable.delete_item(Key = opentrade)
    #manipulate JSON 
    opentrade['sell_price'] = current_price
    opentrade['profit'] = Decimal(str(round(profit,2)))
    opentrade['date_sold'] = current_date
    opentrade['time_sold'] = now
    opentrade['outcome'] = outcome
    #add to historictrades
    historictrades.put_item(
        Item = opentrade
    )

def handler(event, context):
    logger.info(event)

    open_trades = tradestable.scan()['Items']
    #check existing trades
    for opentrade in open_trades:

        trade_expiration = opentrade['expiration_date']

        if opentrade['stop'] >= current_price: 
            close_trade(opentrade, 'STOP')
        elif opentrade['target'] <= current_price:
            close_trade(opentrade, 'PROFIT')
        elif datetime.fromisoformat(opentrade['date']) >= datetime.fromisoformat(trade_expiration):
            close_trade(opentrade, 'EXPIRED')

    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }
