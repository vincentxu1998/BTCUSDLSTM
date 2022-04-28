import wget
import os

import pandas as pd
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator
import talib
import numpy as np
import gzip
import talib


#scrape data from api
csv_path = 'input/bitstampUSD.csv.gz'
if not os.path.exists(csv_path):
    wget.download('http://api.bitcoincharts.com/v1/csv/bitstampUSD.csv.gz', csv_path)

# num_lines = sum(1 for line in gzip.open('input/bitstampUSD.csv.gz','rt'))
num_lines = 59635943

df = pd.read_csv(csv_path,
                 compression='gzip',
                 sep=',',
                 quotechar='"',
                 names=['unixtime', 'price', 'volume'],
                 skiprows=num_lines-num_lines
                 )

# df['<DATETIME>'] = pd.to_datetime(df['unixtime'], unit='s')
# df = df.set_index('<DATETIME>')
df['Date'] = pd.to_datetime(df['unixtime'], unit='s')

df = df.set_index('Date')


df.drop(['unixtime'], axis=1, inplace=True)
ohlc = {
    'price': 'ohlc',
    'volume': 'sum',
}



df_4H = df.resample('4H').agg(ohlc)
df_4H.columns = df_4H.columns.droplevel()
closes = df_4H['close'].fillna(method='pad')
df_4H = df_4H.apply(lambda x: x.fillna(closes))

n=10
df_4H['RSI'] = talib.RSI(np.array(df_4H['close']), timeperiod=n)
df_4H['macd'], df_4H['macdsignal'], df_4H['macdhist'] =  talib.MACD(df_4H['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df_4H['SAR']=talib.SAR(np.array(df_4H['high']),np.array(df_4H['low']), 0.2,0.2) #Ta-lib stores result to T+1, so the input should be close, high, low at T and open at T+1
df_4H['ADX']=talib.ADX(np.array(df_4H['high']),np.array(df_4H['low']), np.array(df_4H['close']), timeperiod =n)
df_4H = df_4H.fillna(0)

df_4H = df_4H[df_4H.index > '2018-10-01']
df_4H.to_csv("input/BTCUSD4H2018.csv")




df_4Hsmall = df_4H[df_4H.index > '2021-10-01']


df_4Hsmall.to_csv("input/BTCUSD4H2021.csv")