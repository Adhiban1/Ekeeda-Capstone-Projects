#!/home/adhiban/anaconda3/bin/python
from functions import create_model

for coinpair in ['BNB-USD', 'BTC-USD', 'TRX-USD', 'ETH-USD']:
    create_model(coinpair)
