text = '''import os
import pickle
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from glob import glob'''

text = text.split('\n')
text.sort(key=len)
print('\n'.join(text))