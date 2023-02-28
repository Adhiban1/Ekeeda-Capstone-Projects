# Cryptocurrency Price Prediction

## Clone
```bash
git clone https://github.com/Adhiban1/Ekeeda-Capstone-Projects.git
```
Run this command in terminal to clone this repo.
## Anaconda users
If you are using Anaconda. Run this code in terminal.
```bash
conda activate base
```
or you can activate your another environment. Go to install requirements.

## Install dependencies
```bash
pip install -r requirements.txt
```
Run this command in terminal to install python packages for this project.
## Create Models
```bash
python "create model.py"
```
## Run the main function
```
python main.py
```
This will create `report.html`. `report.html` contains graphs of cryptocurrencies and the tomorrow prediction result.
# Documentation
- Create a new folder `Cryptocurrency Price Prediction` for our project.
- Create `create model.py`, `functions.py` and `main.py` in this folder
- Let us write all functions in `functions.py`
## functions.py
### Importing packages
```python
import os
import pickle
import numpy as np
import yfinance as yf
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

sns.set()
```
### Historical data of coin price
```python
def historical_data(coinpair, column='Open'):
    end = datetime.now()
    start = end - timedelta(days=365)
    return yf.download(coinpair, start, end)[column]
```
Here, using `yfinance` we can get historical data of coins. This function give last one year data.
### Convert
```python
def convert(s):
    a = np.array([])

    for i in range(len(s)-50):
        a = np.append(a, s[i:51+i])

    a = a.reshape(-1, 51)
    x = a[:, :-1]
    y = a[:, -1]
    return x, y
```
This function gets historical data and take 50 days price to x and the next day price in y, x will be 2D matrix and y is also 2D but the shape of (n, 1), where n is any positive number.
### create_model
```python
def create_model(coinpair='BTC-USD'):
    lr = LinearRegression().fit(*convert(historical_data(coinpair)))

    with open(f'models/{coinpair}.pickle', 'wb') as f:
        pickle.dump(lr, f)
```
This function gets coinpair, then this function accessing `convert` and `historical_data` functions. `convert` function returns `x` and `y` numpy array. Here `LinearRegression` is used to create the model. You can use any other machine learning algorithm to create model. `lr` model is create and `x` and `y` fitted to the model. Then this model is saved as `coinpair.pickle` file in `models` folder, so create a folder `models` before running `create model.py`.
### last_50days_data
```python
def last_50days_data(coinpair, column='Open'):
    end = datetime.now()
    start = end - timedelta(days=50)
    return yf.download(coinpair, start, end)[column].to_numpy()
```
This function gets coinpair as input and it returns last 50 days historical data.
### get_model
```python
def get_model(coinpair):
    with open(f'models/{coinpair}.pickle', 'rb') as f:
        lr = pickle.load(f)
    return lr
```
This function gets the input of coinpair, and load `coinpair.pickle` file from `models` folder.
### graph
```python
def graph(coinpair):
    lr = get_model(coinpair)

    last_50 = last_50days_data(coinpair)
    last_10 = last_50[-10:]
    prediction = lr.predict(last_50days_data(coinpair).reshape(1, -1))
    combined = np.append(last_10, prediction)

    color = 'green' if last_10[-1] < prediction[0] else 'red'
    plt.plot([10, 11], combined[-2:], 'o--', color=color)
    plt.plot(range(1, 11), combined[:10], 'o-', color='blue')
    plt.title(coinpair)

    try:
        os.mkdir('graphs')
    except:
        pass

    plt.savefig(f'graphs/{coinpair}.svg')
    plt.close()
```
This function also gets coinpair as input parameter, then gives the output of last 10 days price plot and give tomorrow price prediction in that plot.
### html_report
```python
def html_report(coinpairs):
    for coinpair in coinpairs:
        print(coinpair)
        graph(coinpair)
```
This function gets many coinpairs as a list, this will use `graph` function and give the report of containg the graph of all coinpairs.
## create model.py
```python
from functions import create_model

for coinpair in ['BNB-USD', 'BTC-USD', 'TRX-USD', 'ETH-USD']:
    create_model(coinpair)
```
Create a list of all coinpairs and run `create model.py`, this will create trained model in `models` folder.
## main.py
```python
from functions import html_report
html_report(['BNB-USD', 'BTC-USD', 'TRX-USD', 'ETH-USD'])
```
This python file will give report of tomorrow price of each coinpair.