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


def historical_data(coinpair, column='Open'):
    end = datetime.now()
    start = end - timedelta(days=365)
    return yf.download(coinpair, start, end)[column]


def convert(s):
    a = np.array([])

    for i in range(len(s)-50):
        a = np.append(a, s[i:51+i])

    a = a.reshape(-1, 51)
    x = a[:, :-1]
    y = a[:, -1]
    return x, y


def create_model(coinpair='BTC-USD'):
    lr = LinearRegression().fit(*convert(historical_data(coinpair)))

    with open(f'models/{coinpair}.pickle', 'wb') as f:
        pickle.dump(lr, f)


def last_50days_data(coinpair, column='Open'):
    end = datetime.now()
    start = end - timedelta(days=50)
    return yf.download(coinpair, start, end)[column].to_numpy()


def get_model(coinpair):
    with open(f'models/{coinpair}.pickle', 'rb') as f:
        lr = pickle.load(f)
    return lr


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


def html_report(coinpairs):
    for coinpair in coinpairs:
        print(coinpair)
        graph(coinpair)

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cryptocurrency Prediction</title>
</head>
<body>
<center>'''

    images = glob('*/*.svg')
    for image in images:
        html += f'\n<img src="{image}">'

    html += '\n</center>\n</body>\n</html>'

    with open('report.html', 'w') as f:
        f.write(html)
