from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.special import boxcox, inv_boxcox
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
from modelsearch import getPredict

connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)


nforecast = 12
lmbda = 0.25

region = 'Российская Федерация'

products = ['Пшеница мягкая 3 класса']
start = 3
for sproduct in products:

    cc = [(2,1,2,0,1,1,1)] # getPredict(region, sproduct, start = start)

    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}"'.format(region, sproduct),
        con=connection)

    df = df.drop(range(start))
    df = df.reset_index(drop=True)

    dta = df.price
    xt = boxcox(dta, lmbda)
    train = xt[:len(xt) - nforecast]

    # Graph
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.xaxis.grid()
    ax.yaxis.grid()
    ax.plot(inv_boxcox(xt, lmbda), 'k.')

    for param in cc:
        print(param)
        mod = sm.tsa.statespace.SARIMAX(train, order=(int(param[0]), int(param[1]), int(param[2])),
                                         seasonal_order=(int(param[3]), int(param[4]), int(param[5]), int(12*param[6])) )
        res = mod.fit(disp=-1)
        #print(res.summary())

        predict = res.get_prediction(end=mod.nobs + nforecast)
        idx = np.arange(len(predict.predicted_mean))

        p_main = inv_boxcox(predict.predicted_mean, lmbda)

        predict_ci1 = inv_boxcox(predict.conf_int(alpha=0.50).values[:, 0], lmbda)
        predict_ci2 = inv_boxcox(predict.conf_int(alpha=0.50).values[:, 1], lmbda)


        # Plot
        ax.plot(idx[:-nforecast], p_main[:-nforecast], 'gray')
        ax.plot(idx[-nforecast:], p_main[-nforecast:], 'k--', linestyle='--', linewidth=2)
        ax.fill_between(idx, predict_ci1, predict_ci2, alpha=0.15)

    ax.set(title=sproduct)
    #plt.xlim(len(dtx)-50,len(dtx))
#   plt.ylim(20000,27500)

    plt.show()

connection.close()