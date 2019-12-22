from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.special import boxcox, inv_boxcox
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import random

r = lambda: random.randint(0,1)


connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)


nforecast = 12
lmbda = 0.5
start = 5

region = 'Российская Федерация'

products = ['Молоко сырое крупного рогатого скота',
            'Пшеница мягкая 3 класса',
            'Пшеница мягкая 5 класса',
            'Ячмень',
            'Гречиха',
            'Семена подсолнечника',
            'Свекла столовая',
            'Птица сельскохозяйственная живая',
            'Олени северные',
            'Картофель']

for sproduct in products:


    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}"'.format(region, sproduct),
        con=connection)
    dta = df.price.values[start:]
    #dta = dta.reindex()
    xt = boxcox(dta, lmbda)
    train = xt[:len(xt) - nforecast]

    df = pd.read_sql(
        'SELECT * FROM price.model WHERE region = "{}" and product="{}"'.format(region, sproduct),
        con=connection)
    # Graph
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.xaxis.grid()
    ax.yaxis.grid()
    ax.plot(inv_boxcox(xt, lmbda), 'k.')

    for param in df.iterrows():

        mod = sm.tsa.statespace.SARIMAX(train, order=(param[1].p, param[1].d, param[1].q),
                                         seasonal_order=(param[1].sp, param[1].sd, param[1].sq, param[1].ss))
        res = mod.fit(disp=False)
        #print(res.summary())

        predict = res.get_prediction(end=mod.nobs + nforecast)
        idx = np.arange(len(predict.predicted_mean))

        p_main = inv_boxcox(predict.predicted_mean, lmbda)

        predict_ci1 = inv_boxcox(predict.conf_int(alpha=0.5)[:, 0], lmbda)
        predict_ci2 = inv_boxcox(predict.conf_int(alpha=0.5)[:, 1], lmbda)


        # Plot
        col = ((r(),r(),r()))
        ax.plot(idx[:-nforecast], p_main[:-nforecast], color = col)
        ax.plot(idx[-nforecast:], p_main[-nforecast:], '--', color = col, linestyle='--', linewidth=2)
        ax.fill_between(idx, predict_ci1, predict_ci2, alpha=0.15)

    ax.set(title=sproduct)
    plt.xlim(len(dta)-50,len(dta))
#   plt.ylim(20000,27500)

    plt.show()

connection.close()