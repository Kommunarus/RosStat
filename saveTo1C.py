import warnings

import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
import numpy as np
import datetime

from scipy.special import boxcox, inv_boxcox

register_matplotlib_converters()

connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)
region = 'Российская Федерация'
start = 5
lmbda = 0.5
nforecast = 12

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


    dtx = df.ymd.values[start:]


    xt = boxcox(dta, lmbda)

    warnings.simplefilter('ignore')


    train = xt[:len(xt)-nforecast]

    df = pd.read_sql(
        'SELECT * FROM price.model WHERE region = "{}" and product="{}"'.format(region, sproduct),
        con=connection)

    for param in df.iterrows():

        mod = sm.tsa.statespace.SARIMAX(train, order=(param[1].p, param[1].d, param[1].q),
                                         seasonal_order=(param[1].sp, param[1].sd, param[1].sq, param[1].ss))
        res = mod.fit(disp=False)


        predict = res.get_prediction(end=mod.nobs + nforecast)

        p_main = inv_boxcox(predict.predicted_mean, lmbda)

        predict_ci1 = inv_boxcox(predict.conf_int(alpha=0.50)[:,0], lmbda)
        predict_ci2 = inv_boxcox(predict.conf_int(alpha=0.50)[:,1], lmbda)

        for ii in range(nforecast):
            query = "INSERT INTO price.predict(region, product, PlanningDate, model, ymd, mean, button, up) " \
                             "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
            index = ii+1
            args = (region,
                    sproduct,
                    dtx[-(nforecast+1)].strftime('%Y-%m-%d %H:%M:%S'),
                    param[1].criterion,
                    dtx[-index].strftime('%Y-%m-%d %H:%M:%S'),
                    float(p_main[-index]),
                    float(predict_ci1[-index]),
                    float(predict_ci2[-index]))
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()

connection.close()
