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
#region = 'Российская Федерация'
start = 5
lmbda = 0.5
nforecast = 12

'''products = ['Молоко сырое крупного рогатого скота',
            'Пшеница мягкая 3 класса',
            'Пшеница мягкая 5 класса',
            'Ячмень',
            'Гречиха',
            'Семена подсолнечника',
            'Свекла столовая',
            'Птица сельскохозяйственная живая',
            'Олени северные',
            'Картофель']'''

region = 'Краснодарский край'

products = ['Бобы соевые',
                'Виноград',
                'Зерновые и зернобобовые культуры',
                'Картофель',
                'Кукуруза',
                'Овес',
                'Пшеница мягкая 3 класса',
                'Пшеница мягкая 5 класса',
                'Птица сельскохозяйственная живая',
                'Молоко сырое крупного рогатого скота']


for sproduct in products:

    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}"'.format(region, sproduct),
        con=connection)
    dta = df.price.values[(len(df)-nforecast): (len(df)-nforecast+6)]


    dtx = df.ymd.values[(len(df)-nforecast): (len(df)-nforecast+6)]


    for ii in range(len(dta)):
        query = "INSERT INTO price.fact(region, product, ymd, price) " \
                "VALUES(%s,%s,%s,%s)"
        args = (region,
                sproduct,
                dtx[ii].strftime('%Y-%m-%d %H:%M:%S'),
                float(dta[ii]))
        cursor = connection.cursor()
        cursor.execute(query, args)
        connection.commit()

connection.close()
