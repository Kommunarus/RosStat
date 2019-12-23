import warnings
from matplotlib import pyplot as plt
from itertools import product
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import statsmodels.api as sm
import numpy as np
import datetime
from tqdm import tqdm

warnings.simplefilter('ignore')

from scipy.special import boxcox, inv_boxcox




def getPredict(region, sproduct, start = 0, nforecast= 12, lmbda = 0.25):
    connection = pymysql.connect(
        host='localhost',
        user='kommunar',
        password='123',
        db='price',
        charset='utf8mb4',
        cursorclass=DictCursor
    )
    print('region {} product {}'.format(region, sproduct))
    df = pd.read_sql(
        "SELECT ymd, price FROM price.tab WHERE region = '{}' and products='{}'".format(region, sproduct),
        con=connection)


    dta = df.price[start:]
    #plt.plot(dta)
    #plt.show()

    xt = boxcox(dta, lmbda)



    n_p = range(0, 5)
    n_d = range(0, 3)
    n_q = range(0, 4)
    n_sp = range(0, 4)
    n_sd = range(0, 2)
    n_sq = range(0, 2)
    n_ss = range(0, 2)

    parameters = product(n_p, n_d, n_q, n_sp, n_sd, n_sq, n_ss)
    parameters_list = list(parameters)

    best_aic = float("inf")
    best_bic = float("inf")
    best_hqic = float("inf")

    train = xt[:len(xt) - nforecast]

    niter = 0
    for param in tqdm(parameters_list):
        niter += 1
        try:
            model = sm.tsa.statespace.SARIMAX(train, order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], int(param[6] * 12)),
                                            enforce_invertibility=False)
            res = model.fit(disp=-1)
        except:
            #print('wrong parameters:', param)
            continue

        aic = res.aic
        if aic < best_aic:
            best_aic = aic
            best_param_aic = param
        bic = res.bic
        if bic < best_bic:
            best_bic = bic
            best_param_bic = param
        hqic = res.hqic
        if hqic < best_hqic:
            best_hqic = hqic
            best_param_hqic = param


    connection.close()


    return [best_param_aic, best_param_bic, best_param_hqic]

if __name__ == '__main__':

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

    region = 'Краснодарский край'

    products = [#'Бобы соевые',
                #'Виноград',
                'Зерновые и зернобобовые культуры',
                'Картофель',
                'Кукуруза',
                'Овес',
                'Пшеница мягкая 3 класса',
                'Пшеница мягкая 5 класса',
                'Птица сельскохозяйственная живая',
                'Молоко сырое крупного рогатого скота']

    for sproduct in products:
        cc = getPredict(region, sproduct, start=5)

        for index, coef in enumerate(cc):
            query = "INSERT INTO price.model(region, product, criterion, p, d, q, sp, sd, sq, ss) " \
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            if index == 0:
                text = 'aic'
            if index == 1:
                text = 'bic'
            if index == 2:
                text = 'hqic'

            args = (region, sproduct, text, int(coef[0]),int(coef[1]),int(coef[2]), int(coef[3]),int(coef[4]),int(coef[5]), int(12*coef[6]) )
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()

    connection.close()
