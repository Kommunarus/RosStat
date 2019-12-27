import numpy as np
import statsmodels.api as sm
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import MinimalFCParameters

import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from scipy.special import boxcox, inv_boxcox


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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def prepareData(data, test_size, lag=[] , ):

    data = pd.DataFrame(data.copy())

    for i in lag:
        data["lag_{}".format(i)] = data.price.shift(i+test_size)
    '''for i in lag:
        data["lag__{}".format(i)] = data.ValueVal.shift(i+test_size)'''

    data.ymd = pd.to_datetime(data["ymd"])
    data["year"] = data.ymd.dt.year


    '''features = extract_features(data[:test_index],  column_id= 'year', column_sort='ymd', n_jobs = 6,
                                column_value = 'price',
                                default_fc_parameters = MinimalFCParameters())
    features = features.dropna(axis=1)
    features['yearPast'] = features.index+1
    data = pd.merge(data, features, left_on='year', right_on='yearPast' )
    #features['yearPast2'] = features.index+2
    #data = pd.merge(data, features, left_on='year', right_on='yearPast2' )
    data.drop(["yearPast"], axis=1, inplace=True)
    #data.drop(["yearPast_y"], axis=1, inplace=True)
    #data.drop(["yearPast2"], axis=1, inplace=True)'''


    data.drop(["year"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    # считаем индекс в датафрейме, после которого начинается тестовый отрезок
    test_index = int(len(data) - test_size)
    #   test_index = int(len(data)*(1-test_size))

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:test_index].drop(["price"], axis=1)
    y_train = data.loc[:test_index]["price"]
    X_test = data.loc[test_index:].drop(["price"], axis=1)
    y_test = data.loc[test_index:]["price"]

    return X_train, X_test, y_train, y_test





def getPredictArima(region, sproduct, start = 5):

    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}"'.format(region, sproduct),
        con=connection)


    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)



    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')
    ex = data.ValueVal.values[start:]



    dta = df_train.price.values[start:]
    dttime = df_train.ymd.values[start:]
    xt = boxcox(dta, lmbda)
    train = xt[:len(xt) - nforecast]
    exog = ex[:len(xt) - nforecast]

    df = pd.read_sql(
        'SELECT * FROM price.model WHERE region = "{}" and product="{}" and criterion = "aic"'.format(region, sproduct),
        con=connection)
    # Graph
    param = df.iloc[0]

    mod = sm.tsa.statespace.SARIMAX(train, order=(param.p, param.d, param.q),
                                    seasonal_order=(param.sp, param.sd, param.sq, param.ss),
                                    )
    res = mod.fit(disp=False)

    predict = res.get_prediction(end=mod.nobs + nforecast - 1)

    p_main = inv_boxcox(predict.predicted_mean, lmbda)

    mode = sm.tsa.statespace.SARIMAX(train, order=(param.p, param.d, param.q),
                                    seasonal_order=(param.sp, param.sd, param.sq, param.ss),
                                     exog= exog)
    rese = mode.fit(disp=False)

    exog_forecast = data.iloc[-nforecast: ] ['ValueVal'].values[..., np.newaxis]
    predicte = rese.get_prediction(end=mode.nobs + nforecast - 1, exog = exog_forecast)

    p_maine = inv_boxcox(predicte.predicted_mean, lmbda)


    plt.plot(p_main[-nforecast:], label="arima")
    plt.plot(p_maine[-nforecast:], label="arimaX")
    plt.plot(dta[-nforecast:], label="fact")
    plt.axis('tight')
    plt.grid(True)
    plt.legend()
    plt.title("{} \n MAPE arima {} \n MAPE arimaX {} ".format(sproduct, round(
        mean_absolute_percentage_error(dta[-nforecast:], p_main[-nforecast:]), 2), round(mean_absolute_percentage_error(dta[-nforecast:], p_maine[-nforecast:]), 2)))

    plt.show()



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
            'Картофель'
 ]

for sproduct in products:
    getPredictArima(region, sproduct)

connection.close()
