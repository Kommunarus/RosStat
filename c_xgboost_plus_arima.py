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


def getPredictXgboost(residual, pre_arima, train_arima):
    lag = [1, 2, 3, 4, 5, 6, ]
    X_train, X_test, y_train, y_test = prepareData(residual, test_size=nforecast, lag = lag,)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # задаём параметры
    params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'max_depth': 10
    }

    params = {
        'colsample_bynode': 0.8,
        'learning_rate':0.5,
        'max_depth': 5,
        'num_parallel_tree': 100,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'tree_method': 'gpu_hist'
    }

    # прогоняем на кросс-валидации с метрикой rmse
    #cv = xgb.cv(params, dtrain, metrics=('mae'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees, seed=42)

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, )

    # можно построить кривые валидации
    # cv.plot(y=['test-mae-mean', 'train-mae-mean'])

    # запоминаем ошибку на кросс-валидации
#    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]


    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    prediction_train = bst.predict(dtrain)

    test_index = len(pre_arima)


    prediction_test = bst.predict(dtest)

    y3 = train_arima
    y4 = prediction_test + pre_arima

    plt.plot(y3, label="y_test")
    plt.plot(y4, label="prediction xgb+arima")
    plt.plot(pre_arima, label="prediction arima")
    plt.axis('tight')
    plt.grid(True)
    plt.legend()
    plt.title("{} \n MAPE xgb+arima {} \n MAPE arima {}".format(sproduct, round(mean_absolute_percentage_error(y3, y4),2), round(mean_absolute_percentage_error(y3, pre_arima),2) ))

    plt.show()


def getPredictArima(region, sproduct, start = 5):

    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}"'.format(region, sproduct),
        con=connection)
    dta = df_train.price.values[start:]
    dttime = df_train.ymd.values[start:]
    #dta = dta.reindex()
    xt = boxcox(dta, lmbda)
    train = xt[:len(xt) - nforecast]

    df = pd.read_sql(
        'SELECT * FROM price.model WHERE region = "{}" and product="{}"'.format(region, sproduct),
        con=connection)
    # Graph

    for param in df.iterrows():

        mod = sm.tsa.statespace.SARIMAX(train, order=(param[1].p, param[1].d, param[1].q),
                                         seasonal_order=(param[1].sp, param[1].sd, param[1].sq, param[1].ss))
        res = mod.fit(disp=False)

        predict = res.get_prediction(end=mod.nobs + nforecast - 1)

        p_main = inv_boxcox(predict.predicted_mean, lmbda)

        return pd.DataFrame({'price':dta - p_main, 'ymd': dttime}), p_main[-nforecast:], dta[-nforecast:]


region = 'Российская Федерация'

products = ['Молоко сырое крупного рогатого скота',
            #'Пшеница мягкая 3 класса',
            #'Пшеница мягкая 5 класса',
            #'Ячмень',
            'Гречиха'
            #'Семена подсолнечника',
            #'Свекла столовая',
            #'Птица сельскохозяйственная живая',
            #'Олени северные',
            #'Картофель'
 ]

for sproduct in products:
    res, pre_arima, train_arima = getPredictArima(region, sproduct)
    cc = getPredictXgboost(res, pre_arima, train_arima)

connection.close()
