from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import MinimalFCParameters
import numpy as np

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

def prepareData(data, lag_start=5, lag_end=20, test_size=12):

    data = pd.DataFrame(data.copy())


    test_index = int(len(data) - test_size)

    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.price.shift(i)

    data.ymd = pd.to_datetime(data["ymd"])
    data["year"] = data.ymd.dt.year


    '''features = extract_features(data[:test_index],  column_id= 'year', column_sort='ymd', n_jobs = 6,
                                column_value = 'price',
                                default_fc_parameters = MinimalFCParameters())
    features = features.dropna(axis=1)
    features['yearPast'] = features.index+1
    data = pd.merge(data, features, left_on='year', right_on='yearPast' )
    features['yearPast2'] = features.index+2
    data = pd.merge(data, features, left_on='year', right_on='yearPast2' )
    data.drop(["yearPast_x"], axis=1, inplace=True)
    data.drop(["yearPast_y"], axis=1, inplace=True)
    data.drop(["yearPast2"], axis=1, inplace=True)'''


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


def getPredict(region, sproduct,  scale=1.96):
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

    #df.set_index('ymd')

    df.price = boxcox(df.price, lmbda)

    X_train, X_test, y_train, y_test = prepareData(df, test_size=12, lag_start=12, lag_end=24,)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # задаём параметры
    params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree'
    }
    trees = 1000

    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees, seed=0)

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].values.argmin())

    # можно построить кривые валидации
    # cv.plot(y=['test-mae-mean', 'train-mae-mean'])

    # запоминаем ошибку на кросс-валидации
#    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))

    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    prediction_train = inv_boxcox(bst.predict(dtrain), lmbda)
    y_train = inv_boxcox(y_train, lmbda)
    ax1.plot(y_train, label="y_train")
    ax1.plot(prediction_train, label="prediction")
    ax1.axis('tight')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("{} \n MAPE {}".format(sproduct, round(mean_absolute_percentage_error(y_train, prediction_train))))

    # и на тестовом
    prediction_test = inv_boxcox(bst.predict(dtest),lmbda)
    y_test = inv_boxcox(y_test, lmbda)
    ax2.plot(list(y_test), label="y_test")
    ax2.plot(prediction_test, label="prediction")
    ax2.axis('tight')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("{} \n MAPE {}".format(sproduct, round(mean_absolute_percentage_error(y_test, prediction_test))))

    plt.show()

    connection.close()


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
    cc = getPredict(region, sproduct)
