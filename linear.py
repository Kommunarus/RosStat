from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
import numpy as np

import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)

nforecast = 12

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

    features = extract_features(data[:test_index],  column_id= 'year', column_sort='ymd', n_jobs = 6,
                                column_value = 'price',
                                default_fc_parameters = MinimalFCParameters())
    features = features.dropna(axis=1)
    features['yearPast'] = features.index+1
    #features['yearPastPast'] = features['yearPast']+1


    data = pd.merge(data, features, left_on='year', right_on='yearPast' )
    data.drop(["year"], axis=1, inplace=True)
    data.drop(["yearPast"], axis=1, inplace=True)
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


def getPredict(region, sproduct):
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

    X_train, X_test, y_train, y_test = prepareData(df, test_size=12, lag_start=12, lag_end=15)
    lr = LinearRegression(normalize=False)
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)
    plt.figure(figsize=(10, 10))
    plt.plot(prediction, "r", label="prediction")
    plt.plot(y_test.values, label="actual")
    plt.legend(loc="best")
    plt.title("{}\n MAPE {}".format(sproduct, round(mean_absolute_percentage_error(y_test, prediction))))
    plt.grid(True);
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
