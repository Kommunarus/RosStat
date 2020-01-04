import numpy as np
import statsmodels.api as sm
from itertools import product
from tqdm import tqdm
from dateutil.relativedelta import *
from copy import deepcopy
from statsmodels.tsa.seasonal import seasonal_decompose
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import MinimalFCParameters
import datetime
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from scipy.special import boxcox, inv_boxcox
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
test_size = 12

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def getTrainData(region, product, datein, dateout, lag, lagVal, lagS, test_size):


    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)

    test_index = int(len(df_train) - test_size)
    df_train = df_train[:test_index]

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)



    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')

    data["price"] = boxcox( data["price"], lmbda)


    trend = LinearRegression()
    x = data['ymd'].map(datetime.datetime.toordinal)
    trend.fit(x.values.reshape(-1,1), data.price.values.reshape(-1,1))

    data["Trend"] =  trend.predict( data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    data["PriceWithOutTrend"] = data["price"] - data["Trend"]

    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagS:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagVal:
        data["lagValute_{}".format(i)] = data.ValueVal.shift(i+test_size)

    # средние , максимум, минимум за квартал , полгода, год
    data.ymd = pd.to_datetime(data["ymd"])

    data["month"] = data.ymd.dt.month

    meanPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('mean')
    maxPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('max')
    minPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('min')

    data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
    data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
    data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    data.drop(["price"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
    data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    return data, trend

def getTestData(region, product, datein, dateout, lag, lagVal, lagS, test_size, trend, y_past):

    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)


    test_index = int(len(df_train) - test_size)
    ytrue = df_train.iloc[test_index+len(y_past), :].price
    df_train = df_train[:test_index]

    past_ymd = df_train.iloc[-1,:].ymd

    for indd,  past_value in enumerate(y_past):
        df_train = df_train.append({'price': past_value, 'ymd': past_ymd+relativedelta(months=+1)}, ignore_index=True)
    past_ymd = past_ymd + relativedelta(months=+(1+len(y_past)))
    df_train = df_train.append({'price': 0, 'ymd': past_ymd }, ignore_index=True)

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)



    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')


    data["price"] = boxcox( data["price"], lmbda)

    data["Trend"] =  trend.predict( data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    data["PriceWithOutTrend"] = data["price"] - data["Trend"]

    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagS:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagVal:
        data["lagValute_{}".format(i)] = data.ValueVal.shift(i+test_size)

    # средние , максимум, минимум за квартал , полгода, год
    data.ymd = pd.to_datetime(data["ymd"])

    data["month"] = data.ymd.dt.month

    meanPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('mean')
    maxPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('max')
    minPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('min')

    data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
    data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
    data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    data.drop(["price"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
    data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    return data, ytrue

class myxgBoost(BaseEstimator, TransformerMixin):
    def __init__(self, depth):
        self.depth = depth

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)

        # задаём параметры
        params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'max_depth': self.depth
        }

        self.bst = xgb.train(params, dtrain, )
        return self

    def predict(self, Xtest):
        dtest = xgb.DMatrix(Xtest)
        pred = self.bst.predict(dtest)

        return pred

def performTimeSeriesCV(X_train, y_train, number_folds, model, metrics):


    k = int(np.floor(float(X_train.shape[0]) / number_folds))

    errors = np.zeros(number_folds-1)

    # loop from the first 2 folds to the total number of folds
    for i in range(2, number_folds + 1):
        split = float(i-1)/i

        X = X_train[:(k*i)]
        y = y_train[:(k*i)]

        index = int(np.floor(X.shape[0] * split))

        # folds used to train the model
        X_trainFolds = X[:index]
        y_trainFolds = y[:index]

        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        model.fit(X_trainFolds, y_trainFolds)
        errors[i-2] = metrics(y_testFold, model.predict(X_testFold))

    # the function returns the mean of the errors on the n-1 folds
    return errors.mean()

def getPredictArima(region, sproduct, start = 5):

    ex = data.ValueVal.values[start:]



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
                                         seasonal_order=(param[1].sp, param[1].sd, param[1].sq, param[1].ss),
                                        )
        res = mod.fit(disp=False)


        return pd.DataFrame({'price':dta - p_main, 'ymd': dttime}), p_main[-nforecast:], dta[-nforecast:]


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
    print(sproduct)

    lag = range(3, 8)
    lag_v = range(1, 3)
    lag_s = range(11, 13)
    depth = range(3,10,3)

    parameters = product(lag, lag_v, lag_s, depth)
    parameters_list = list(parameters)

    err = 99999
    param_best = []
    for param in parameters_list:

        data, trend = getTrainData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                       lag=range(3,param[0]), lagVal=range(1,param[1]), lagS= range(11,param[2]), test_size=test_size)


        X_train = data.drop(["PriceWithOutTrend"], axis=1)
        y_train = data["PriceWithOutTrend"]

        xgbts = myxgBoost(param[3])

        err_par = performTimeSeriesCV(X_train, y_train, 5, xgbts, mean_absolute_percentage_error)
        if err > err_par:
            err = err_par
            param_best = param
            xgbts_best = deepcopy(xgbts)


    print('min error CV {:5.2f}'.format(err))
    print(param_best)

    y_past = []
    y_true = []
    for test_point in range(test_size):


        data, ytrue = getTestData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                       lag=range(3, param_best[0]), lagVal=range(1, param_best[1]), lagS=range(11, param_best[2]),
                       test_size=test_size, trend = trend, y_past = y_past)


        X_test = data.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)
        y_test = data.iloc[[-1]]["PriceWithOutTrend"]

        predict = xgbts_best.predict(X_test)

        ypred = predict + X_test.Trend.values

        y_true.append(ytrue)
        y_past.append(inv_boxcox(ypred[0], lmbda))

    MAPE = mean_absolute_percentage_error(y_true, y_past)
    plt.plot(y_true, label = 'true')
    plt.plot(y_past, label = "predict")
    plt.axis('tight')
    plt.grid(True)
    plt.legend()
    plt.title("{}, mape {:.2}".format(sproduct, MAPE))
    plt.show()


    #res, pre_arima, train_arima = getPredictArima(region, sproduct)
    #cc = getPredictXgboost(res, pre_arima, train_arima)

connection.close()
