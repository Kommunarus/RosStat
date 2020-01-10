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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.figure(figsize=(10, 10))

connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)

lmbda = 0.25
test_size = 12

trend = None


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.01))) * 100

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.square((y_true - y_pred)))

def me(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred))

def getTestData(region, product, datein, dateout, lag =[], lagVal =[], lagS =[], test_size = 12, AvPr = 0, AvPrVal = 0, winWeather = 0, us = False, y_past=[]):

    global trend

    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)

    if len(df_train) == 0:
        return (df_train, 0)

    test_index = int(len(df_train) - test_size)

    ytrue       = df_train.iloc[test_index+len(y_past), :].price

    df_train = df_train[:test_index]

    past_ymd = df_train.iloc[-1,:].ymd

    if len(y_past) != 0:
        for indd,  past_value in enumerate(y_past):
            df_train = df_train.append({'price': past_value, 'ymd': past_ymd+relativedelta(months=+1)}, ignore_index=True)
        past_ymd = past_ymd + relativedelta(months=+(1+len(y_past)))
        df_train = df_train.append({'price': 0, 'ymd': past_ymd }, ignore_index=True)

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)



    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')

    data["ymd"]   = pd.to_datetime(data["ymd"])
    data["month"] = data.ymd.dt.month

    data = data.replace({"price": {0: np.nan}})
    data["price"].interpolate(inplace =True)

    data["price_boxcox"] = boxcox( data["price"], lmbda)

    if len(y_past) == 0:
        x = data['ymd'].map(datetime.datetime.toordinal)
        #trend.fit(x.values.reshape(-1, 1), data.price_boxcox.values.reshape(-1, 1))

        trend = np.poly1d(np.polyfit(x.values, data.price_boxcox.values, 2))
        #plt.plot(x, data.price_boxcox.values)
        #plt.plot(x, p(x))

    #plt.show()
    #data["Trend"] = trend.predict(data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1, 1))
    data["Trend"] = trend(data['ymd'].map(datetime.datetime.toordinal).values)


    data["PriceWithOutTrend"] = data["price_boxcox"] - data["Trend"]

    data["diff"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(1)
    #data["diff2"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(2)
    data.loc[0, "diff"] = 0
    #data.loc[[0,1], "diff2"] = 0
    #data.loc[data['price'] == 0, "PriceWithOutTrend"]= 0.1 # если цена равна нулю, то поставим среднюю


    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
        data["PriceWithOutTrend{}".format(i)].fillna(0, inplace=True)
    for i in lagS:
        if i!= 0:
            data["PriceWithOutTrendS{}".format(i)] = data.PriceWithOutTrend.shift(i)
            data["PriceWithOutTrendS{}".format(i)].fillna(0, inplace=True)
    for i in lagVal:
        data["lagValute{}".format(i)] = data.ValueVal.shift(i)
        data["lagValute{}".format(i)].fillna(0, inplace=True)

    # средние , максимум, минимум за квартал , полгода, год


    if us:
        meanPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('mean')
        maxPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('max')
        minPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('min')

        data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
        data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
        data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    if AvPr!= 0:
        df = data.set_index('ymd').resample('MS', label='right').first()
        df1 = df['PriceWithOutTrend'].shift().rolling(min_periods=1, window=AvPr).agg(['mean']).reset_index()
        df1 = df1.add_suffix('_AvPr')
        data = pd.merge(data, df1, left_on=['ymd'], right_on=['ymd_AvPr'], how='left')
        data["mean_AvPr"].fillna(0, inplace=True)
        data.drop(["ymd_AvPr"], axis=1, inplace=True)

        for i in lag:
            data["mean_AvPr{}".format(i)] = data.mean_AvPr.shift(i)
            data["mean_AvPr{}".format(i)].fillna(0, inplace=True)

    if AvPrVal != 0:
        df = data.set_index('ymd').resample('MS', label='right').first()
        df2 = df['ValueVal'].shift().rolling(min_periods=1, window=AvPrVal).agg(['mean']).reset_index()
        df2 = df2.add_suffix('_AvPrVal')
        data = pd.merge(data, df2, left_on=['ymd'], right_on=['ymd_AvPrVal'], how='left')
        data["mean_AvPrVal"].fillna(0, inplace=True)
        data.drop(["ymd_AvPrVal"], axis=1, inplace=True)

    if winWeather != 0:
        dt_weather = pd.read_sql('SELECT UTC as ymd, T, R FROM price.weather WHERE id = "/weather.php?id=37099"',  con=connection)
        df = dt_weather.set_index('ymd').resample('D', label='right').agg({'T': 'mean', 'R': 'mean'})
        df1 = df['T'].shift().rolling(min_periods=1, window=winWeather).agg(['mean', 'sum']).reset_index()
        df2 = df['R'].shift().rolling(min_periods=1, window=winWeather).agg(['sum']).reset_index()
        data = pd.merge(data, df1, on=['ymd'], how='left', suffixes=('data', 'df1'))
        data = pd.merge(data, df2, on=['ymd'], how='left', suffixes=('data', 'df2'))


    data.drop(["price"], axis=1, inplace=True)
    data.drop(["price_boxcox"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
    data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    #data = data.dropna()
    data = data.reset_index(drop=True)

    return data, ytrue

class myxgBoost(BaseEstimator, TransformerMixin):
    def __init__(self, depth, reg_lambda=0, reg_alpha=0):
        self.depth = depth
        #self.reg_lambda = reg_lambda
        #self.reg_alpha = reg_alpha

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)

        # задаём параметры
        params = {
            'objective': 'reg:squarederror',
            #'booster': 'gblinear',
            'booster': 'gbtree',
            'max_depth': self.depth,
            #'eval_metric': 'rmse',
            'lambda' : 0.1,
            #'min_child_weight ' : 3,
            #'subsample': 0.8,
            #'num_parallel_tree': 100,
            #'tree_method': 'gpu_hist',
            #'colsample_bynode': 0.8,
            #'learning_rate': 1
        }

        self.bst = xgb.train(params, dtrain)
        return self

    def predict(self, Xtest):
        dtest = xgb.DMatrix(Xtest)
        pred = self.bst.predict(dtest)

        return pred

def performTimeSeries(X_train, y_train, model, metrics):

    model.fit(X_train, y_train)
    errors = metrics(y_train, model.predict(X_train))
    print(errors)
    # the function returns the mean of the errors on the n-1 folds
    return errors


def predict_dot(region, sproduct, param_best_all, xgbts_best_all, y_true, y_past):
    data, ytrue = getTestData(region, sproduct, datein=datetime.date(2011, 1, 1), dateout=datetime.date(2020, 1, 1),
                              lag=range(1, param_best_all[0]), lagVal=range(1, param_best_all[1]),
                              lagS=[param_best_all[2]],
                              AvPr=param_best_all[3], AvPrVal=param_best_all[4], winWeather=param_best_all[5],
                              us=param_best_all[6], y_past=y_past, test_size=test_size, )

    X_test = data.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)

    predict = xgbts_best_all.predict(X_test)

    ypred = predict + X_test.Trend.values

    y_true.append(ytrue)
    y_past.append(inv_boxcox(ypred[0], lmbda))

    return (y_true, y_past)

region = 'Российская Федерация'
#region = 'Краснодарский край'


products = [
            #'Семена подсолнечника',
            #'Гречиха',
            #'Молоко сырое крупного рогатого скота',
            #'Пшеница мягкая 3 класса',
            #'Пшеница мягкая 5 класса',
            #'Ячмень',
            #'Свекла столовая',
            #'Птица сельскохозяйственная живая',
            'Олени северные',
            #'Картофель'
 ]

for sproduct in products:
    print(sproduct)

    lag = [2,4,6,8,10]
    lag_v = [0,6]
    lag_s = [0,12]
    AvPr = [0,3,6]
    AvPrVal = [0]
    us = [False, True]
    winWeather = [0]

    parameters = product(lag, lag_v, lag_s, AvPr, AvPrVal, winWeather, us)
    parameters_list = list(parameters)

    err = 99999
    param_best = []
    all_param = []
    for param in parameters_list:

        data, _ = getTestData(region, sproduct, datein=datetime.date(2011, 1, 1), dateout=datetime.date(2020, 1, 1),
                       lag=range(1,param[0]), lagVal=range(1,param[1]), lagS= [param[2]], AvPr = param[3], AvPrVal = param[4], winWeather = param[5], us = param[6], test_size=test_size)

        if len(data) ==0:
            continue

        X_train = data.drop(["PriceWithOutTrend"], axis=1)
        y_train = data["PriceWithOutTrend"]

        xgbts = myxgBoost(5)

        err_par = performTimeSeries(X_train, y_train, xgbts, mape)

        all_param.append([err_par, param, deepcopy(xgbts)])
        if err > np.abs(err_par):
            err = np.abs(err_par)
            param_best = param
            xgbts_best = deepcopy(xgbts)
            X_train_best = deepcopy(X_train)
            y_train_best = deepcopy(y_train)


    print('error {:5.2f}'.format(err))
    print(param_best)

    if len(data) != 0:

        err = 99999
        param_best_test = []
        for all_p in all_param:
            param_best_all, xgbts_best_all = all_p[1], all_p[2]

            y_past = []
            y_true = []
            for test_point in range(test_size):
                predict_dot(region, sproduct, param_best_all, xgbts_best_all, y_true, y_past)


            MAPE = mape(y_true, y_past)
            if err > MAPE:
                err = MAPE
                param_best_test = all_p
                xgbts_best_test = deepcopy(xgbts_best_all)

            #predict_X_train = xgbts_best.predict(X_train_best)

            #fig, axs = plt.subplots(2, figsize=(9,13))


            plt.plot(y_past )
            #plt.plot(y_past, label="predict {:.2f}".format(MAPE))
#            plt.plot(np.concatenate((inv_boxcox(predict_X_train + X_train_best.Trend.values, lmbda) , y_past)) , label = "predict")

        plt.plot(y_true, '+', label='true')

        y_past = []
        y_true = []
        for test_point in range(test_size):
            predict_dot(region, sproduct, param_best_test, xgbts_best_test, y_true, y_past)
            plt.plot(y_past, '*', label='best')

        #        plt.plot(np.concatenate((inv_boxcox(y_train_best + X_train_best.Trend.values, lmbda), y_true)), label='true')
        #plt.axvspan(len(predict_X_train), len(predict_X_train)+len(y_past), alpha=0.5, color='lightgrey')
        plt.axis('tight')
        plt.grid(True)
        plt.legend()
        plt.title("{}, mape {:.2f} \n {} \n {}".format(sproduct, MAPE, param_best, param_best_test))


        plt.show()


    #res, pre_arima, train_arima = getPredictArima(region, sproduct)
    #cc = getPredictXgboost(res, pre_arima, train_arima)

connection.close()
