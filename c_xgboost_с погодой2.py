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

lmbda = 1
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

def f_index_month(row):
    if row['ymd'].month in (4, 11) and row['ymd'].day == 1:
        val = 1
    else:
        val = 0
    return val


def getTestData(datal, lag =[], lagVal =[], lagS =[], AvPr = 0, AvPrVal = 0, winWeather = 0, us = False, y_past=[],  tr = 1):

    global trend
    data = datal.copy()


    data["ymd"]   = pd.to_datetime(data["ymd"])
    data["month"] = data.ymd.dt.month
    data["year"] = data.ymd.dt.year
    data['yearofchange'] = (data["ymd"] > datetime.datetime(2015,1,1))

    data = data.replace({"price": {0: np.nan}})
    data["price"].interpolate(inplace =True)
    
    data = data.fillna(method='bfill')


    data["price_boxcox"] = boxcox( data["price"], lmbda)

    if len(y_past) == 0:
        x = data['ymd'].map(datetime.datetime.toordinal)
        #trend.fit(x.values.reshape(-1, 1), data.price_boxcox.values.reshape(-1, 1))

        trend = np.poly1d(np.polyfit(x.values, data.price_boxcox.values, tr))
        #plt.plot(x, data.price_boxcox.values)
        #plt.plot(x, p(x))

    #plt.show()
    #data["Trend"] = trend.predict(data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1, 1))
    data["Trend"] = trend(data['ymd'].map(datetime.datetime.toordinal).values)


    data["PriceWithOutTrend"] = data["price_boxcox"] - data["Trend"]

    #data["diff"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(1)
    #data["diff2"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(2)
    #data.loc[0, "diff"] = 0
    #data.loc[[0,1], "diff2"] = 0
    #data.loc[data['price'] == 0, "PriceWithOutTrend"]= 0.1 # если цена равна нулю, то поставим среднюю


    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
        #data["PriceWithOutTrend{}".format(i)].fillna(0, inplace=True)
    for i in lagS:
        if i!= 0:
            data["PriceWithOutTrendS{}".format(i)] = data.PriceWithOutTrend.shift(i)
            #data["PriceWithOutTrendS{}".format(i)].fillna(0, inplace=True)
    for i in lagVal:
        data["lagValute{}".format(i)] = data.ValueVal.shift(i)
        #data["lagValute{}".format(i)].fillna(0, inplace=True)

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
        #data["mean_AvPr"].fillna(0, inplace=True)
        data.drop(["ymd_AvPr"], axis=1, inplace=True)

        for i in lag:
            data["mean_AvPr{}".format(i)] = data.mean_AvPr.shift(i)
            data["mean_AvPr{}".format(i)].fillna(0, inplace=True)

    if AvPrVal != 0:
        df = data.set_index('ymd').resample('MS', label='right').first()
        df2 = df['ValueVal'].shift().rolling(min_periods=1, window=AvPrVal).agg(['mean']).reset_index()
        df2 = df2.add_suffix('_AvPrVal')
        data = pd.merge(data, df2, left_on=['ymd'], right_on=['ymd_AvPrVal'], how='left')
        #data["mean_AvPrVal"].fillna(0, inplace=True)
        data.drop(["ymd_AvPrVal"], axis=1, inplace=True)

    if winWeather != 0:
        dt_weather = pd.read_sql('SELECT UTC as ymd, T, R FROM price.weather WHERE id = "/weather.php?id=37006"',  con=connection)
        df = dt_weather.set_index('ymd').resample('D', label='right').agg({'T': 'mean', 'R': 'mean'})
        df['ymd'] = df.index
        df['indexmonth'] = df.apply(f_index_month, axis=1)
        df['indexmonth'] = df['indexmonth'].cumsum()
        df['cumT'] = df.groupby('indexmonth')['T'].cumsum()
        df['cumR'] = df.groupby('indexmonth')['R'].cumsum()
        df.loc[ df['indexmonth'] % 2 == 0, 'cumT'] = 0
        df.loc[ df['indexmonth'] % 2 == 0, 'cumR'] = 0
        df.drop('ymd', inplace = True, axis=1,)
        df.drop('indexmonth', inplace = True, axis=1,)
        data = pd.merge(data, df, on=['ymd'], how='left', suffixes=('data', 'dt_weather'))
        #data = pd.merge(data, df, on=['ymd'], how='left', suffixes=('data', 'df2'))


    data.drop(["price"], axis=1, inplace=True)
    data.drop(["price_boxcox"], axis=1, inplace=True)
    #data.drop(["ymd"], axis=1, inplace=True)
    #data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    #data = data.dropna()
    data = data.reset_index(drop=True)

    return data

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
            #'lambda' : 0.1,
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
    #print(errors)
    # the function returns the mean of the errors on the n-1 folds
    return errors

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
        errors[i-2] = metrics(model.predict(X_testFold), y_testFold)

    # the function returns the mean of the errors on the n-1 folds
    return errors.mean()


def predict_dot(df_train, df_valute, param_b, xgbts_b, y_true, y_past, ymd_test):

    df_train_copy = df_train.copy()
    test_index = int(len(df_train_copy) - test_size)

    ytrue = df_train_copy.iloc[test_index + len(y_past), :].price
    ymd_t = df_train_copy.iloc[test_index + len(y_past), :]['ymd']
    y_true.append(ytrue)
    df_train_copy = df_train_copy[:test_index]

    past_ymd = df_train_copy.iloc[-1, :].ymd

    if len(y_past) != 0:
        for indd, past_value in enumerate(y_past):
            df_train_copy = df_train_copy.append({'price': past_value, 'ymd': past_ymd + relativedelta(months=+1)},
                                                 ignore_index=True)
        past_ymd = past_ymd + relativedelta(months=+(1 + len(y_past)))
        df_train_copy = df_train_copy.append({'price': 0, 'ymd': past_ymd}, ignore_index=True)
    data = pd.merge(df_train_copy, df_valute, left_on='ymd', right_on='dateCalendar')

    datatab = getTestData(data,
                                 lag=range(1, param_b[0]), lagVal=range(1, param_b[1]),
                                 lagS=[param_b[2]],
                                 AvPr=param_b[3], AvPrVal=param_b[4], winWeather=param_b[5],
                                 us=param_b[6], y_past=y_past,  tr = param_b[7] )

    datatab.drop(["ymd"], axis=1, inplace=True)
    X_test = datatab.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)

    predict = xgbts_b.predict(X_test)

    ypred = predict + X_test.Trend.values

    y_past.append(inv_boxcox(ypred[0], lmbda))
    ymd_test.append(ymd_t)

    return 1

#region = 'Российская Федерация'
region = 'Краснодарский край'

products = [
            'Кукуруза',
            'Огурцы открытого грунта',
            'Пшеница',
            'Свекла сахарная',
            #'Томаты (помидоры) открытого грунта',
            'Яйца куриные',
            'Ячмень',
 ]


lag = [24]
lag_s = [0]
AvPr = [0]
lag_v = [0,1]
AvPrVal = [0]
us = [True]
winWeather = [0, 1]
trend_param = [1,2]

for sproduct in products:
    print(sproduct)

    parameters = product(lag, lag_v, lag_s, AvPr, AvPrVal, winWeather, us, trend_param)
    parameters_list = list(parameters)

    err = float('inf')
    param_best = []
    all_param = []

    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, sproduct, datetime.date(2011, 1, 1), datetime.date(2020, 1, 1)),
        con=connection)

    if len(df_train) < 30:
        continue

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)


    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')


    for param in parameters_list:

        datatab= getTestData(data, lag=range(1,param[0]), lagVal=range(1,param[1]), lagS= [param[2]],
                                 AvPr = param[3], AvPrVal = param[4], winWeather = param[5], us = param[6], tr = param[7])

        ymd_train = datatab["ymd"]
        datatab.drop(["ymd"], axis=1, inplace=True)
        X_train = datatab.drop(["PriceWithOutTrend"], axis=1)
        y_train = datatab["PriceWithOutTrend"]

        xgbts = myxgBoost(3)

        err_par = performTimeSeriesCV(X_train, y_train, 5, xgbts, mape)

        all_param.append([err_par, param, deepcopy(xgbts)])
        if err > np.abs(err_par):
            err = np.abs(err_par)
            param_best = param
            xgbts_best = deepcopy(xgbts)
            X_train_best = deepcopy(X_train)
            y_train_best = deepcopy(y_train)


    print('error {:5.2f}'.format(err))
    print(param_best)


    err = float('inf')
    param_best_test = []
    '''for all_p in all_param:
        param_best_all, xgbts_best_all = all_p[1], all_p[2]

        y_past = []
        y_true = []
        ymd_test =[]
        for test_point in range(test_size):
            predict_dot(df_train, df_valute, param_best_all, xgbts_best_all, y_true, y_past, ymd_test)


        MAPE = mape(y_true, y_past)
        if err > MAPE:
            err = MAPE
            param_best_test = param_best_all
            xgbts_best_test = deepcopy(xgbts_best_all)'''


        #plt.plot(ymd_test, y_past )


    plt.figure(figsize=(10,10))
    '''y_past = []
    y_true = []
    ymd_test = []
    for test_point in range(test_size):
        predict_dot(df_train, df_valute, param_best_test, xgbts_best_test, y_true, y_past, ymd_test)
    plt.plot(ymd_test, y_past, 'ro--', linewidth=3, label='best for test')
    MAPE_best_test = mape(y_true, y_past)'''
    y_past = []
    y_true = []
    ymd_test = []
    for test_point in range(test_size):
        predict_dot(df_train, df_valute, param_best, xgbts_best, y_true, y_past, ymd_test)
    plt.plot(ymd_test, y_past, 'mo--', linewidth=3, label='best for train')
    plt.plot(ymd_train, inv_boxcox(y_train+ X_train.Trend.values, lmbda), 'yo-', linewidth=3, label='true')

    plt.axvspan(min(ymd_test), max(ymd_test) , alpha=0.5,  color='lightgrey')
    #plt.plot(ymd_test , y_true, 'yo-', linewidth=5, label='true')
    MAPE_best = mape(y_true, y_past)

    plt.axis('tight')
    plt.grid(True)
    plt.legend()
    plt.title("{} \n mape {:.2f} {} ".format(sproduct, MAPE_best, param_best))


    plt.show()


#res, pre_arima, train_arima = getPredictArima(region, sproduct)
#cc = getPredictXgboost(res, pre_arima, train_arima)

connection.close()
