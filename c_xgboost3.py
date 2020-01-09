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
from sklearn.pipeline import Pipeline
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



def getTrainData(region, product, datein, dateout, lag, lagVal, lagS, AvPr, AvPrVal, test_size):


    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)

    if len(df_train) == 0:
        return (df_train, 0)


    test_index = int(len(df_train) - test_size)
    df_train = df_train[:test_index]

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}" '.format("USD"),
        con=connection)



    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar')

    data["price_boxcox"] = boxcox( data["price"], lmbda)


    trend = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False))])
    x = data['ymd'].map(datetime.datetime.toordinal)
    trend.fit(x.values.reshape(-1,1), data.price_boxcox.values.reshape(-1,1))

    data["Trend"] =  trend.predict( data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    data["PriceWithOutTrend"] = data["price_boxcox"] - data["Trend"]
    data.loc[data['price'] == 0, "PriceWithOutTrend"]= 0.01 # если цена равна нулю, то поставим среднюю

    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagS:
        if i!= 0:
           data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagVal:
        data["lagValute_{}".format(i)] = data.ValueVal.shift(i)

    # средние , максимум, минимум за квартал , полгода, год
    data.ymd = pd.to_datetime(data["ymd"])

    data["month"] = data.ymd.dt.month

    meanPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('mean')
    maxPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('max')
    minPrice = data.groupby('month')['PriceWithOutTrend'].aggregate('min')

    data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
    data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
    data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    df = data.set_index('ymd').resample('MS', label='right').first()
    df1 = df['PriceWithOutTrend'].shift().rolling(min_periods=1, window=AvPr).agg(['mean', 'median']).reset_index()
    data = pd.merge(data, df1, on=['ymd'], how='left')
    if AvPrVal != 0:
        df2 = df['ValueVal'].shift().rolling(min_periods=1, window=AvPrVal).agg(['mean', 'median']).reset_index()
        data = pd.merge(data, df2, on=['ymd'], how='left')


    data.drop(["price"], axis=1, inplace=True)
    data.drop(["price_boxcox"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
    data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    return data, trend

def getTestData(region, product, datein, dateout, lag, lagVal, lagS, test_size, trend, AvPr, AvPrVal, y_past):

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


    data["price_boxcox"] = boxcox( data["price"], lmbda)

    data["Trend"] =  trend.predict( data['ymd'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    data["PriceWithOutTrend"] = data["price_boxcox"] - data["Trend"]
    data.loc[data['price'] == 0, "PriceWithOutTrend"]= 0.1 # если цена равна нулю, то поставим среднюю

    for i in lag:
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagS:
        if i!= 0:
            data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
    for i in lagVal:
        data["lagValute_{}".format(i)] = data.ValueVal.shift(i)

    # средние , максимум, минимум за квартал , полгода, год
    data.ymd = pd.to_datetime(data["ymd"])

    data["month"] = data.ymd.dt.month

    meanPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('mean')
    maxPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('max')
    minPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('min')

    data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
    data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
    data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    df = data.set_index('ymd').resample('MS', label='right').first()
    df1 = df['PriceWithOutTrend'].shift().rolling(min_periods=1, window=AvPr).agg(['mean', 'median']).reset_index()
    data = pd.merge(data, df1, on=['ymd'], how='left')
    if AvPrVal != 0:
        df2 = df['ValueVal'].shift().rolling(min_periods=1, window=AvPrVal).agg(['mean', 'median']).reset_index()
        data = pd.merge(data, df2, on=['ymd'], how='left')

    data.drop(["price"], axis=1, inplace=True)
    data.drop(["price_boxcox"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
    data.drop(["month"], axis=1, inplace=True)
    data.drop(["dateCalendar"], axis=1, inplace=True)
    #data.drop(["Trend"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)

    return data, ytrue

class myxgBoost(BaseEstimator, TransformerMixin):
    def __init__(self, depth, reg_lambda=0, reg_alpha=0):
        self.depth = depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)

        # задаём параметры
        params = {
            'objective': 'reg:squarederror',
            #'booster': 'gblinear',
            'booster': 'gbtree',
            'max_depth': self.depth,
            'eval_metric': 'rmse',

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

def performTimeSeries(X_train, y_train, model, metrics):

    model.fit(X_train, y_train)
    errors = metrics(y_train, model.predict(X_train))

    # the function returns the mean of the errors on the n-1 folds
    return errors


region = 'Российская Федерация'
#region = 'Краснодарский край'


products = [
            'Семена подсолнечника',
            'Гречиха',
            'Молоко сырое крупного рогатого скота',
            'Пшеница мягкая 3 класса',
            'Пшеница мягкая 5 класса',
            'Ячмень',
            'Свекла столовая',
            'Птица сельскохозяйственная живая',
            'Олени северные',
            'Картофель'
 ]

for sproduct in products:
    print(sproduct)

    lag = [3,5,7,9]
    lag_v = [0,1,2,3]
    lag_s = [0,12]
    AvPr = [2,4,6]
    AvPrVal = [0,2,4]

    parameters = product(lag, lag_v, lag_s, AvPr, AvPrVal)
    parameters_list = list(parameters)

    err = 99999
    param_best = []
    for param in parameters_list:

        data, trend = getTrainData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                       lag=range(1,param[0]), lagVal=range(1,param[1]), lagS= [param[2]], AvPr = param[3], AvPrVal = param[4], test_size=test_size)

        if len(data) ==0:
            continue

        X_train = data.drop(["PriceWithOutTrend"], axis=1)
        y_train = data["PriceWithOutTrend"]

        xgbts = myxgBoost(7)

        err_par = performTimeSeries(X_train, y_train, xgbts, mean_absolute_percentage_error)
        if err > err_par:
            err = err_par
            param_best = param
            xgbts_best = deepcopy(xgbts)
            X_train_best = deepcopy(X_train)
            y_train_best = deepcopy(y_train)


    print('error {:5.2f}'.format(err))
    print(param_best)

    lag = [3,5,7,9]
    lag_s = [0,12]
    AvPr = [2,4,6]

    parameters = product(lag, lag_s, AvPr)
    parameters_list = list(parameters)

    err = 99999
    param_best_withoutVal = []
    for param in parameters_list:

        data_withoutVal, trend_withoutVal = getTrainData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                       lag=range(1,param[0]), lagVal=[], lagS= [param[1]], AvPr = param[2], AvPrVal = 0, test_size=test_size)

        if len(data_withoutVal) ==0:
            continue

        X_train = data_withoutVal.drop(["PriceWithOutTrend"], axis=1)
        y_train = data_withoutVal["PriceWithOutTrend"]

        xgbts = myxgBoost(7)

        err_par = performTimeSeries(X_train, y_train, xgbts, mean_absolute_percentage_error)
        if err > err_par:
            err = err_par
            param_best_withoutVal = param
            xgbts_best_withoutVal = deepcopy(xgbts)
            X_train_best_withoutVal = deepcopy(X_train)
            y_train_best_withoutVal = deepcopy(y_train)


    if len(data) != 0:

        y_past = []
        y_past2 = []

        y_true = []
        y_true2 = []
        for test_point in range(test_size):


            data, ytrue = getTestData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                           lag=range(1, param_best[0]), lagVal=range(1, param_best[1]), lagS=[param_best[2]],
                           test_size=test_size, trend = trend, AvPr = param_best[3], AvPrVal = param_best[4], y_past = y_past)


            X_test = data.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)

            predict = xgbts_best.predict(X_test)

            ypred = predict + X_test.Trend.values

            y_true.append(ytrue)
            y_past.append(inv_boxcox(ypred[0], lmbda))

            #без вылюты

            data_withoutVal, ytrue_withoutVal = getTestData(region, sproduct, datein=datetime.date(2010, 1, 1), dateout=datetime.date(2020, 1, 1),
                                      lag=range(1, param_best_withoutVal[0]), lagVal=[], lagS=[param_best_withoutVal[1]],
                                      test_size=test_size, trend=trend_withoutVal, AvPr = param_best_withoutVal[2], AvPrVal = 0, y_past=y_past2)

            X_test_withoutVal = data_withoutVal.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)

            predict_withoutVal = xgbts_best_withoutVal.predict(X_test_withoutVal)

            ypred_withoutVal = predict_withoutVal + X_test_withoutVal.Trend.values

            y_true2.append(ytrue_withoutVal)
            y_past2.append(inv_boxcox(ypred_withoutVal[0], lmbda))

        MAPE = mean_absolute_percentage_error(y_true, y_past)
        MAPE2 = mean_absolute_percentage_error(y_true2, y_past2)

        predict_X_train = xgbts_best.predict(X_train_best)
        predict_X_train_withoutVal = xgbts_best_withoutVal.predict(X_train_best_withoutVal)

        fig, axs = plt.subplots(2)


        axs[0].plot(np.concatenate((inv_boxcox(y_train_best + X_train_best.Trend.values, lmbda) , y_true)) , label = 'true')
        axs[0].plot(np.concatenate((inv_boxcox(predict_X_train + X_train_best.Trend.values, lmbda) , y_past)) , label = "predict")
        axs[0].axvspan(len(predict_X_train), len(predict_X_train)+len(y_past), alpha=0.5, color='lightgrey')
        axs[0].axis('tight')
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title("{}, mape {:.2f} \n {}".format(sproduct, MAPE, param_best))

        axs[1].plot(np.concatenate((inv_boxcox(y_train_best_withoutVal + X_train_best_withoutVal.Trend.values, lmbda) , y_true2)) , label = 'true')
        axs[1].plot(np.concatenate((inv_boxcox(predict_X_train_withoutVal + X_train_best_withoutVal.Trend.values, lmbda) , y_past2)) , label = "predict_withoutVal")
        axs[1].axvspan(len(predict_X_train_withoutVal), len(predict_X_train_withoutVal)+len(y_past2), alpha=0.5, color='lightgrey')
        axs[1].axis('tight')
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title("{}, mape {:.2f} \n {}".format(sproduct, MAPE2, param_best_withoutVal))

        plt.show()


    #res, pre_arima, train_arima = getPredictArima(region, sproduct)
    #cc = getPredictXgboost(res, pre_arima, train_arima)

connection.close()
