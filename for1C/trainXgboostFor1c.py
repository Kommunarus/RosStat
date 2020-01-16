import numpy as np
import sys
from dateutil.relativedelta import *
import datetime
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import xgboost as xgb
from scipy.special import boxcox, inv_boxcox
import argparse
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


def getTestData(datal, lag=0, lagVal =0, lagS =0, AvPr = 0, AvPrVal = 0, winWeather=0, AvMonth=0, y_past=[],  tr=1):

    global trend
    data = datal.copy()


    data["month"] = data.ymd.dt.month
    #data["year"] = data.ymd.dt.year
    #data['yearofchange'] = (data["ymd"] > datetime.datetime(2015,1,1))

    data = data.replace({"price": {0: np.nan}})
    data["price"].interpolate(inplace =True)

    data = data.fillna(method='bfill')


    data["price_boxcox"] = boxcox( data["price"], lmbda)

    if len(y_past) == 0:
        x = data['ymd'].map(datetime.datetime.toordinal)

        trend = np.poly1d(np.polyfit(x.values, data.price_boxcox.values, tr))

    data["Trend"] = trend(data['ymd'].map(datetime.datetime.toordinal).values)


    data["PriceWithOutTrend"] = data["price_boxcox"] - data["Trend"]

    #data["diff"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(1)
    #data["diff2"] = data["PriceWithOutTrend"] - data["PriceWithOutTrend"].shift(2)
    #data.loc[0, "diff"] = 0
    #data.loc[[0,1], "diff2"] = 0
    #data.loc[data['price'] == 0, "PriceWithOutTrend"]= 0.1 # если цена равна нулю, то поставим среднюю


    for i in range(1,lag+1):
        data["PriceWithOutTrend{}".format(i)] = data.PriceWithOutTrend.shift(i)
        #data["PriceWithOutTrend{}".format(i)].fillna(0, inplace=True)
    for i in [lagS]:
        if i!= 0:
            data["PriceWithOutTrendS{}".format(i)] = data.PriceWithOutTrend.shift(i)
            #data["PriceWithOutTrendS{}".format(i)].fillna(0, inplace=True)
    for i in range(1,lagVal+1):
        data["lagValute{}".format(i)] = data.ValueVal.shift(i)
        #data["lagValute{}".format(i)].fillna(0, inplace=True)

    # средние , максимум, минимум за квартал , полгода, год


    if AvMonth == 1:
        meanPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('mean')
        maxPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('max')
        minPrice = data[:-1].groupby('month')['PriceWithOutTrend'].aggregate('min')

        data.loc[:, 'meanPrice'] = [meanPrice[month] for month in data['month']]
        data.loc[:, 'maxPrice'] = [maxPrice[month] for month in data['month']]
        data.loc[:, 'minPrice'] = [minPrice[month] for month in data['month']]

    if AvPr!= 0:
        df = data.set_index('ymd').resample('MS', label='right').first()
        df1 = df['PriceWithOutTrend'].shift().rolling(min_periods=1, window=AvPr).agg(['mean', 'max', 'min']).reset_index()
        df1 = df1.add_suffix('_AvPr')
        data = pd.merge(data, df1, left_on=['ymd'], right_on=['ymd_AvPr'], how='left')
        #data["mean_AvPr"].fillna(0, inplace=True)
        data.drop(["ymd_AvPr"], axis=1, inplace=True)

        for i in range(1,lag+1):
            data["mean_AvPr{}".format(i)] = data.mean_AvPr.shift(i)
            data["mean_AvPr{}".format(i)].fillna(0, inplace=True)
            data["max_AvPr{}".format(i)] = data.max_AvPr.shift(i)
            data["max_AvPr{}".format(i)].fillna(0, inplace=True)
            data["min_AvPr{}".format(i)] = data.min_AvPr.shift(i)
            data["min_AvPr{}".format(i)].fillna(0, inplace=True)

    if AvPrVal != 0:
        df = data.set_index('ymd').resample('MS', label='right').first()
        df2 = df['ValueVal'].shift().rolling(min_periods=1, window=AvPrVal).agg(['mean', 'max', 'min']).reset_index()
        df2 = df2.add_suffix('_AvPrVal')
        data = pd.merge(data, df2, left_on=['ymd'], right_on=['ymd_AvPrVal'], how='left')
        #data["mean_AvPrVal"].fillna(0, inplace=True)
        data.drop(["ymd_AvPrVal"], axis=1, inplace=True)
        for i in range(1,lagVal+1):
            data["mean_AvPrVal{}".format(i)] = data.mean_AvPrVal.shift(i)
            data["mean_AvPrVal{}".format(i)].fillna(0, inplace=True)
            data["max_AvPrVal{}".format(i)] = data.max_AvPrVal.shift(i)
            data["max_AvPrVal{}".format(i)].fillna(0, inplace=True)
            data["min_AvPrVal{}".format(i)] = data.min_AvPrVal.shift(i)
            data["min_AvPrVal{}".format(i)].fillna(0, inplace=True)



    if winWeather != 0:
        dt_weather = pd.read_sql(
            'SELECT UTC as ymd, T, R FROM price.weather WHERE id = "{}" '.format(
                "/weather.php?id=37006"),
            con=connection)
        df = dt_weather.set_index('ymd').resample('D', label='right').agg({'T': 'mean', 'R': 'mean'})
        df['ymd'] = df.index
        df['indexmonth'] = df.apply(f_index_month, axis=1)
        df['indexmonth'] = df['indexmonth'].cumsum()
        df['cumT'] = df.groupby('indexmonth')['T'].cumsum()
        df['cumR'] = df.groupby('indexmonth')['R'].cumsum()

        df.loc[ df['indexmonth'] % 2 == 0, 'cumT'] = 0
        df.loc[ df['indexmonth'] % 2 == 0, 'cumR'] = 0


        df.drop('T', inplace = True, axis=1,)
        df.drop('R', inplace = True, axis=1,)
        df.drop('ymd', inplace = True, axis=1,)
        df.drop('indexmonth', inplace = True, axis=1,)
        for i in range(1,30,7):
            df["cumT{}".format(i)] = df.cumT.shift(i)
            df["cumR{}".format(i)] = df.cumR.shift(i)
        data = pd.merge(data, df, on=['ymd'], how='left', suffixes=('data', 'dt_weather'))
        #data = pd.merge(data, df, on=['ymd'], how='left', suffixes=('data', 'df2'))

    if lagVal == 0:
        data.drop(["ValueVal"], axis=1, inplace=True)
    data.drop(["price"], axis=1, inplace=True)
    data.drop(["price_boxcox"], axis=1, inplace=True)
    data.drop(["ymd"], axis=1, inplace=True)
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


def predict_dot(df_train, df_valute, param, xgbts, y_past, listValV, listDateV):

    df_train_copy = df_train.copy()
    past_ymd = df_train_copy.iloc[-1, :].ymd

    if len(y_past) != 0:
        for indd, past_value in enumerate(y_past):
            df_train_copy = df_train_copy.append({'price': past_value, 'ymd': past_ymd + relativedelta(months=+(1+indd))},
                                                 ignore_index=True)
        past_ymd = past_ymd + relativedelta(months=+(1 + len(y_past)))
        df_train_copy = df_train_copy.append({'price': 0, 'ymd': past_ymd}, ignore_index=True)


    data = pd.merge(df_train_copy, df_valute, left_on='ymd', right_on='dateCalendar', how ='left')

    dfv = pd.DataFrame({'ymd': listDateV, 'ValueVal': listValV})
    dfv.ymd = pd.to_datetime(dfv["ymd"])

    data = data.set_index('ymd')
    dfv = dfv.set_index('ymd')
    c = data.ValueVal
    c.update(dfv.ValueVal)
    data['ValueVal'] = c
    data = data.reset_index()

    #data['ValueVal'] = data['ymd'].map(dfv.set_index('ymd')['Val'])

    datatab = getTestData(data, **param)

    X_test = datatab.iloc[[-1]].drop(["PriceWithOutTrend"], axis=1)
    Trend  = X_test["Trend"]
    X_test.drop(["Trend"], axis=1, inplace=True)
    predict = xgbts.predict(X_test)

    ypred = predict[0] + Trend.values[0]

    return ypred

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='x1c2v3')
    parser.add_argument('--lag', type=int, default=0)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--lag_s', type=int, default=0)
    parser.add_argument('--AvPr', type=int, default=0)
    parser.add_argument('--AvPrVal', type=int, default=0)
    parser.add_argument('--lag_v', type=int, default=0)
    parser.add_argument('--AvMonth', type=int, default=0)
    parser.add_argument('--winWeather', type=int, default=0)
    parser.add_argument('--trend_param', type=int, default=0)
    parser.add_argument('--lmbda', type=float, default=1)
    parser.add_argument('--region', default='Российская Федерация')
    parser.add_argument('--product', default='Картофель')
    parser.add_argument('--nforecast', type=int, default=12)
    parser.add_argument('--datein', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(1,1,1))
    parser.add_argument('--dateout', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2999,1,1))
    parser.add_argument('--listValV', nargs='+', default=[])
    parser.add_argument('--listDateV', nargs='+', default=[])

    return parser

if __name__ == '__main__':
    connection = pymysql.connect(
        host='localhost',
        user='kommunar',
        password='123',
        db='price',
        charset='utf8mb4',
        cursorclass=DictCursor
    )

    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    id = namespace.id
    region = namespace.region
    product = namespace.product
    lmbda = namespace.lmbda
    nforecast = namespace.nforecast
    lag = namespace.lag
    max_depth = namespace.max_depth
    lag_s = namespace.lag_s
    AvPr = namespace.AvPr
    AvPrVal = namespace.AvPrVal
    lag_v = namespace.lag_v
    AvMonth = namespace.AvMonth
    winWeather = namespace.winWeather
    trend_param = namespace.trend_param
    datein = namespace.datein
    dateout = namespace.dateout
    listValV  = list(map(lambda x: float(x), namespace.listValV))
    listDateV = list(map(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), namespace.listDateV))

    '''id = 'cfed2'
    max_depth =8
    lag =20
    lag_s =0
    AvPr =0
    lag_v =3
    trend_param =1
    AvMonth =0
    winWeather =1
    region ="Краснодарский край"
    product ="Огурцы"
    lmbda =0.25
    nforecast =36
    datein ='2011-02-01'
    dateout ='2018-08-01'
    listValV  = [70, 80, 90, 100, 150, 150, 150, 160, 170, 170, 170, 180, 180, 170, 180, 170, 180, 170, 180, 170]
    listDateV = list(map(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), ['2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01', '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01' ]))
    '''


    df_train = pd.read_sql(
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd >= "{}" and ymd <= "{}"'.format(region, product, datein, dateout),
        con=connection)

    df_train["ymd"]   = pd.to_datetime(df_train["ymd"])

    df_valute = pd.read_sql(
        'SELECT ValueVal, dateCalendar FROM price.valuta WHERE CharCode = "{}"  and dateCalendar >= "{}" and dateCalendar <= "{}"'.format("USD", datein, dateout),
        con=connection)

    dfv = df_valute.copy().set_index('dateCalendar')
    dfv.index = pd.to_datetime(dfv.index)
    df_valute = dfv.resample('MS', label='right').agg({'ValueVal': 'median'}).reset_index()

    data = pd.merge(df_train, df_valute, left_on='ymd', right_on='dateCalendar', how ='left')

    param = {'lag':lag, 'lagVal':lag_v, 'lagS':lag_s,  'AvPr':AvPr, 'AvPrVal':AvPrVal, 'winWeather':winWeather, 'AvMonth':AvMonth, 'tr':trend_param}
    datatab= getTestData(data, **param)

    datatab.drop(["Trend"], axis=1, inplace=True)
    X_train = datatab.drop(["PriceWithOutTrend"], axis=1)
    y_train = datatab["PriceWithOutTrend"]

    xgbts = myxgBoost(max_depth)

    err_par = performTimeSeriesCV(X_train, y_train, 5, xgbts, mape)
    print(err_par)
    y_past = []
    y_true = []
    for test_point in range(nforecast):
        ypred = predict_dot(df_train, df_valute, param, xgbts, y_past, listValV, listDateV)
        print(ypred)
        y_past.append(inv_boxcox(ypred, lmbda))

        query = "INSERT INTO price.sarima_predict_1c(id, ymd, mean, up, botton) " \
                "VALUES(%s,%s,%s,%s,%s)"
        args = (id,
                test_point,
                float(inv_boxcox(ypred, lmbda)),
                0,
                0)

        cursor = connection.cursor()
        try:
            cursor.execute(query, args)
        except:
            print('error')
        connection.commit()




    connection.close()
