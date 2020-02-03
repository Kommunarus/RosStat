import argparse
from itertools import product
import sys
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import statsmodels.api as sm
import datetime
import warnings

warnings.simplefilter('ignore')


connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)

from scipy.special import boxcox, inv_boxcox

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='x1c2v3')
    parser.add_argument('--p', type=int, default=0)
    parser.add_argument('--d', type=int, default=0)
    parser.add_argument('--q', type=int, default=0)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--sd', type=int, default=0)
    parser.add_argument('--sq', type=int, default=0)
    parser.add_argument('--ss', type=int, default=0)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--region', default='Российская Федерация')
    parser.add_argument('--product', default='Картофель')
    parser.add_argument('--season', type=int, default=12)
    parser.add_argument('--datein', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(1,1,1))
    parser.add_argument('--dateout', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2999,1,1))

    return parser


def getPredict(namespace):

    id = namespace.id
    region = namespace.region
    sproduct = namespace.product
    lmbda = namespace.lmbda
    season = namespace.season
    p = namespace.p
    d = namespace.d
    q = namespace.q
    sp = namespace.sp
    sd = namespace.sd
    sq = namespace.sq
    ss = namespace.ss
    datein = namespace.datein
    dateout = namespace.dateout


    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab_price WHERE region = "{}" and product="{}" and ymd > "{}" and ymd < "{}"'.format(region, sproduct, datein, dateout),
        con=connection)


    dta = df.price.values

    train = boxcox(dta, lmbda)



    n_p = range(0,p)
    n_d = range(0,d)
    n_q = range(0,q)
    n_sp = range(0,sp)
    n_sd = range(0,sd)
    n_sq = range(0,sq)
    n_ss = range(0,ss)
    print(n_p)
    parameters = product(n_p, n_d, n_q, n_sp, n_sd, n_sq, n_ss)
    parameters_list = list(parameters)

    best_aic = float("inf")
    best_bic = float("inf")
    best_hqic = float("inf")

    niter = 0
    for param in parameters_list:
        niter += 1
        try:
            model = sm.tsa.statespace.SARIMAX(train, order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], int(param[6] * season)),
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



    return [best_param_aic, best_param_bic, best_param_hqic]

if __name__ == '__main__':


    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    cc = getPredict(namespace)

    for index, coef in enumerate(cc):
        query = "INSERT INTO price.model(id, region, product, criterion, p, d, q, sp, sd, sq, ss) " \
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        if index == 0:
            text = 'aic'
        if index == 1:
            text = 'bic'
        if index == 2:
            text = 'hqic'

        args = (namespace.id, namespace.region, namespace.product, text, int(coef[0]),int(coef[1]),int(coef[2]), int(coef[3]),int(coef[4]),int(coef[5]), int(namespace.season*coef[6]) )
        cursor = connection.cursor()
        cursor.execute(query, args)
        connection.commit()

connection.close()
