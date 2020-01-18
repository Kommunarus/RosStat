import argparse
import numpy as np
import sys
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import statsmodels.api as sm
import datetime
from dateutil.relativedelta import *


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
    parser.add_argument('--nforecast', type=int, default=12)
    parser.add_argument('--datein', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(1,1,1))
    parser.add_argument('--dateout', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2999,1,1))

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
        'SELECT ymd, price FROM price.tab WHERE region = "{}" and products="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)
    dty = df.price.values
    dtx = df.ymd.values
    train = boxcox(dty, lmbda)





    mod = sm.tsa.statespace.SARIMAX(train, order=(p, d, q),
                                    seasonal_order=(sp, sd, sq, ss))
    res = mod.fit(disp=False)

    param = {'p':p, 'd':d, 'q':q, 'sp':sp, 'sd':sd, 'sq':sq, 'ss':ss}
    predict = res.get_prediction(end=mod.nobs + nforecast)

    p_main = inv_boxcox(predict.predicted_mean, lmbda)

    predict_ci1 = inv_boxcox(predict.conf_int(alpha=0.5)[:,0], lmbda)
    predict_ci2 = inv_boxcox(predict.conf_int(alpha=0.5)[:,1], lmbda)
    print(len(p_main))
    for ii in range(nforecast):
        query = "INSERT INTO price.sarima_predict_1c(id, ymd, mean, up, botton, ymd_date, comment_text, region, product,model) " \
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        index = ii+1
        args = (id,
                nforecast-ii,
                float(p_main[-index]),
                0 if np.isnan(predict_ci2[-index])  else float(predict_ci2[-index]),
                0 if np.isnan(predict_ci1[-index]) else float(predict_ci1[-index]),
                dateout + relativedelta(months=+(nforecast-ii)),
                ' '.join(['{}:{}'.format(k, i) for (k, i) in param.items()]),
                region,
                product,
                'sarima')

        print(args)
        cursor = connection.cursor()
        try:
            cursor.execute(query, args)
        except:
            print('error')
        connection.commit()

    connection.close()
