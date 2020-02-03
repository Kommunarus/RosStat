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
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--q', type=int, default=2)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--sd', type=int, default=1)
    parser.add_argument('--sq', type=int, default=1)
    parser.add_argument('--ss', type=int, default=12)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--region', default='Российская Федерация')
    parser.add_argument('--product', default='Огурцы')
    parser.add_argument('--nforecast', type=int, default=20)
    parser.add_argument('--timeforcast', type=str, default='m')
    parser.add_argument('--datein', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2013,1,1))
    parser.add_argument('--dateout', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2018,1,1))

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
    timeforcast = namespace.timeforcast


    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab_price WHERE region = "{}" and product="{}" and ymd > "{}" and ymd < "{}" '
        'ORDER BY ymd  '.format(region, product, datein, dateout),
        con=connection)
    df['price'] = boxcox(df['price'], lmbda)
    df['ymd'] = pd.to_datetime(df['ymd'])
    df = df.set_index('ymd')





    mod = sm.tsa.statespace.SARIMAX(df['price'], order=(p, d, q),  seasonal_order=(sp, sd, sq, ss))
    res = mod.fit(disp=False)

    param = {'p':p, 'd':d, 'q':q, 'sp':sp, 'sd':sd, 'sq':sq, 'ss':ss}
    datend = df['price'].index[-1] + (relativedelta(months=+(nforecast))) if timeforcast == "m" else (dateout + relativedelta(weeks=+(nforecast)))
    predict = res.get_prediction(start =  df['price'].index[0], end= datend)

    p_main = inv_boxcox(predict.predicted_mean, lmbda)

    #predict_ci1 = inv_boxcox(predict.conf_int(alpha=0.5)[:,0], lmbda)
    #predict_ci2 = inv_boxcox(predict.conf_int(alpha=0.5)[:,1], lmbda)
    print(len(p_main))
    date_predict = datetime.datetime.now()
    for index in range(len(p_main)):
        query = "INSERT INTO price.predict_1c(id, mean, up, botton, ymd_date, comment_text, region, product,model, date_predict) " \
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        args = (id,
                float(p_main[index]),
                0, #if np.isnan(predict_ci2[index])  else float(predict_ci2[index]),
                0, # if np.isnan(predict_ci1[index]) else float(predict_ci1[index]),
                predict.predicted_mean.index[index].strftime('%Y-%m-%d %H:%M:%S'),
                ' '.join(['{}:{}'.format(k, i) for (k, i) in param.items()]),
                region,
                product,
                'sarima',
                date_predict)

        print(args)
        cursor = connection.cursor()
        cursor.execute(query, args)
        connection.commit()

    connection.close()
