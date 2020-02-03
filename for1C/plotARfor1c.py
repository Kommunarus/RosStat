import argparse
import numpy as np
import sys
import pymysql
from pymysql.cursors import DictCursor
import pandas as pd
import statsmodels.api as sm
import datetime

from scipy.special import boxcox, inv_boxcox


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='x1c2v3')
    parser.add_argument('--d', type = int, default=0)
    parser.add_argument('--TypeGraph', default='ACF')
    parser.add_argument('--region', default='Российская Федерация')
    parser.add_argument('--product', default='Картофель')
    parser.add_argument('--datein', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(1,1,1))
    parser.add_argument('--dateout', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default=datetime.date(2999,1,1))

    return parser


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.asarray(diff)


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
    d = namespace.d
    TypeGraph = namespace.TypeGraph
    region = namespace.region
    product = namespace.product
    datein = namespace.datein
    dateout = namespace.dateout

    '''id = '22630333-ef27-4228-b338-f1b263dca013'
    d = 1
    TypeGraph = ACF
    region = "Республика Мордовия"
    product = "Яйца куриные"
    datein = 2008-02-01
    dateout = 2015-03-01'''


    df = pd.read_sql(
        'SELECT ymd, price FROM price.tab_price WHERE region = "{}" and product="{}" and ymd > "{}" and ymd < "{}"'.format(region, product, datein, dateout),
        con=connection)
    dty = df.price.values

    for n_d in range(d):
        dty = difference(dty)

    if TypeGraph == 'ACF':
        acf = sm.tsa.stattools.acf(dty)
    else:
        acf = sm.tsa.stattools.pacf(dty)


    query = "INSERT INTO price.acf(id, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40) " \
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    args = (id,
            float(acf[0]),
            float(acf[1]),
            float(acf[2]),
            float(acf[3]),
            float(acf[4]),
            float(acf[5]),
            float(acf[6]),
            float(acf[7]),
            float(acf[8]),
            float(acf[9]),
            float(acf[10]),
            float(acf[11]),
            float(acf[12]),
            float(acf[13]),
            float(acf[14]),
            float(acf[15]),
            float(acf[16]),
            float(acf[17]),
            float(acf[18]),
            float(acf[19]),
            float(acf[20]),
            float(acf[21]),
            float(acf[22]),
            float(acf[23]),
            float(acf[24]),
            float(acf[25]),
            float(acf[26]),
            float(acf[27]),
            float(acf[28]),
            float(acf[29]),
            float(acf[30]),
            float(acf[31]),
            float(acf[32]),
            float(acf[33]),
            float(acf[34]),
            float(acf[35]),
            float(acf[36]),
            float(acf[37]),
            float(acf[38]),
            float(acf[39]),
            float(acf[40])
            )

    print(args)
    cursor = connection.cursor()
    cursor.execute(query, args)
    connection.commit()



    connection.close()
