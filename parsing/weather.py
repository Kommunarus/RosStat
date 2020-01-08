import requests
import xml.etree.ElementTree as ET
import pandas as pd
from lxml import html
import datetime

import pymysql
from pymysql.cursors import DictCursor

connection = pymysql.connect(
    host='localhost',
    user='kommunar',
    password='123',
    db='price',
    charset='utf8mb4',
    cursorclass=DictCursor
)



region = 'Краснодарский край'

df = pd.read_sql(
    'SELECT id  FROM price.city_id_weather WHERE region = "{}" '.format( region),
    con=connection)

for rowdf in df.iterrows():

    for year in range(2011,2020):
        for month in range(1,13):

            params = {'bday':1, 'fday':31, 'amonth':month, 'ayear':year, 'bot':2}
            r = requests.get('http://www.pogodaiklimat.ru{}'.format(rowdf[1].id), params=params)
            r.encoding = r.apparent_encoding

            tree = html.fromstring(r.text)

            list_1 = tree.xpath('//div[@class = "archive-table-left-column"]/table')[0].findall("tr")
            data1 = list()
            for row in list_1:
                data1.append([c.text_content()  for c in row.getchildren()])

            list_2 = tree.xpath('//div[@class = "archive-table-wrap"]/table')[0].findall("tr")
            data2 = list()
            for row in list_2:
                data2.append([c.text_content()  for c in row.getchildren()])


            for row in zip(data1, data2):
                try:
                    datatime = datetime.datetime(int(year),int(row[0][1].split('.')[1]),int(row[0][1].split('.')[0]), int(row[0][0])).strftime('%Y-%m-%d %H:%M:%S')

                    T = str(row[1][5])
                    R = str(row[1][15])
                    S = str(row[1][17])

                    query = "INSERT INTO price.weather(id, UTC, T, R, S) " \
                            "VALUES(%s,%s,%s,%s,%s)"
                    args = (rowdf[1].id, datatime, 0 if T == '' else float(T), 0 if R == '' else float(R), 0 if S == '' else  float(S))
                    cursor = connection.cursor()
                    cursor.execute(query, args)
                    connection.commit()
                except:
                    pass
