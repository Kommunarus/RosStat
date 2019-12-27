import requests
import datetime
import calendar
import xml.etree.ElementTree as ET

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


for y in range(1999,2020):
    for m in range(1,13):
        params = {'date_req':'01.{:02}.{}'.format(m,y)}
        response = requests.get('http://www.cbr.ru/scripts/XML_daily.asp', params = params)

        root = ET.fromstring(response.text)

        for idx,Valute_element in enumerate(root.findall('Valute')):
            NumCode = Valute_element.find('NumCode').text
            CharCode = Valute_element.find('CharCode').text
            Nominal = Valute_element.find('Nominal').text
            Name = Valute_element.find('Name').text
            Value2 = Valute_element.find('Value').text


            query = "INSERT INTO price.valuta(NumCode, CharCode, dateCalendar, ValueVal, Nominal) " \
                    "VALUES(%s,%s,%s,%s,%s)"
            args = (NumCode, CharCode, datetime.datetime(int(y),int(m),1).strftime('%Y-%m-%d %H:%M:%S'), float(Value2.replace(',','.')), int(Nominal))
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()

connection.close()