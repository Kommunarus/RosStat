import requests
import xml.etree.ElementTree as ET

from lxml import html

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

params = {'id':'ru', 'region':23}
r = requests.get('http://www.pogodaiklimat.ru/archive.php', params=params)
r.encoding = r.apparent_encoding

tree = html.fromstring(r.text)

list = tree.xpath('//ul[@class = "big-blue-billet__list"]')[0].findall("li")
for row in list:
    li = row.getchildren()[0]
    query = "INSERT INTO price.city_id_weather(id, region, city) " \
            "VALUES(%s,%s,%s)"
    args = (li.attrib['href'], 'Краснодарский край', li.text_content())
    cursor = connection.cursor()
    cursor.execute(query, args)
    connection.commit()

