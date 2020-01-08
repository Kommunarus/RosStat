import requests
import xml.etree.ElementTree as ET

from lxml import html



params = {'id':34720, 'bday':1, 'fday':31, 'amonth':1, 'ayear':2014, 'bot':2}
r = requests.get('http://www.pogodaiklimat.ru/weather.php', params=params)
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
    print(row)
