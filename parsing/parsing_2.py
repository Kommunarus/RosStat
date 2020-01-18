import urllib.request as ur
import requests
import json
import pandas as pd
import datetime
from pandas.io.json import json_normalize

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

'''query = "DELETE FROM price.tab"
cursor = connection.cursor()
cursor.execute(query)
connection.commit()'''

# https://www.fedstat.ru/indicator/31454

params = {  'id':'31454',
            'lineObjectIds':['57831', '57938'],
            'columnObjectIds':['3', '33560'],
            'selectedFilterIds':['0_31454','3_2011','3_2012','3_2013','3_2014','3_2015','3_2016','30611_950351','33560_1540228','33560_1540229','33560_1540230','33560_1540233','33560_1540234','33560_1540235','33560_1540236','33560_1540272','33560_1540273','33560_1540276','33560_1540282','33560_1540283','57831_1688487','57831_1688488','57831_1688489','57831_1688490','57831_1688491','57831_1688492','57831_1688493','57831_1688494','57831_1688495','57831_1688496','57831_1688497','57831_1688498','57831_1688499','57831_1688500','57831_1688501','57831_1688502','57831_1688503','57831_1688504','57831_1688505','57831_1688506','57831_1688507','57831_1688508','57831_1688509','57831_1688510','57831_1688511','57831_1688512','57831_1688513','57831_1688514','57831_1688515','57831_1688516','57831_1688517','57831_1688518','57831_1688520','57831_1688521','57831_1688522','57831_1688523','57831_1688524','57831_1688525','57831_1688526','57831_1688527','57831_1688528','57831_1688529','57831_1688530','57831_1688531','57831_1688532','57831_1688533','57831_1688534','57831_1688535','57831_1688536','57831_1688537','57831_1688538','57831_1688539','57831_1688540','57831_1688541','57831_1688542','57831_1688543','57831_1688544','57831_1688545','57831_1688546','57831_1688547','57831_1688548','57831_1688549','57831_1688550','57831_1688551','57831_1688552','57831_1688553','57831_1688554','57831_1688555','57831_1688556','57831_1688557','57831_1688558','57831_1688559','57831_1688560','57831_1688561','57831_1688562','57831_1688563','57831_1688564','57831_1688565','57831_1688566','57831_1688567','57831_1688568','57831_1688571','57831_1688572','57831_1688573','57831_1688574','57831_1688575','57831_1688576','57831_1688577','57831_1688578','57831_1688579','57831_1688581','57831_1688582','57831_1688583','57831_1688584','57831_1688585','57831_1688586','57831_1688587','57831_1692937','57831_1692938','57831_1692939','57831_1692940','57831_1695534','57831_1697988','57831_1709529','57831_1709530','57831_1709531','57831_1709532','57831_1709533','57831_1709534','57831_1709535','57831_1709536','57831_1709537','57831_1709538','57831_1709539','57831_1710310','57938_1743376','57938_1743377','57938_1743378','57938_1743379','57938_1743380','57938_1743381','57938_1743382','57938_1743383','57938_1743384','57938_1743385','57938_1743386','57938_1743387','57938_1743388','57938_1743389','57938_1743390','57938_1743391','57938_1743392','57938_1743393','57938_1743394','57938_1743395','57938_1743396','57938_1743397','57938_1743398','57938_1743399','57938_1743400','57938_1743401','57938_1743402','57938_1743403','57938_1743404','57938_1743405','57938_1743406','57938_1743407','57938_1743408','57938_1743409','57938_1743410','57938_1743411','57938_1743412','57938_1743413','57938_1743414','57938_1743415','57938_1743416','57938_1743417','57938_1743418','57938_1743419','57938_1743420','57938_1743421','57938_1743422','57938_1743423','57938_1743424','57938_1743425','57938_1743426','57938_1743427','57938_1743428','57938_1743429','57938_1743430','57938_1743431','57938_1743432','57938_1743433','57938_1743434','57938_1743435','57938_1743436','57938_1743437','57938_1743438','57938_1743439','57938_1743440','57938_1743441','57938_1743442','57938_1743443','57938_1743444','57938_1743445','57938_1743446','57938_1743447','57938_1743448','57938_1743449','57938_1743450']}
response = requests.get('https://www.fedstat.ru/indicator/dataGrid.do', params = params)

ss = response.json()


periods = {'1540283':'1',
           '1540282':'2',
           '1540236':'3',
           '1540229':'4',
           '1540235':'5',
           '1540234':'6',
           '1540233':'7',
           '1540228':'8',
           '1540276':'9',
           '1540273':'10',
           '1540272':'11',
           '1540230':'12',
           }

columns = ['region', 'product', 'data', 'price']



for i in ss['results']:
    reg = ''
    prod =''
    for k, v in i.items():
        if k == 'dim57831':
            reg = str(v)
        elif k == 'dim57938':
            prod = str(v)
        else:
            x = k.split('_')
            y = x[0].replace('dim','')
            m = periods[x[1]]
            query = "INSERT INTO price.tab(region, products, ymd, price, type) " \
                    "VALUES(%s,%s,%s,%s)"
            args = (reg, prod, datetime.datetime(int(y),int(m),1).strftime('%Y-%m-%d %H:%M:%S'), float(v.replace(',','.')), 'rosstat')
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()





