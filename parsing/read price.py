from openpyxl import load_workbook
import datetime
import os
import tempfile
import zipfile
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


def fix_xlsx(in_file):
    zin = zipfile.ZipFile(in_file, 'r')
    if 'xl/SharedStrings.xml' in zin.namelist():
        tmpfd, tmp = tempfile.mkstemp(dir=os.path.dirname(in_file))
        os.close(tmpfd)

        with zipfile.ZipFile(tmp, 'w') as zout:
            for item in zin.infolist():
                if item.filename == 'xl/SharedStrings.xml':
                    zout.writestr('xl/sharedStrings.xml', zin.read(item.filename))
                else:
                    zout.writestr(item, zin.read(item.filename))

        zin.close()
        os.remove(in_file)
        os.rename(tmp, in_file)

#fix_xlsx("../price.xlsx")

wb = load_workbook(filename="../price.xlsx", read_only=True)
ws = wb.worksheets[0]

count = 0

for row in ws.rows:
    count += 1
    #if count >= 20:
    #    break
    for pro in range(0,4):
        if count >= 2:
            region = row[0].value
            ymd    = row[5].value
            if pro == 0:
                product = "Гречиха"
                price = float(row[1].value)
            if pro == 1:
                product = "Говядина (убойны вес)"
                price = float(row[2].value)
            if pro == 2:
                product = "Молоко сырое"
                price = float(row[3].value)
            if pro == 3:
                product = "Свинина (убойны вес)"
                price = float(row[4].value)
            #print('{} \t {} \t {} \t {}'.format(region, ymd, product, price))
            query = "INSERT INTO price.tab_price(region, product, ymd, price, type_price) " \
                    "VALUES(%s,%s,%s,%s,%s)"
            args = (region, product, ymd.strftime('%Y-%m-%d %H:%M:%S'), price, 'smpb')
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()

connection.close()