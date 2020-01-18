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
    if count >= 5:
        region = row[0].value
        ymd    = datetime.datetime.strptime(row[3].value, '%d.%m.%Y %H:%M:%S')
        product = row[4].value
        price = float(row[6].value)
        #print('{} \t {} \t {} \t {}'.format(region, ymd, product, price))
        query = "INSERT INTO price.tab(region, products, ymd, price, type) " \
                "VALUES(%s,%s,%s,%s,%s)"
        args = (region, product, ymd.strftime('%Y-%m-%d %H:%M:%S'), price, 'smpb')
        cursor = connection.cursor()
        cursor.execute(query, args)
        connection.commit()

connection.close()