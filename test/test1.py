'''
Created on 2017. 11. 13.

@author: acorn
'''

from xml.etree.ElementTree import parse
from pandas import DataFrame

tree = parse('drug_201610.xml')
myroot = tree.getroot() # 최상위 엘리먼트 취득

# print( type(myroot))

# <Row>라는 엘리먼트에 업체 1군데의 정보가 들어 있다.
# alldrugs : 모든 약에 대한 정보
alldrugs = myroot.findall('Row')
# print( 'findall는 일치하는 모든 태그를 리스트로 반환한다.' )
# print( alldrugs )
# print()

totallist = [] #약의 정보들을 저장할 리스트
for onestore in alldrugs:
    childs = onestore.getchildren()
    sublist = [] # 1개의 음식점 정보
    for onedata in childs:
        # 콤마와 쌍따옴표는 빈 문자열로 처리하도록 한다.
        mydata = onedata.text.replace(',', '')
        mydata = mydata.replace('"', '')
        sublist.append( mydata )
    # print('-------------------')
    totallist.append(sublist)

#mycolumn = ['제품명', '표준코드', '품목명', '품목기준코드', '회수의무자', '회수일자', '제조번호']
mycolumn = ['제품명', '표준코드', '품목명', '품목기준코드', '회수의무자', '회수일자', '제조번호', '제조일자', '포장단위', '회수사유', '위험등급', '등록일자']
# myframe은 약 정보를 담고 있는 DataFrame이다.
myframe = DataFrame(totallist, columns=mycolumn)
# print( type(myframe) ) # <class 'pandas.core.frame.DataFrame'>
# print()
# print(myframe)
# print()

# myframe.to_csv('datadrug.csv', encoding='EUC-KR')
# print( '작업 완료' )

# sqlite3를 사용하기 위하여 임포트
import sqlite3
import pandas as pd

conn = sqlite3.connect('drug.db')

# cursor(커서) : 실제 db에 접속해서 무엇인가를 요청하는 객체
mycursor = conn.cursor()

try:
    mycursor.execute("drop table drugkorea")
except sqlite3.OperationalError:
    print('테이블이 존재하지 않습니다.')

mycursor.execute('''CREATE TABLE drugkorea
             (name text, stdcode text, 
             product_name text, product_stdcode text, 
             return_company text, return_date text, 
             product_num text, product_date text,
             unit text, reason text, danger_grade text, reg_date text)''')


for onedata in range(len(myframe)):
    imsi = myframe.ix[onedata]
    name = imsi['제품명']
    stdcode = imsi['표준코드']
    product_name = imsi['품목명']
    product_stdcode = imsi['품목기준코드']
    return_company = imsi['회수의무자']
    return_date = imsi['회수일자']
    product_num = imsi['제조번호']
    product_date = imsi['제조일자']
    unit = imsi['포장단위']
    reason = imsi['회수사유']
    danger_grade = imsi['위험등급']
    reg_date = imsi['등록일자']
    
    sql = "insert into drugkorea values('" + name + "', '" +  stdcode + "', '" + product_name + "', '" + product_stdcode + "', '" + return_company + "','" + return_date + "', '" + product_num + "', '" + product_date + "', '" + unit + "', '" + reason + "', '" + danger_grade + "', '" + reg_date + "')"
    
    print( sql )
    mycursor.execute( sql )

conn.commit()

# finddata = '신세계'
# sql = "select * from inchon where name like '%" + finddata + "%'"
# print('업소명에 [' + finddata + ']라는 글자가 포함된 가게')
# for row in mycursor.execute( sql ):
#     # print (type(row)) # <class 'tuple'>
#     print (row)
# print()

# finddata = '백숙'
# sql = "select * from inchon where maindish like '%" + finddata + "%'"
# print('이름에 [' + finddata + ']라는 글자가 포함된 가게')
# for row in mycursor.execute( sql ):
#     print (row)
# print()

sql = "select * from drugkorea order by name asc"
print('약 이름순으로 정렬합니다.')
for row in mycursor.execute( sql ):
    print (row)
print()

conn.close()
print()
print('작업 완료^^')