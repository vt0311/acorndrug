'''
Created on 2017. 12. 08

@author: hsw
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
#import sqlite3
import cx_Oracle as oracle
import pandas as pd

conn = oracle.connect('scott/tiger@localhost:1521/xe')

# cursor(커서) : 실제 db에 접속해서 무엇인가를 요청하는 객체
mycursor = conn.cursor()

#try:
#    mycursor.execute("drop table drugkorea")
#except oracle.OperationalError:
#    print('테이블이 존재하지 않습니다.')

#mycursor.execute('''CREATE TABLE drugkorea
#             (name varchar2(80), stdcode varchar2(80), 
#             product_name varchar2(80), product_stdcode varchar2(80), 
#             return_company varchar2(80), return_date varchar2(80), 
#             product_num varchar2(80), product_date varchar2(80),
#             unit varchar2(80), reason varchar2(500), danger_grade varchar2(80), reg_date varchar2(80))''')


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


sql = "select * from drugkorea order by name asc"
print('약 이름순으로 정렬합니다.')
for row in mycursor.execute( sql ):
    print (row)
print()

conn.close()
print()
print('작업 완료')