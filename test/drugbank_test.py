'''
Created on 2017. 12.08

@author: hsw

Drug Bank test 
'''

from xml.etree.ElementTree import parse
from pandas import DataFrame

#tree = parse('C:/PARSING/parsed1_N_test1.xml')
tree = parse('parsed1_N_test1.xml')
myroot = tree.getroot() # 최상위 엘리먼트 취득

# print( type(myroot))

# <product>라는 엘리먼트에 약물 1제품의 정보가 들어 있다.
# alldrugs : 모든 약에 대한 정보
alldrugs = myroot.findall('product')
# print( 'findall는 일치하는 모든 태그를 리스트로 반환한다.' )
# print( alldrugs )
# print()

totallist = [] #약의 정보들을 저장할 리스트
for onedrug in alldrugs:
    childs = onedrug.getchildren()
    sublist = [] # 1개의 약 정보
    for onedata in childs:
        # 콤마와 쌍따옴표는 빈 문자열로 처리하도록 한다.
        mydata = onedata.text.replace(',', '')
        mydata = mydata.replace('"', '')
        sublist.append( mydata )
    # print('-------------------')
    totallist.append(sublist)


mycolumn = ['name', 'labeller', 'ndc-id', 'ndc-product-code', 'dpd-id', 'started-marketing-on', 'ended-marketing-on', 'dosage-form', 'strength', 'route', 'fda-application-number', 'generic', 'over-the-counter', 'approved', 'country', 'source']

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

conn = sqlite3.connect('drugbank.db')

# cursor(커서) : 실제 db에 접속해서 무엇인가를 요청하는 객체
mycursor = conn.cursor()

try:
    mycursor.execute("drop table drugbanktest")
except sqlite3.OperationalError:
    print('테이블이 존재하지 않습니다.')

mycursor.execute('''CREATE TABLE drugbanktest
             (name text primary key, labeller text, ndc_id text, ndc_product_code text
             , dpd_id text, started_marketing_on text, ended_marketing_on text 
             , dosage_form text, strength text, route text
             , fda_application_number text, generic text
             , over_the_counter text
             , approved text, country text
             , source text)''')


for onedata in range(len(myframe)):
    imsi = myframe.ix[onedata]
    name = imsi['name']
    labeller = imsi['labeller']
    ndc_id = imsi['ndc-id']
    ndc_product_code = imsi['ndc-product-code']
    dpd_id = imsi['dpd-id']
    started_marketing_on = imsi['started-marketing-on']
    ended_marketing_on = imsi['ended-marketing-on']
    dosage_form = imsi['dosage-form']
    strength = imsi['strength']
    route = imsi['route']
    fda_application_number = imsi['fda-application-number']
    generic = imsi['generic']
    over_the_counter = imsi['over-the-counter']
    ended_marketing_on = imsi['ended-marketing-on']
    approved = imsi['approved']
    country = imsi['country']
    source = imsi['source']
    
    sql = "insert into drugbanktest values('" + name + "', '" 
    +  labeller + "', '" + ndc_id + "', '" 
    + ndc_product_code + "', '" + dpd_id + "','" 
    + started_marketing_on + " ', ' " 
    + ended_marketing_on + " ', ' " 
    + dosage_form + " ', ' " 
    + strength + " ', ' " 
    + route + " ', ' " 
    + fda_application_number + " ', ' " 
    + generic + " ', ' " 
    + over_the_counter + " ', ' " 
    + approved + " ', ' " 
    + country + " ', ' " 
    + source + "')"
    
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

sql = "select * from drugbanktest order by name asc"
print('약 이름순으로 정렬합니다.')
for row in mycursor.execute( sql ):
    print (row)
print()

conn.close()
print()
print('작업 완료')