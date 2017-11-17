'''
Created on 2017. 11. 17

@author: hsw
'''
from xml.etree.ElementTree import parse
from pandas import DataFrame
import cx_Oracle 


tree = parse('drug_201610.xml')
myroot = tree.getroot() # 최상위 엘리먼트 취들
            
alldrugs = myroot.findall('Row')
  

totallist = [] #정보들을 저장할 리스트
for onedrug in alldrugs:
    childs = onedrug.getchildren()
    sublist = [] # 1개의  정보
    for onedata in childs:
        # 콤마와 쌍따옴표는 빈 문자열로 처리하도록 한다.
        mydata = onedata.text.replace(',', '')
        mydata = mydata.replace('"', '')
        sublist.append( mydata )
    # print('-------------------')
    totallist.append(sublist)

mycolumn = ['제품명', '표준코드', '품목명' , '품목기준코드' , '회수의무자' , '회수일자' , '제조번호' , '제조일자', '포장단위' , '회수사유' , '위험등급'  , '등록일자' ]

myframe = DataFrame(totallist, columns=mycolumn)

        
            
conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')

cursor = conn.cursor()


for onedata in range(len(myframe)):
    imsi = myframe.ix[onedata]
    name = imsi['제품명']
    stdcode = imsi['표준코드']
    productname = imsi['품목명']
    productstdcode = imsi['품목기준코드']
    returncompany = imsi['회수의무자']
    returndate = imsi['회수일자']
    productnum = imsi['제조번호']
    productdate = imsi['제조일자']
    packageunit = imsi['포장단위']
    returnreason = imsi['회수사유']
    dangergrade = imsi['위험등급']
    regdate = imsi['등록일자']
    
    sql = "insert into DRUG_SAFETY values('" + name + "', '" +  stdcode + "', '" + productname + "', '" + productstdcode + "', '" + returncompany + "','" + returndate + " ', ' " + productnum + " ','" + productdate + " ','" + packageunit + " ','" + returnreason + " ','" + dangergrade + " ','" + regdate + " ')"

    cursor.execute( sql )

conn.commit()


cursor.close()

conn.close()




