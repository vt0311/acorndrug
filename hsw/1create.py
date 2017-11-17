'''
Created on 2017. 10. 23.

@author: acorn
'''
#import sqlite3
import cx_Oracle

conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')

cursor = conn.cursor()

cursor.execute('''CREATE TABLE drug_safety
(제품명 varchar2(100) , 표준코드 varchar2(100)  , 품목명 varchar2(100) , 품목기준코드 varchar2(100) , 회수의무자 varchar2(100) , 회수일자 varchar2(100) , 제조번호 varchar2(100) , 포장단위 varchar2(100) , 회수사유 varchar2(100) , 위험등급 varchar2(100)  , 등록일자 varchar2(100)  ) ''')


cursor.close()

conn.close()

