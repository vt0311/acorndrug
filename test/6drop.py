'''
Created on 2017. 10. 23.

@author: acorn
'''
import cx_Oracle

conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')

cursor = conn.cursor()

cursor.execute('DROP TABLE drugkorea')


print('성공')
cursor.close()

conn.close()