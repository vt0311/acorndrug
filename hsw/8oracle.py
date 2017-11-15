'''
Created on 2017. 10. 23.

@author: acorn
'''
import cx_Oracle


conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')
cursor = conn.cursor()
cursor.execute('select * from board')
for row in cursor:
        print(row)
cursor.close()
conn.close()



