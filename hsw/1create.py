'''
Created on 2017. 10. 23.

@author: acorn
'''
#import sqlite3
import cx_Oracle

conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')

cursor = conn.cursor()

#cursor.execute("CREATE TABLE drug(title text, weather text, contents text, reg_date text) ")
cursor.close()

conn.close()

