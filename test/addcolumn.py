'''
Created on 2017. 12. 8.

@author: cecil
'''
import pandas as pd
from pandas.core.frame import DataFrame
class addcolumn:
    
    
    #read file, to dataframe
    df_part1_db=pd.read_csv('2014_part1_db.csv',encoding="EUC-KR")
    #df_part1_py=pd.read_csv('2014_part2_py.csv',encoding="EUC-KR")
  
    pkey=[]
    
    #pkey
    for index,row in df_part1_db.iterrows():
        row_pkey=str(row['가입자일련번호'])+'_'+str(row['처방내역일련번호'])+'_'+str(row['일련번호'])
        pkey.append(row_pkey)

    df_part1_db['pkey']=pkey
    #df_part1_py['pkey']=pkey
    
    print('시작')
    
    #file 생성
    df_part1_db.to_csv('add_pkey_2014db1.csv',encoding='cp949', mode='w')
   # df_part1_py.to_csv('add_pkey_2014py2.csv',encoding='cp949', mode='w')
    
    print('end')
    






