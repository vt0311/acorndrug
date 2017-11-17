'''
Created on 2017. 11. 17

@author: hsw
'''

import cx_Oracle as oracle


def makeFileName():
   
    names = [str(2002+i) for i in range(1)]
    parts = ['_part1', '_part2']

    for i in range(1):
        temp_name = names[i]
        names[i] += parts[1]
        names.insert(i, temp_name+parts[0])
        
    return names

def insertData(sep_name):
    conn = oracle.connect('scott/tiger@localhost:1521/xe')
    cur = conn.cursor()
    sql = "INSERT INTO DRUGPRESC2002(base_year, user_id, pres_id, serial_num, sex, age_code, sido_code, recuperate_date, drug_ingredient_code, dose_once, dose_oneday, dose_days, unit_cost, price) values('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')"
      
    # csv_cols = ['기준년도', '가입자일련번호', '처방내역일련번호', '일련번호', '성별코드', '연령대코드(5세단위)', '시도코드', '요양개시일자', '약품일반성분명코드', '1회 투약량', '1일투약량', '총투여일수', '단가', '금액']
    fileDir = 'NHIS_OPEN_T60_2015/'
    fileName = 'NHIS_OPEN_T60_{}.CSV'.format(sep_name)
      
    with open(fileDir+fileName, 'r') as f:
        f.readline()
        while True:
            temp = f.readline().split(',')
            if(len(temp) == 1): break
            del temp[14]
            cur.execute(sql.format(*temp))
            conn.commit()
            print(temp)
            
conn = cx_Oracle.connect('scott/tiger@localhost:1521/xe')

cursor = conn.cursor()

cursor.execute("insert into diary values('2nd','ro2','hi2','20171023')")

conn.commit()

cursor.close()

conn.close()




