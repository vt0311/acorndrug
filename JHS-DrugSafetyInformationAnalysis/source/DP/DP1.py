# Drug Prescription Information Data
# DATE : `17. 11. 14
# Ver : 0.1
# Description : insert data to DB
# Revised by JHS
import cx_Oracle as oracle

def makeFileName():
    names = [str(2002+i) for i in range(14)]
    parts = ['_part1', '_part2']
    for i in range(13,5,-1):
        temp_name = names[i]
        names[i] += parts[1]
        names.insert(i, temp_name+parts[0])
    return names

def insertData(sep_name):
    conn = oracle.connect('fkrfkrdk/root@localhost:1521/xe')
    cur = conn.cursor()
    sql = "INSERT INTO DRUGPRESC(base_year, user_id, pres_id, serial_num, sex, age_code, sido_code, recuperate_date, drug_ingredient_code, dose_once, dose_oneday, dose_days, unit_cost, price) values('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')"
      
    # csv_cols = ['기준년도', '가입자일련번호', '처방내역일련번호', '일련번호', '성별코드', '연령대코드(5세단위)', '시도코드', '요양개시일자', '약품일반성분명코드', '1회 투약량', '1일투약량', '총투여일수', '단가', '금액']
    fileDir = 'D:/Acorn/1차 Project/DataSet/국민건강정보데이터_의약품처방정보/의약품처방정보데이터/'
    fileName = 'NHIS_OPEN_T60_{}.CSV'.format(sep_name)
      
    with open(fileDir+fileName, 'r') as f:
        f.readline()
        while True:
            temp = f.readline().split(',')
            if(len(temp) < 2): break
            del temp[14]
            cur.execute(sql.format(*temp))
            conn.commit()
            print(temp)
    
    cur.close()
    conn.close()
    
if __name__ == '__main__':
    names = makeFileName()
    for name in names:
        insertData(name)