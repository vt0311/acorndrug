from xml.etree.ElementTree import parse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def readFile():
    tree = parse('DrugSafetyInformation(2016.10)_Revised.xml')
    myroot = tree.getroot()
    
    data = myroot.findall('Row')
    
    cols = ['제품명','표준코드','품목명','품목기준코드','회수의무자','회수일자','제조번호','제조일자','포장단위','회수사유','위험등급','등록일자']
    
    totallist = []
    for item in data:
        childs = item.getchildren()
        sublist = []
        for d in childs:
            sublist.append(d.text)
        totallist.append(sublist)
        
    return pd.DataFrame(totallist, columns=cols)

def madeDate(input) :
    if input=='제조일자확인불가' or input=='해당없음':
        return '제조일자확인불가'
    else :
        return pd.to_datetime(input)

if __name__ == '__main__':
    df = readFile()
    df.회수일자 = df.회수일자.apply(lambda x : pd.to_datetime(x))
    df.제조일자 = df.제조일자.apply(madeDate)
    df['판매일수'] = 0
    for index, item in df.iterrows():
        if item.제조일자 == '제조일자확인불가':
            item['판매일수'] = '확인불가'
        else :
            temp = abs((item.회수일자- item.제조일자).days)
            item['판매일수'] = temp
        df.ix[index] = item

print(df.groupby('판매일수')['제품명'].count())