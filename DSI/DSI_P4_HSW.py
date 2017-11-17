'''
Created on 2017. 11. 15.

회수 의무자 빈도수 분석

@author: 하승원
'''
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
from pandas import DataFrame
from DSI.DSI_P35_OKC import setKoreanFont

def readFile_P4():
    tree = parse('DrugSafetyInformation(2016.10)_Revised.xml')
    myroot = tree.getroot()
    
    cols = ['제품명','표준코드','품목명','품목기준코드','회수의무자','회수일자','제조번호','제조일자','포장단위','회수사유','위험등급','등록일자']
    
    data = myroot.findall('Row')
    
    totallist = []
    for item in data:
        childs = item.getchildren()
        sublist = []
        for d in range(len(childs) ) :
            if d == len(childs) - 1 :
                sublist.append((childs[d].text)[0:6])
            else :
                sublist.append(childs[d].text)
        totallist.append(sublist)
    return DataFrame(totallist, columns=cols) 

def problem_4(df):
    mygroup = df.groupby('회수의무자')['회수의무자']
    mynewgroup = mygroup.count()
    
    print(mynewgroup.sort_values().tail(10))
    
    result = mynewgroup.sort_values(ascending=False).head(10)
    
    result.plot(kind='bar')
    plt.title('회수 의무자 빈도수 분석')
    plt.xlabel('회수 의무자')
    plt.ylabel('빈도 수')
    
    plt.xticks(rotation='25')
    plt.savefig('P1-4.png', dpi=400, bbox_inches='tight')

def HSW_main():
    setKoreanFont()
    
    print('[P4] : 회수 의무자 빈도수 분석')
    df = readFile_P4()
    problem_4(df)



