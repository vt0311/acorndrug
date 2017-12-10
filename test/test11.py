'''
Created on 2017. 11. 13.

@author: RYZEN-HSW
'''

from xml.etree.ElementTree import parse
from pandas import DataFrame

tree = parse('의약품 안전성정보(2016.10)수정.xml')
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
    
df = DataFrame(totallist, columns=cols)
# print(df)

newframe = df[ df['등록일자'] == '20091008']
print(newframe)

mygroup = df.groupby('위험등급')['제품명']
# print(mygroup.count())