from xml.etree.ElementTree import parse
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

tree = parse('drug_201610.xml')
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

df2 = df[df['회수의무자']=='바이엘코리아(주)']
df2 = df2.reindex(columns=cols)


mygroup2 = df2.groupby(['위험등급'])['제품명'].count()

mygroup = df.groupby(['위험등급'])['제품명'].count()

print(mygroup)
print(mygroup2)

mergedf = DataFrame([mygroup, mygroup2])
print(mergedf)
