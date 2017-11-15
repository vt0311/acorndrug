'''
Created on 2017. 11. 15.

회수의무자

@author: 오기철
'''
import pandas as pd
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
from matplotlib import font_manager, rc
from pandas import DataFrame

font_location = 'c:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name) 

#tree = parse('의약품 안전성정보(2016.10)수정.xml')
tree = parse('drug_201610.xml')
myroot = tree.getroot()
# print(tree)

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


df = DataFrame(totallist, columns=cols)
# print(df)

mygroup = df.groupby('등록일자')['회수의무자']
result = mygroup.count()
print(result)

result.plot(kind='bar')

filename = 'drug4.png'
plt.savefig(filename, dpi=400, bbox_inches='tight')
plt.show()

mygroup = df.groupby('위험등급')