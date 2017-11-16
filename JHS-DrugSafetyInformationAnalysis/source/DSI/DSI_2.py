from xml.etree.ElementTree import parse
import pandas as pd
import matplotlib.pyplot as plt

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
    
df = pd.DataFrame(totallist, columns=cols)

print(df.groupby(['회수사유'])['제품명'].count)


# mygroup = df.groupby(['위험등급'])['제품명'].count()

# 
# newnew = pd.DataFrame([mygroup, mynewgroup]).transpose()
# newnew.rename(columns={'제품명':'전체','회수일자':'자진회수'}, inplace=True)
# print(newnew)
# plt.rc('font', family='Malgun Gothic')
# newnew.plot(kind='bar', rot=0)
# # plt.legend(loc='upper right')
# # plt.title('위험 등급 별 자진회수 빈도수 분석')
# plt.xlabel('위험 등급')
# plt.ylabel('빈도 수')
# plt.grid(True)
# plt.show()