from xml.etree.ElementTree import parse
import pandas as pd
import math
import matplotlib.pyplot as plt

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

def problem_2(df):
    new_df = df.copy(True)
    for i in range(len(new_df)-1,-1,-1):
        if '자진' not in df['회수사유'].ix[i]:
            new_df = new_df.drop(i)
    
    mygroup = df.groupby(['위험등급'])['제품명'].count()
    mynewgroup = new_df.groupby(['위험등급'])['회수일자'].count()
    
    newnew = pd.DataFrame([mygroup, mynewgroup]).transpose()
    newnew.rename(columns={'제품명':'전체','회수일자':'자진회수'}, inplace=True)

    newnew.plot(kind='bar', rot=0)
    
    plt.suptitle('위험등급별 전체 빈도수와 자진회수 빈도수')
    plt.xlabel('위험 등급')
    plt.ylabel('빈도 수')
    plt.grid(True)

def __madeDate_6(inp) :
    if inp=='제조일자확인불가' or inp=='해당없음':
        return '제조일자확인불가'
    else :
        return pd.to_datetime(inp)

def correlation(x,y):
    n = len(x)
    vals = range(n)
    
    x_sum = 0.0
    y_sum = 0.0
    x_sum_pow = 0.0
    y_sum_pow = 0.0
    mul_xy_sum = 0.0
    
    for i in vals:
        mul_xy_sum = mul_xy_sum + float(x[i])*float(y[i])
        x_sum = x_sum + float(x[i])
        y_sum = y_sum + float(y[i])
        x_sum_pow = x_sum_pow + pow(float(x[i]), 2)
        y_sum_pow = y_sum_pow + pow(float(y[i]), 2)
    try :
        r = ((n*mul_xy_sum)-(x_sum*y_sum)) / math.sqrt(((n*x_sum_pow)-pow(x_sum,2)) * ((n*y_sum_pow)-pow(y_sum,2)))
    except:
        r = 0.0
    return r

def problem_6(df):
    df = df.copy(True)
    df.회수일자 = df.회수일자.apply(lambda x : pd.to_datetime(x))
    df.제조일자 = df.제조일자.apply(__madeDate_6)
    df['판매일수'] = 0
    for index, item in df.iterrows():
        if item.제조일자 == '제조일자확인불가':
            item['판매일수'] = '확인불가'
        else :
            temp = abs((item.회수일자- item.제조일자).days)
            item['판매일수'] = temp
        df.ix[index] = item
    
    for index, item in df.iterrows():
        if item.판매일수 == '확인불가' or item.위험등급 == '9' or item.위험등급 == '해당없음':
            df = df.drop(index)
    
    fig = plt.figure()
    fig.suptitle('판매일수와 위험등급 간 상관관계')
    r = correlation(list(df['판매일수']),list(df['위험등급']))
    print('상관계수 :', r)
    plt.grid(True)
    plt.scatter(list(df['판매일수']), list(df['위험등급']), edgecolor='none', alpha=0.75, s=15, c='blue')
    plt.ylabel('위험 등급')
    plt.xlabel('판매일수')
    plt.show()
    
if __name__ == '__main__':
    df = readFile()
    plt.rc('font', family='Malgun Gothic')
    print('[P2] : 위험 등급별 전체 빈도수와 자진회수 빈도수(막대그래프)')
    problem_2(df)
    
    print('[P6] : 회수일자와 제조일자간 일수 계산 후 판매일수라는 컬럼으로 저장 후,\n\t판매일수와 위험등급 간 상관관계 분석')
    problem_6(df)

