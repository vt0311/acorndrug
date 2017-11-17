from DSI.DSI_P1267_JHS import readFile
from xml.etree.ElementTree import parse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pytagcloud
import webbrowser
from pandas import DataFrame

def setKoreanFont():
    font_location = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    plt.rc('font', family=font_name)
    matplotlib.rc('font', family=font_name)

def __saveWordCloud( wordInfo, filename ):
    taglist = pytagcloud.make_tags(dict(wordInfo).items(), maxsize=50)
    pytagcloud.create_tag_image(taglist, filename, \
                                size=(1020, 960), fontname='korean', rectangular=False)
    webbrowser.open( filename )

def problem_3(df):
    df = df.copy(True)
    mygroup = df.groupby('회수사유')['회수사유']
    mynewgroup = mygroup.count()
    
    result = mynewgroup.sort_values(ascending=False).head(10)
   
    __saveWordCloud(result, 'P1-3.png')

def __readFile_P5():
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
                sublist.append((childs[d].text)[0:4])
            else :
                sublist.append(childs[d].text)
        totallist.append(sublist)
    return DataFrame(totallist, columns=cols)

def problem_5(df):
    mygroup = df.groupby('등록일자')['제품명']
    result = mygroup.count()
    
    result.plot(kind='bar', grid = True, rot=0)
    
    plt.title('연단위 등록일자 빈도수 분석')
    plt.xlabel('연도')
    plt.ylabel('빈도 수')
    plt.savefig('P1-5.png', dpi=1600, bbox_inches='tight')

def OKC_main():
    setKoreanFont()
    
    print('[P3] : 회수 사유 워드클라우드')
    df = readFile()
    problem_3(df)
    
    print('[P5] : 연단위  등록일자 빈도수 분석')
    df = __readFile_P5()
    problem_5(df)
