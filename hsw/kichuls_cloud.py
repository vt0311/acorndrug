'''
Created on 2017. 11. 15.

회수 사유 워드 클라우드

@author: 오기철
'''

import pytagcloud
import matplotlib
import webbrowser 
import pandas as pd

from xml.etree.ElementTree import parse
from matplotlib import font_manager, rc
from pandas import DataFrame

def showGraph( wordInfo ):
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name) 

def saveWordCloud( wordInfo, filename ):
    taglist = pytagcloud.make_tags(dict(wordInfo).items(), maxsize=60)
    pytagcloud.create_tag_image(taglist, filename, \
                                size=(1920, 640), fontname='korean', rectangular=False)
    webbrowser.open( filename )
       
def main():
    tree = parse('의약품 안전성정보(2016.10)수정.xml')
    myroot = tree.getroot()
    
    cols = ['제품명','표준코드','품목명','품목기준코드','회수의무자','회수일자','제조번호','제조일자','포장단위','회수사유','위험등급','등록일자']
    
    data = myroot.findall('Row')
    
    totallist = []
    for item in data:
        childs = item.getchildren()
        sublist = []
        for d in childs:
            sublist.append(d.text)
        totallist.append(sublist)
        
    df = DataFrame(totallist, columns=cols)
 
    mygroup = df.groupby('회수사유')['제품명']
    result = mygroup.count()
    
    cloudImagePath = 'drugcloud' + '.png'
   
    showGraph( result )
    saveWordCloud( result, cloudImagePath )
       
if __name__ == '__main__' :
    main()
