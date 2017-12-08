import pandas as pd
import urllib.request
import datetime
import xml.etree.ElementTree as ET

saved_folder = 'c:/work/'
result = []

def get_request_url(url, enc='utf-8'):    
    #print( url )
    req = urllib.request.Request(url)

    try: 
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print ("[%s] Url Request Success" % datetime.datetime.now())
            
            try:
                rcv = response.read()
                ret = rcv.decode(enc)
            except UnicodeDecodeError:
                ret = rcv.decode(enc, 'replace')    
            
            return ret
            
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
        return None

def  getNeneAdddress(result):

    Nene_URL = 'http://nenechicken.com/subpage/where_list.asp?target_step2=%s&proc_type=step1&target_step1=%s' % (urllib.parse.quote('전체'), urllib.parse.quote('전체'))
#     Nene_URL = 'http://nenechicken.com/subpage/where_list.asp?target_step1=&target_step2=&proc_type=step1&'
#     print(Nene_URL)

    rcv_data = get_request_url(Nene_URL)

    root = ET.fromstring(rcv_data)
    
    for element in root.findall('item'):
        store_name = element.findtext('aname1')
        store_sido = element.findtext('aname2')
        store_gungu = element.findtext('aname3')
        store_address = element.findtext('aname5')
        
        result.append([store_name] + [store_sido] + [store_gungu] + [store_address])

print('NENE ADDRESS CRAWLING START')
    
getNeneAdddress(result)    

nene_table = pd.DataFrame(result, columns=('store', 'sido', 'gungu', 'store_address'))

nene_table.to_csv( saved_folder + "nene_table.csv", encoding="cp949", mode='w', index=True)

print('NENE ADDRESS CRAWLING End')