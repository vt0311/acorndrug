# Date : `17. 12. 07
# Writer : JHS
# Description : parsing full DB to get atc-codes & products for each drug

fileDir = 'C:/Users/acorn/Desktop/'
fileName_read = 'full database.xml'
fileName_write = 'parsed3.xml'

with open(fileDir+fileName_read, 'r', encoding='UTF-8') as drbk_file:
    with open(fileDir+fileName_write, 'w', encoding='UTF-8') as wf:
        print(drbk_file.readline()) # ignore the first dummy line
        
        drug_flag = False # use for recognize a <drug>
        atc_flag = False # use for recognize a <atc-codes>
        product_flag = False # use for recognize a <products>
        properties_flag = False
        druginteraction_flag = False
        
        while True:
            temp = drbk_file.readline()
            
            
            # end of file
            if(temp.strip() == ''):
                break
            
            print(temp)
            if temp.startswith('<drug'):
                wf.write(temp)
                drug_flag = True
            elif drug_flag and temp.startswith('</drug'):
                wf.write(temp)
            elif drug_flag and temp.startswith('  <name>'):
                wf.write(temp)
                
            elif (not atc_flag) and temp.strip() == '<atc-codes/>':
                wf.write(temp)
            elif (not atc_flag) and temp.startswith('  <atc-codes'):
                atc_flag = True
                wf.write(temp)
            elif atc_flag and temp.startswith('  </atc-codes'):
                atc_flag = False
                wf.write(temp)
            elif atc_flag:
                wf.write(temp)
                
            elif (not product_flag) and temp.strip() == '<products/>':
                wf.write(temp)
            elif (not product_flag) and temp.strip() == '<products>':
                product_flag = True
                wf.write(temp)
            elif product_flag and temp.startswith('  </products>'):
                product_flag = False
                wf.write(temp)
            elif product_flag:
                wf.write(temp)
                
            # 하승원 추가    
            elif (not properties_flag) and temp.strip() == '<calculated-properties/>':
                wf.write(temp)
            elif (not properties_flag) and temp.strip() == '<calculated-properties>':
                properties_flag = True
                wf.write(temp)
            elif properties_flag and temp.startswith('  </calculated-properties>'):
                properties_flag = False
                wf.write(temp)
            elif properties_flag:
                wf.write(temp)  
                
            # 하승원 추가2
            elif (not druginteraction_flag) and temp.strip() == '<drug-interaction/>':
                wf.write(temp)
            elif (not druginteraction_flag) and temp.strip() == '<drug-interaction>':
                druginteraction_flag = True
                wf.write(temp)
            elif druginteraction_flag and temp.startswith('  </drug-interaction>'):
                druginteraction_flag = False
                wf.write(temp)
            elif druginteraction_flag:
                wf.write(temp)    
