'''
Created on 2018. 1. 4.

@author: acorn
'''
import numpy as np
import nltk
import string

result = np.random.randint(0, 3, 4)

print(result)

testString = 'hello brother'

testString1 = testString.split('l')

print(testString1)


testString2 = testString.nltk.tokenize('l')

print(testString2)