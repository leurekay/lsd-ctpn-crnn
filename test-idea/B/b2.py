import sys


import os
print ('\n====sys.path========\n',sys.path,'\n=============\n')
print ('\n====os.getcwd=======\n',os.getcwd(),'\n============\n')
sys.path.append('../')
sys.path.append('../A')
print ('\n======after=======\n',sys.path,'\n======after=======\n')
print ('\n====os.getcwd=======\n',os.getcwd(),'\n============\n')


path='../data/t.txt'
with open(path,'r') as f:
    print (f.read())
