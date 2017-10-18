import csv
import numpy as np
import ast
import time

VOC_NUM = 61188
TRAINING_NUM = 12000
GROUP_NUM = 20
num_group = np.zeros(GROUP_NUM)
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])
PY=np.zeros(GROUP_NUM)

def Main(beta):
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  testfile = open("testing.csv", 'rb')
  testReader = csv.reader(testfile)
  outfile = open("answer"+str(beta)+".csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["id","class"])
  i=0
  for row in resultReader:
    row = np.array(row)
    row = row.astype(float)
    num_group[i] = row[-1]
    voc_in_group[i] = row[:-1]
    i+=1
  total=0
  for i in range(GROUP_NUM):
    s=sum(voc_in_group[i])
    total=total+s
    print s
    PY[i]=s
    for j in range(VOC_NUM):
      voc_in_group[i][j]=np.log((voc_in_group[i][j]+beta)/(s+ VOC_NUM*beta))
 
  for i in range(GROUP_NUM):
    PY[i]=np.log(PY[i]/total)
  
  print "Start reading test file..."
  
  for row in testReader:
    #print type(row)  
    row = np.array(row)
    row = row.astype(float)
    #print type(row)
    row2=np.reshape(row[1:], (VOC_NUM,1))
    answer =np.dot(voc_in_group,row2)
    for i in range(GROUP_NUM):
		answer[i]=answer[i]+PY[i]
    m=max(answer)
    for i in range(20):
		if answer[i]==m:
			r=i+1
			break
    outputWriter.writerow([int(row[0]),int(r)])
    
    
    


beta=0.00001
while( beta <=1 ):
	Main(beta)
	beta=beta*10
