import csv
import numpy as np
import ast

VOC_NUM = 61188
TRAINING_NUM = 12000
GROUP_NUM = 20
num_group = np.zeros(GROUP_NUM)
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])
beta = 1.0/VOC_NUM

def Main():
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  testfile = open("testing.csv", 'rb')
  testReader = csv.reader(testfile)
  outfile = open("answer.csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["id","class"])
  i=0
  for row in resultReader:
    row = np.array(row)
    row = row.astype(float)
    num_group[i] = row[-1]
    voc_in_group[i] = row[:-1]
    i+=1
  print "Start reading test file..."
  for row in testReader:
    row = np.array(row)
    row = row.astype(float)
    answer = Classification(row[1:])
    outputWriter.writerow([int(row[0]),int(answer)])
    if (int(row[0])%500==0):
      print row[0], "Finished"
  print "Done"
    
     
def Classification(data):
  result = []
  for group in range(GROUP_NUM):
    py = np.log(num_group[group] / TRAINING_NUM)
    prob = (voc_in_group[group]+1+beta)/(np.sum(voc_in_group[group])+1+beta)
    pxy = np.power(prob,data)
    pxy = np.sum(np.log(pxy[pxy!=0]))
    result.append(py+pxy)
  return result.index(max(result))+1
    
Main()
