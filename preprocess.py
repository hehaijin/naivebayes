import csv
import numpy as np

VOC_NUM = 61188
TRAINING_NUM = 12000
GROUP_NUM = 20


voc_in_group = np.zeros([GROUP_NUM,VOC_NUM+1])
def Main():
  csvfile = open("training.csv", 'rb')
  csvreader = csv.reader(csvfile)
  print "Start reading training file..."
  for row in csvreader:
    row = np.array(row)
    row = row.astype(int)
    if (row[0]%500==0):
      print row[0] , "/12000 Finished"
    label = row[-1]-1
    voc_in_group[label][:-2] += row[1:-2]
    voc_in_group[label][-1] +=1 
    
  print "Start output..."
  outcsv = open('result.csv','wb')
  output = csv.writer(outcsv)
  for group in voc_in_group:
    output.writerow(group)
  print "Done"

Main()
