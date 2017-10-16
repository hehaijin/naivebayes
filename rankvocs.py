import csv
import numpy as np
import ast
import math

VOC_NUM = 61188
TRAINING_NUM = 12000
GROUP_NUM = 20
num_group = np.zeros(GROUP_NUM)
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])
voc_prob = np.zeros([GROUP_NUM,VOC_NUM])
beta = 1.0/VOC_NUM

def Main():
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  vocfile = open("vocabulary.txt", 'rb')
  vocReader = csv.reader(vocfile)
  outfile = open("Top_100_Vocs.csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["Top 100 Vocabularies"])

  # read result.csv output from preprocess.py
  i=0
  for row in resultReader:
    row = np.array(row)
    row = row.astype(float)
    num_group[i] = row[-1]
    voc_in_group[i] = row[:-1]
    i+=1

  # read vocabulary.txt
  voc = []
  for row in vocReader:
    voc.append(row[0])
  answer = ComputeRanking()
  #output the top 100 vocabulary with smallest entropy
  for i in range(100): 
    outputWriter.writerow([voc[int(answer[i])]])

#calculate the entropy for each vocabulary
def ComputeRanking():
  voc_ranking = np.zeros(VOC_NUM)
  for i in range(GROUP_NUM):
    py = num_group[i] / TRAINING_NUM
    for j in range(VOC_NUM):
      pxy = (voc_in_group[i][j]+1+beta)/(np.sum(voc_in_group[i][j])+1+beta)
      # compute the P(y_k|x_i) = P(y_k)*P(x_i|y_k)
      voc_prob[i][j] = py*pxy
      # conditional entropy
      voc_prob[i][j] = -voc_prob[i][j]* math.log(voc_prob[i][j],2)
      # total entropy
      voc_ranking += voc_prob[i][j] 
  # return the sort from smallest entropy    
  return np.argsort(voc_ranking)

Main()
