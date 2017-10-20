import csv
import numpy as np
import ast
import math
import pprint

pp = pprint.PrettyPrinter(indent=4)

# number of vocabularies
VOC_NUM = 61188
# number of training samples
TRAINING_NUM = 12000
# number of classes
GROUP_NUM = 20
# number of docs labeled Y_k
num_group = np.zeros(GROUP_NUM)
# vocabulary lists according to each classes
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])
# P(Y|X) for each vocabulary with respect to each class
voc_prob = np.zeros([GROUP_NUM,VOC_NUM])
beta = 1.0 + 1.0/VOC_NUM 


########################################
# Main method handles input and outputs#
# and calls ComputeRanking()           #
######################################## 
def Main():
  # input file
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  # input file
  vocfile = open("vocabulary.txt", 'rb')
  vocReader = csv.reader(vocfile)
  # output file
  outfile = open("Top_100_Vocs.csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["Index", "Top 100 Vocabularies",  "Class Belongs To"])

  # read result.csv generated from preprocess.py
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

  #for i in range(100):
	  #for j in range(GROUP_NUM):
		  #print voc_in_group[j][answer[i]]
# # output the top 100 vocabulary without class
#   for i in range(100): 
#     outputWriter.writerow([voc[int(answer[i])]])

  # outputwrites the 100 vocs and the class index that voc appears the most in
  a = 1
  for j in range(100):
    maxX = 0
    index = 0
    for i in range(GROUP_NUM): 
      if(voc_in_group[i][int(answer[j])] > maxX):
        maxX = voc_in_group[i][int(answer[j])]
        index = i
      # print ("voc_in_group["+str(i)+"][int(answer["+str(j)+"])]: "+str(voc_in_group[i][int(answer[j])]))
    outputWriter.writerow([a, voc[int(answer[j])],  int(index+1)])
    a += 1

##############################################################################
# This method ouputs the ranking of vocabularies' indices by smallest entropy#
# 1.computes the probability of the vucabulary given class k:                #
# P(Y=y_k|Xi) = P(Y=y_k)*P(Xi=X|Y=y_k)the prior and likehood                 #
# 2.computes the conditional entropy for every vocabulary with respect to    #
# each class                                                                 #
# 3.sums all the conditional entropies and get a total entropy for the       #
# vocabulary.                                                                #
# 4. ranks and output the vocabularies' index                                #
############################################################################## 
def ComputeRanking():
  voc_ranking = np.zeros(VOC_NUM)
  # sum up the count of total vocabularies in class K, size 1 by 20
  sumX = np.sum(voc_in_group, axis=1)
  pp.pprint( sumX)
  # sum up the count of each voc appears in all classes, size 1 by 61188
  sumY = np.sum(voc_in_group, axis=0)
  
  for i in range(GROUP_NUM):
    # compute the prior using MLE
    # we could set all the p(y) = 1/20, to be iid
    py = num_group[i] / TRAINING_NUM
   
    for j in range(VOC_NUM):
      # compute the likelyhood using MAP
      pxy = (voc_in_group[i][j]+beta-1)/(sumX[i]+(beta-1)*VOC_NUM)
      #print pxy
      # compute the P(y_k|x_i) = P(y_k)*P(x_i|y_k)
      voc_prob[i][j] = py*pxy
      # conditional entropy
      #voc_prob[i][j] = -voc_prob[i][j]* math.log(voc_prob[i][j],2)
      # filter out the voc appeared in no classes by giving its entropy a penalty
      #if(sumY[j] == 0.0): 
      #  voc_prob[i][j] += 1
  # total entropy
  vocnomal=np.sum(voc_prob,axis=0)
 
  #voc_prob=np.divide(voc_prob,vocnomal)
  for i in range(GROUP_NUM):
	  for j in range(VOC_NUM):
		  voc_prob[i][j]=voc_prob[i][j]/vocnomal[j]
		  voc_prob[i][j]=-1* voc_prob[i][j]* np.log(voc_prob[i][j])
  voc_ranking = np.sum(voc_prob, axis=0)
  
  pp.pprint(voc_ranking)
  # return the indices of vocabularies sorting from smallest entropy    
  return np.argsort(voc_ranking)

Main()
