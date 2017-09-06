import cv2
import numpy as np
import sys
import os
import shutil
#from sklearn.cluster import KMeans

from random import shuffle
import CateoryConfig as myconfig
from sklearn import svm

final_map=[]

categories=myconfig.categories.split(",")
catCount=0;
print ("Computing histogram categorywise")
for cat in categories:
	path = os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/"
	output=open(path+"Document_Indexes",'r')
	doc_names = output.readlines()
	histogram_local=np.load(path+"LocalHistograph",'r')
	catg=0;
	if cat == "Economics":
		catg=1
	if cat == "Health":
		catg=2
	if cat == "Politics":
		catg=3
	if cat == "Sports":
		catg=4
	for i in range(0,len(histogram_local)):
		final_map.append([doc_names[i],histogram_local[i],catg])

shuffle(final_map)

histogramData = np.zeros(shape=(len(final_map),400))
doc_name_shuffled=[]
doc_category=[]
doc_name=[]
for i in range(0,len(final_map)):
	histogramData[i]=final_map[i][1]
	doc_name.append(final_map[i][0])
	doc_category.append(final_map[i][2])

trainDataSize=int(0.8*len(final_map))


train_histogram, test_histogram = histogramData[:trainDataSize], histogramData[trainDataSize:]
train_doc_category,test_doc_category=doc_category[:trainDataSize], doc_category[trainDataSize:]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(train_histogram,train_doc_category)


print("Predicted = "+str(clf.predict(test_histogram)))

print("Actual = "+str(doc_category[trainDataSize:]))

efficiencyCount=0;

for i in range (0,len( clf.predict(test_histogram))):
	if(clf.predict(test_histogram)[i]==doc_category[trainDataSize:][i]):
		
		efficiencyCount=efficiencyCount+1

print("% efficiency = "+str(efficiencyCount*100/(len(final_map)-trainDataSize)))

#print("Documents which formed the test set" + str(doc_name[trainDataSize:]))










