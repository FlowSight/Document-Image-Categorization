import cv2
import numpy as np
import sys
import os
import shutil
#from sklearn.cluster import KMeans


import CateoryConfig as myconfig




categories=myconfig.categories.split(",")
catCount=0;
print ("Computing histogram categorywise")
for cat in categories:
	path = os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters"
	keypoints=np.load(path+"/SIFTPoints") 
	print("Processing category "+cat+"...")
	#keypointMap=np.concatenate((Sift_of_Clusters, sift_subcluster), axis=0)
	keypointMap=np.load(path+"/SIFTPoints_Mapping") 
	overallDesPath=os.path.dirname(os.path.abspath(__file__))+"/AllCluster/TotalClusters"
	overallDes=np.load(overallDesPath)
	distance=9999
	path = os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters"
	filenames=open(path+"/Document_Indexes",'r')
	doc_names = filenames.readlines()
	histogram = np.zeros(shape=(len(doc_names),400)).astype(np.float64)
	selectedCluster=0;
	for i in range(0,len(keypoints)):
		for j in range(0,len(overallDes)):
			dist = np.linalg.norm(keypoints[i]-overallDes[j])
			#print dist
			if dist<distance:
				distance=dist
				selectedCluster=j
		print("Processing Point "+str(i)+" of category "+cat+"...")
		histogram[keypointMap[i],selectedCluster]=histogram[keypointMap[i],selectedCluster]+1		
	output=open(os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/LocalHistograph",'w')
	for i in range(0,len(histogram)):
		temp=0
		for j in range(len(histogram[0])):
			temp=temp+histogram[i][j]
		if not (temp==0):
			histogram[i]=histogram[i]/temp
	
	np.save(output, histogram)
	
