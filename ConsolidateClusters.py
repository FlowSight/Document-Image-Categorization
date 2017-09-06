import cv2
import numpy as np
import sys
import os
import shutil
#from sklearn.cluster import KMeans


import CateoryConfig as myconfig
Sift_of_Clusters = np.zeros(shape=(0,128)).astype(np.float64)
categories=myconfig.categories.split(",")
for cat in categories:
	subcluster = os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/"
	sift_subcluster=np.load(subcluster+"SIFTofClusters")
	Sift_of_Clusters=np.concatenate((Sift_of_Clusters, sift_subcluster), axis=0)
output=open(os.path.dirname(os.path.abspath(__file__))+"/AllCluster/TotalClusters",'w')
np.save(output, Sift_of_Clusters)
