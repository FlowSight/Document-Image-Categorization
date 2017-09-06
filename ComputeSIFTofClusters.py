
import numpy as np
import sys
import os

from sklearn.cluster import KMeans

#from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from random import shuffle





import CateoryConfig as myconfig

categories=myconfig.categories.split(",")
for cat in categories:
	path = os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/"
	siftpts=np.load(path+"SIFTPoints") 
	print("clustering the sift of "+cat+" category...")
	km = KMeans(n_clusters=100, max_iter=10, n_init=4).fit(siftpts)
	labels = km.labels_
	n_clusters_ = len(set(labels))

	for j in range (0,len(labels)):
		print(str(j)+"-->"+str(labels[j]))


	AVG_sift_of_Clusters = np.zeros(shape=(n_clusters_,128)).astype(np.float64)
	AVG_sift_countpercluster = np.zeros(shape=(n_clusters_,1)).astype(np.int32)

	for i in range(0,len(labels)):
		AVG_sift_of_Clusters[labels[i]]=AVG_sift_of_Clusters[labels[i]]+siftpts[i]
		AVG_sift_countpercluster[labels[i]]=AVG_sift_countpercluster[labels[i]]+1
	for i in range(0,n_clusters_):
		AVG_sift_of_Clusters[i]=AVG_sift_of_Clusters[i]/AVG_sift_countpercluster[i]

	AVG_sift_of_Clusters = AVG_sift_of_Clusters.astype(np.float32)
	output=open(os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/SIFTofClusters",'w')
	np.save(output, AVG_sift_of_Clusters)
