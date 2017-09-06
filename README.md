# Document-Image-Categorization
Machine learning based approach to classify scanned images of documents (written by different authors) based on their category, without recognizing the text. The writers and topics are having a many-to-many relation. Gaussian blur, SIFT followed by SVM is used in two layer optimization to reach desired accuracy. Python is primarily used along with OpenCV framework.

Unzip the package and copy the Folder in working directory.

NOTE: The dataset is not uploaded in the repository. Own data set can be created by uploaded scanned images of documents from given 4 cateories.


*************************
Execute python scripts in following sequence:
	1. Sift_Compute.py
	2. ComputeSIFTofClusters.py
	3. ConsolidateClusters.py
	4. ComputeHistogram.py
	5. FinalClassify.py
	
*************************


	
**************************	
Decsription of above scripts:
**************************


****************
Sift_Compute.py:
**************** 
This computes the sift of the documents of each category individualy. Outputs the Sift descriptiors, the mapping of descriptor to corresponding document and document sequence in the SIFTPoints, SIFTPoints_Mapping, Document_Indexes files respectively in corresponding <category>_Clusters folder.
This also outputs the documents with SIFT points drawn on them in <category>_Clusters\OutputImages folder.

************************
ComputeSIFTofClusters.py:
************************
This averages the descriptiors for each of the cluster and assigns the value as the descriptor of corresponding cluster.
Output in <category>_Clusters\SIFTofClusters folder

**********************
ConsolidateClusters.py:
**********************
Merges all the cluster obtained in above step
Output in \AllCluster\TotalClusters folder

*******************
ComputeHistogram.py:
*******************
Assigns each of the keypoint of image to its nearest cluster and then creats a histogram of document with its corresponding points (cluster versues frequency of point on each cluster).
Output in <category>_Clusters\LocalHistograph

****************
FinalClassify.py:
****************
Finally classifies the data set with 80% for training and 20% for testing based on above histogram


