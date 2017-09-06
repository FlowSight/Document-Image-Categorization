import cv2
import numpy as np
import sys
import os
import shutil
#from sklearn.cluster import KMeans


import CateoryConfig as myconfig

categories=myconfig.categories.split(",")
for cat in categories:
	print("processing " +cat)


	totaldes = np.empty(shape=[0, 128])
	output=open("output.txt",'w')


	path = os.path.dirname(os.path.abspath(__file__))+"/Doc_Module/"+cat

	fileCount=0
	total_len=0
	doc_name=[]
	keypoint_doc_Map = []
	doc_category=[]
	fileList=os.listdir(path)

	print("Finding SIFTs...")


	for filename in fileList:
		doc_name.append(filename)
	
		print("Finding SIFTs.. File-->"+filename+" number-->"+str(fileCount))
		img = cv2.imread(path+"/"+filename)
		th1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		th1 = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		ret,th1 = cv2.threshold(th1,127,255,cv2.THRESH_BINARY_INV)
		th1 = cv2.GaussianBlur(th1,(11,11),0)
		ret,th1 = cv2.threshold(th1,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		kernel = np.ones((17,17),np.float32)/189
		th1 = cv2.filter2D(th1,-1,kernel)
		ret,th1 = cv2.threshold(th1,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	
		thresh = cv2.adaptiveThreshold(th1,255,1,1,11,2)
		thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
		contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		avg_height=0
		cnt_length=0
		for cnt in contours:
		    x,y,w,h = cv2.boundingRect(cnt)
		    avg_height=avg_height+h
		    cnt_length=cnt_length+1;
		avg_height=avg_height/cnt_length
		

		contourCount=0;	
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			if h>avg_height*0.50 and h<(2.5*avg_height):
				siftim=img[y:y+h, x:x+w]
				sift = cv2.SIFT()
				kp, des = sift.detectAndCompute(siftim,None)
				temp=cv2.drawKeypoints(siftim,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			    	img[y:y+h,x:x+w]=temp
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				if des is not None:	
					total_len=total_len+len(kp)
					totaldes=np.concatenate(( des, totaldes),axis=0)
					for descnt in range(0,len(des)):
						keypoint_doc_Map.append(fileCount)		
			if  h>(2.5*avg_height):
				tempImage1=th1[y:y+(h/2),x:x+w]
				tempImage2=th1[y+(h/2):y+h,x:x+w]
				subcontours, hierarchy = cv2.findContours(tempImage1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				for subcnt in subcontours:
					x1,y1,w1,h1 = cv2.boundingRect(subcnt)
					if h1>avg_height*0.50 and h1<(2.5*avg_height):
						siftim=img[y+y1:y+y1+h1, x+x1:x+x1+w1]
						sift = cv2.SIFT()
						kp, des = sift.detectAndCompute(siftim,None)
						temp=cv2.drawKeypoints(siftim,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			    			img[y+y1:y+y1+h1, x+x1:x+x1+w1]=temp
						cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(255,0,0),2)
						if des is not None:	
							total_len=total_len+len(kp)
							totaldes=np.concatenate((totaldes, des), axis=0)
							for descnt in range(0,len(des)):
								keypoint_doc_Map.append(fileCount)
				subcontours, hierarchy = cv2.findContours(tempImage2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				for subcnt in subcontours:
					x1,y1,w1,h1 = cv2.boundingRect(subcnt)
					if h1>avg_height*0.50 and h1<(2.5*avg_height):
						siftim=img[y+y1+(h/2):y+y1+h1+(h/2), x+x1:x+x1+w1]
						sift = cv2.SIFT()
						kp, des = sift.detectAndCompute(siftim,None)
						temp=cv2.drawKeypoints(siftim,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			    			img[y+y1+(h/2):y+y1+h1+(h/2),x+x1:x+x1+w1]=temp
						cv2.rectangle(img,(x+x1,y+(h/2)+y1),(x+x1+w1,y+(h/2)+y1+h1),(255,0,0),2)					
						if des is not None:	
							total_len=total_len+len(kp)
							totaldes=np.concatenate((totaldes, des), axis=0)
							for descnt in range(0,len(des)):
								keypoint_doc_Map.append(fileCount)
		cv2.imwrite("Words"+filename,img)
		shutil.move(os.path.dirname(os.path.abspath(__file__))+"/Words"+filename,os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/OutputImages/"+filename)
		fileCount=fileCount+1
		

	keypoint_doc_Map=np.asarray(keypoint_doc_Map)

	print("total SIFTs -> "+str(len(totaldes)))

	output=open(os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/SIFTPoints",'w')
	np.save(output, totaldes)
	output=open(os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/SIFTPoints_Mapping",'w')
	np.save(output,keypoint_doc_Map)
	output=open(os.path.dirname(os.path.abspath(__file__))+"/"+cat+"_Clusters/Document_Indexes",'w')
	for doc in doc_name:
		output.write(str(doc)+"\n")




