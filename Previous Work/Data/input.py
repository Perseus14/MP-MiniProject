#Convert Images to Pickle Object for easier access
#Pickle object contains numpy array of images

import pickle
import cv2
import numpy as np

file_Name = "input.pkl"
fileObj = open(file_Name,'wb')
L=[]
while(1):
	try:
		name=raw_input()
		img=cv2.imread(name)
		L.append(img)
	except:
		break
L=np.array(L)
pickle.dump(L,fileObj)
#print L.shape
fileObj.close()

'''
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
print b.shape
for x in b:
	cv2.imshow('result', x)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
'''
