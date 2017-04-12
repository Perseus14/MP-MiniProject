import cv2
import numpy as np
import pickle

file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)

img1 = cv2.imread("fig3.jpg") #First Image
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img = cv2.GaussianBlur(img1,(13,13),0)

#mask for sobel horizontal detector
sobel_h = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [2,8,12,8,2], [1,4,6,4,1]])
sobelHorEdge = cv2.filter2D(img, -1, sobel_h)

(y,x)=img.shape
xt=(6*x)/14

#make the image from grayscale to binary image
for i in range(x):
	for j in range(y):
		if(sobelHorEdge.item(j,i)>150):
			sobelHorEdge.itemset((j,i),255)			
		else:
			sobelHorEdge.itemset((j,i),0)

clothes=0
nmbr_zeros=5
edge_len=0
#counting the number of horizontal edges by setting a threshold
for i in range(y):
	val=sobelHorEdge.item(y-i-1,xt)
	if(val==0):
		nmbr_zeros+=1
		edge_len=0
	if(val==255):
		if(nmbr_zeros>=2):
			edge_len+=1
		else:
			nmbr_zeros=0
		if(sobelHorEdge.item(y-i-2,xt)==0):
			nmbr_zeros=0
		if(edge_len==2):
			clothes+=1
print "number of clothes:",clothes

cv2.imshow("img",sobelHorEdge)
cv2.imshow("img1",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
