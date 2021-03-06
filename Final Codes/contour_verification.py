import cv2
import numpy as np
import pickle

#file_Name = "../../Data/input.pkl"
#fileObj=open(file_Name,'rb')
#b=pickle.load(fileObj)

img1 = cv2.imread("crop3.jpg") #First Image (same code used below)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img = cv2.GaussianBlur(img1,(25,25),0)
img = cv2.bilateralFilter(img, 19, 250, 250)      
#BILATERAL TAKES EDGES IN THE DESIGN AS WELL SO IT DOESN'T WORK AS WELL AS GAUSSIAN

#img = cv2.GaussianBlur(img1,(21,21),0)

#mask for sobel horizontal detector
sobel_h = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [2,8,12,8,2], [1,4,6,4,1]])
sobel_v = np.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0, -12, -6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
sobelHorEdge = cv2.filter2D(img, -1, sobel_h)
#sobelVerEdge = cv2.filter2D(img, -1, sobel_v)

(y,x)=img.shape

#width = x/4.5 #4.5
#print width
width = 120
#xt=(6*x)/14
xt = width*0.5

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
for j in range(int(xt),int(x),int(width)):
	for i in range(y):
		val=sobelHorEdge.item(y-i-1,j)
		if(val==0):
			nmbr_zeros+=1
			edge_len=0
		if(val==255):
			if(nmbr_zeros>=2):
				edge_len+=1
			else:
				nmbr_zeros=0
			if(sobelHorEdge.item(y-i-2,j)==0):
				nmbr_zeros=0
			if(edge_len==2):
				clothes+=1

#Contour Code
	
img_contour= img1.copy()
#img_contour= cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)

#img_blur = cv2.GaussianBlur(img_contour,(3,3),0)
#img_filter = cv2.bilateralFilter(img_contour, 19, 250, 250)

ret,thresh = cv2.threshold(img_contour,20,255,0)
img_fresh = cv2.imread('crop3.jpg')
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

sobelHorEdge2 = cv2.cvtColor(sobelHorEdge, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img_fresh, contours, -1, (0,0,255), 3)
cv2.drawContours(sobelHorEdge2, contours, -1, (0,0,255), 3)

print "number of clothes:",clothes

cv2.imshow("thresh2",thresh)
cv2.imshow("sobel2",sobelHorEdge)
cv2.imshow("cnt",img_fresh)

'''cv2.imwrite("thresh2.jpg",thresh)
cv2.imwrite("sobel2.jpg",sobelHorEdge2)
cv2.imwrite("cnt.jpg",img_fresh)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()