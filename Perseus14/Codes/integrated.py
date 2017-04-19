import cv2
import numpy as np
import pickle
import random
import math
from matplotlib import pyplot as plt

grid_size=5

def tf_cv_plt(img):
	M=img.copy()
	for x in xrange(len(img)):
		for y in xrange(len(img[x])):
			M[x][y]=M[x][y][::-1]
	return M

def hough_transform(img,edges):
	threshold = 200
	
	lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
	
	while(lines==None):
		lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Hough line detection
		threshold -= 1
		#print "None-",threshold
	
	while(len(lines) < 200):
		lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Hough line detection
		threshold -= 1
		#print len(lines),threshold
	
	#lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
	hough_lines_per = []
	hough_lines_par = []
	for line in lines:
		try:
			for rho,theta in line:
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				if(theta < 0.01 and theta > -0.01): 				
					hough_lines_per.append(((x1, y1), (x2, y2)))
				if(theta > 1.55 and theta < 1.59): 				
					hough_lines_par.append(((x1, y1), (x2, y2)))
		except:
			pass
	#print len(hough_lines)
	for line in hough_lines_per[:min(len(hough_lines_per),10)]:
		[(x1,y1),(x2,y2)]=line 		
		cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
	for line in hough_lines_par[:min(len(hough_lines_par),10)]:
		[(x1,y1),(x2,y2)]=line 		
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	return hough_lines_per[:min(len(hough_lines_per),10)],hough_lines_par[:min(len(hough_lines_par),10)]

def det(a, b):
	return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
	x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	div = det(x_diff, y_diff)
	if div == 0:
		return None  # Lines don't cross

	d = (det(*line1), det(*line2))
	x = det(d, x_diff) / div
	y = det(d, y_diff) / div

	return x, y


def Phough_transform(img,edges):
	minLineLength = 5
	maxLineGap = 2
	hough_lines=[]
	lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength,maxLineGap)
	#lsd = cv2.createLineSegmentDetector(0)	
	#lines=lsd.detect(img)	
	for i in range(0,lines.shape[0]):
		for x1,y1,x2,y2 in lines[i]:
			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
			hough_lines.append(((x1, y1), (x2, y2)))
	return hough_lines

file_Name = "../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
#b = [cv2.imread("sari-store.jpg")]
for img in [b[3]]:
	kernel = np.ones((5, 5), np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
	edges = cv2.Canny(opening,100,200)

	img_h = img.copy()
	img_bound = img.copy()
	h_lines_per,h_lines_par=hough_transform(img_h,edges)

	hough=np.zeros((img.shape[0],img.shape[1]))
	for line in h_lines_per:
		[(x1,y1),(x2,y2)]=line
		cv2.line(hough,(x1,y1),(x2,y2),(255,255,255),1)
	
	sort_per_lines = sorted(h_lines_per)
	sort_par_lines = sorted(h_lines_par, key=lambda x: x[0][1])

	intersect_points=[]
	for line1 in sort_per_lines:
		temp=[]
		for line2 in  sort_par_lines:
			(x,y)=line_intersection(line1, line2)
			temp.append((x,y))
		intersect_points.append(temp)
	#intersect_points of perpendicular & parallel, first element set of intersect points of first per line with all par line
	for l in xrange(len(intersect_points)-1):
		for k in xrange(len(intersect_points[l])-1):
			(x1,y1) = intersect_points[l][k]
			(x2,y2) = intersect_points[l+1][k+1]
			if((x2-x1)*(y2-y1)>5000):			 
				cv2.rectangle(img_bound,(x1,y1),(x2,y2),(255,0,0),2) 
	
	

	#Counting Code

	img_count=img.copy()
	img_gray = cv2.cvtColor(img_count, cv2.COLOR_BGR2GRAY) 
	img_blur = cv2.GaussianBlur(img_gray,(25,25),0)
	img_filter = cv2.bilateralFilter(img_blur, 19, 250, 250)      
	#BILATERAL TAKES EDGES IN THE DESIGN AS WELL SO IT DOESN'T WORK AS WELL AS GAUSSIAN

	#mask for sobel horizontal detector
	sobel_h = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [2,8,12,8,2], [1,4,6,4,1]])
	sobel_v = np.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0, -12, -6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
	sobelHorEdge = cv2.filter2D(img_filter, -1, sobel_h)
	#sobelVerEdge = cv2.filter2D(img, -1, sobel_v)

	(y,x)=img_filter.shape

	width = x/4.5
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
		

	print "Number of clothes:",clothes

	#Contour Code
	
	img_contour= img.copy()
	img_contour= cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
	#img_blur = cv2.GaussianBlur(img_gray,(25,25),0)
	#img_filter = cv2.bilateralFilter(img_contour, 19, 250, 250)
	ret,thresh = cv2.threshold(img_contour,20,255,0)
	img_fresh = img.copy()

	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img_fresh, contours, -1, (0,0,255), 3)
	cv2.drawContours(sobelHorEdge, contours, -1, (200,200,200), 3)
	for line in h_lines_per:
		[(x1,y1),(x2,y2)]=line
		cv2.line(sobelHorEdge,(x1,y1),(x2,y2),(100,100,100),1)		
	cv2.imshow('contour', img_fresh)
	cv2.imshow("Sobel",sobelHorEdge)
	cv2.imshow("hough_img",img_h);
	cv2.imshow("Pure Hough",hough);
	cv2.imshow('Bounding', img_bound)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite("contour.jpg",img_fresh)
	cv2.imwrite("Sobel.jpg",sobelHorEdge)
	cv2.imwrite("hough_img.jpg",img_h)
	cv2.imwrite("Pure_hough.jpg",hough)
	cv2.imwrite('Bounding.jpg', img_bound)
