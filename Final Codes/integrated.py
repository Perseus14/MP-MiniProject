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
		cv2.line(img,(x1,y1),(x2,y2),(255,255,0),2)
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

file_Name = "../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
b = [cv2.imread("sari-store.jpg")]
for img in b:
	kernel = np.ones((5, 5), np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
	edges = cv2.Canny(opening,100,200)

	img_h = img.copy()
	img_bound = img.copy()
	h_lines_per,h_lines_par=hough_transform(img_h,edges) #Hough Transform - output Perpendicular and Parallel

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

	list_array=[]
	for l in xrange(len(intersect_points)-1):
		for k in xrange(len(intersect_points[l])-1):
			(x1,y1) = intersect_points[l][k]
			(x2,y2) = intersect_points[l+1][k+1]
			if((x2-x1)*(y2-y1)>5000):
				roi=img_bound[y1:y2,x1:x2]			 
				cv2.rectangle(img_bound,(x1,y1),(x2,y2),(0,255,255),2) 
				list_array.append(roi)
	if(list_array==[]):
		list_array=[img.copy()]

	#Counting Code
	count_saree=0
	for img_count in list_array:
		img_gray = cv2.cvtColor(img_count, cv2.COLOR_BGR2GRAY) 
		img_blur = cv2.GaussianBlur(img_gray,(25,25),0)
		img_filter = cv2.bilateralFilter(img_blur, 19, 250, 250)      
		#BILATERAL TAKES EDGES IN THE DESIGN AS WELL SO IT DOESN'T WORK AS WELL AS GAUSSIAN

		#mask for sobel horizontal detector
		sobel_h = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [2,8,12,8,2], [1,4,6,4,1]])
		sobel_horizontal_edge_detector = cv2.filter2D(img_filter, -1, sobel_h)

		(y,x) = img_filter.shape

		average_saree_width = 120 #value obtained from hit and trial
		start = average_saree_width*0.5

		#convert the image from grayscale to binary image using a threshold value
		for i in range(x):
			for j in range(y):
				if(sobel_horizontal_edge_detector.item(j,i) > 150):
					sobel_horizontal_edge_detector.itemset((j,i), 255)			
				else:
					sobel_horizontal_edge_detector.itemset((j,i), 0)

		number_of_saris = 0
		cnt = 5
		edge_length = 0
		#counting the number of horizontal edges by setting a threshold
		for j in range(int(start), int(x), int(average_saree_width)):
			for i in range(y):
				val = sobel_horizontal_edge_detector.item(y-i-1, j)
				if(val == 0):
					cnt += 1
					edge_length = 0
				if(val == 255):
					if(cnt >= 2):
						edge_length += 1
					else:
						cnt = 0
					if(sobel_horizontal_edge_detector.item(y-i-2,j) == 0):
						cnt = 0
					if(edge_length == 2):
						number_of_saris += 1
	
		#print "The number of number_of_saris are", number_of_saris
		count_saree+=number_of_saris
	print "Total number of sarees is", count_saree

	#Contour Code
	
	img_contour= img.copy()
	img_contour= cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
	#img_blur = cv2.GaussianBlur(img_gray,(25,25),0)
	#img_filter = cv2.bilateralFilter(img_contour, 19, 250, 250)
	ret,thresh = cv2.threshold(img_contour,20,255,0)
	img_fresh = img.copy()

	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img_fresh, contours, -1, (0,0,255), 3)
	cv2.drawContours(sobel_horizontal_edge_detector, contours, -1, (200,200,200), 3)
	for line in h_lines_per:
		[(x1,y1),(x2,y2)]=line
		cv2.line(sobel_horizontal_edge_detector,(x1,y1),(x2,y2),(100,100,100),1)		
	cv2.imshow('contour', img_fresh)
	#cv2.imshow("Sobel",sobel_horizontal_edge_detector)
	cv2.imshow("hough_img",img_h);
	#cv2.imshow("Pure Hough",hough);
	cv2.imshow('Bounding', img_bound)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite("contour.jpg",img_fresh)
	#cv2.imwrite("Sobel.jpg",sobel_horizontal_edge_detector)
	cv2.imwrite("hough_img.jpg",img_h)
	#cv2.imwrite("Pure_hough.jpg",hough)
	cv2.imwrite('Bounding.jpg', img_bound)
