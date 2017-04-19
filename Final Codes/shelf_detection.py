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
		print "None-",threshold
	
	while(len(lines) < 20):
		lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Hough line detection
		threshold -= 1
		print len(lines),threshold
	
	#lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
	hough_lines = []
	
	
	for line in lines[:20]:
		try:
			for rho,theta in line:
			#if( theta > 0.09 and theta < 1.48 or theta < 3.14 and theta > 1.66): 
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
				hough_lines.append(((x1, y1), (x2, y2)))
			#cv2.line(h,(x1,y1),(x2,y2),(255,255,255),6)
		except:
			pass
	#print len(hough_lines)
	return hough_lines



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

# Random sampling of lines 
def sample_lines(lines, size):
	if size > len(lines):
		size = len(lines)
	return random.sample(lines, size)


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


# Find intersections between multiple lines (not line segments!)
# If lines cross, then add
# Don't include intersections that happen off-image
# Seems to cost more time than it saves
# if not (intersection[0] < 0 or intersection[0] > img.shape[1] or intersection[1] < 0 or intersection[1] > img.shape[0]):
# print 'adding', intersection[0],intersection[1],img.shape[1],img.shape[0]
def find_intersections(lines, img):
	intersections = []
	for i in xrange(len(lines)):
		line1 = lines[i]
		for j in xrange(i + 1, len(lines)):
			line2 = lines[j]
			intersection = line_intersection(line1, line2)
			if intersection:
				intersections.append(intersection)
	return intersections



#length of a vector
def length_vector(vec):
	return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

#find angle between two lines (x1,y1) and (x2,y2) is line1 ,(x3,y3) and (x4,y4) is line2.
def angle(x1,y1,x2,y2,x3,y3,x4,y4):
	line1 = [x2-x1,y2-y1]
 	line2 = [x4-x3,y4-y3]
	print line1,line2
	angle = math.asin(det(line1, line2)/math.ceil(length_vector(line1)* length_vector(line2)));
	return angle

# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, intersections):
	# Image dimensions
	image_height = img.shape[0]
	image_width = img.shape[1]

	# Grid dimensions
	grid_rows = (image_height // grid_size) + 1
	grid_columns = (image_width // grid_size) + 1

	print grid_rows,grid_columns
	# Current cell with most intersection points
	max_intersections = 0
	best_cell = None

	for i in xrange(grid_rows):
		for j in xrange(grid_columns):
			cell_left = i * grid_size
			cell_right = (i + 1) * grid_size
			cell_bottom = j * grid_size
			cell_top = (j + 1) * grid_size
			cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

			current_intersections = 0  # Number of intersections in the current cell
			for x, y in intersections:
				if cell_left < x < cell_right and cell_bottom < y < cell_top:
					current_intersections += 1	
			# Current cell has more intersections that previous cell (better)
			if current_intersections > max_intersections:
				max_intersections = current_intersections
				best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)

	print(best_cell)
	
	if not(best_cell == [None, None]):
		rx1 = best_cell[0] - grid_size / 2
		ry1 = best_cell[1] - grid_size / 2
		rx2 = best_cell[0] + grid_size / 2
		ry2 = best_cell[1] + grid_size / 2
		cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 10)
		#cv2.imwrite('center.jpg', img)
		
	return best_cell

file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
t=1
for img in b:
	kernel = np.ones((5, 5), np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
	#img = cv2.GaussianBlur(img, (5,5), 200)
	edges = cv2.Canny(opening,100,200)
	img_c = img.copy()
	img_cc = img.copy()
	'''
	lsd = cv2.createLineSegmentDetector(0)
	img_cc = cv2.cvtColor(img_cc, cv2.COLOR_BGR2GRAY) 
	#Detect lines in the image
	lines = lsd.detect(img_cc)[0]#Position 0 of the returned tuple are the detected lines
	draw_img = lsd.drawSegments(img_cc,lines)	
	#Draw detected lines in the image
	h_lines=[]
	for line in lines:
		[[x1,y1,x2,y2]] = line
		h_lines.append([(x1,y1),(x2,y2)])
	'''
	h_lines=hough_transform(img_c,edges)
	#print edges.shape,img.shape
	h=np.zeros((img.shape[0],img.shape[1]))
	new_img=edges.copy()
	for line in h_lines:
		[(x1,y1),(x2,y2)]=line
		cv2.line(h,(x1,y1),(x2,y2),(255,255,255),1)
	#print h.shape
		
	cv2.imshow("img2",img_c);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#cv2.imwrite(str(t)+"_idk.jpg",img_c)
	#t+=1
	
	cv2.imshow("img2",edges);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''	
	
	#new_img=cv2.bitwise_and(h,edges)
	edges1 = cv2.Canny(opening,50,200)
	for x in xrange(edges1.shape[0]):
		for y in xrange(edges1.shape[1]):
			if(edges1[x][y]==255 and h[x][y]==255):
				new_img[x][y]=255
			else:
				new_img[x][y]=0
	#print new_img.type()
		
		
	cv2.imshow("img2",new_img);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
	#ph_lines=Phough_transform(img,new_img)
	n_img = new_img.copy()
	lsd = cv2.createLineSegmentDetector(0)
	lines = lsd.detect(new_img)[0]
	drawn_img = lsd.drawSegments(img,lines)
	
	for x in xrange(img.shape[0]):
		for y in xrange(img.shape[1]):
			if(n_img[x][y]==255):
				img[x][y]=(0,255,0)	
		
	cv2.imshow("img2",img);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
	cv2.imwrite(str(t)+"_img_n.jpg",n_img)	
	cv2.imwrite(str(t)+"_img.jpg",img)
	t+=1	
	
	
	print len(h_lines)
	l= sample_lines(h_lines, 10)
	intersections = find_intersections(l,img)
	x_avg = 0;
	y_avg = 0;
	count = 0;
	
	print len(intersections)
	for x in intersections:
			count+=1
			x_avg +=x[0]
			y_avg +=x[1]
	
	vanishing_point = find_vanishing_point(img_cc,intersections) #[x_avg/count, y_avg/count]
	top_right = [img.shape[1],0]
	bottom_right = [img.shape[1],img.shape[0]]
	required_angle = angle(vanishing_point[0], vanishing_point[1], bottom_right[0], bottom_right[1], vanishing_point[0], vanishing_point[1], top_right[0], top_right[1])

	step = img.shape[0] / 100
    
	result_lines = []
	for x in range(0,99):
		line1 = [vanishing_point, [img.shape[1],img.shape[0] - x*step]]
		line2 = [top_right, bottom_right]
		(x1,y1) = line_intersection(line1, line2)
		line1 = [vanishing_point, [ img.shape[1] , img.shape[0] - x*step]]
		line2 = [[0 , 0], [0 , img.shape[0]]]
		(x2 , y2) = line_intersection(line1, line2)
		result_lines.append(((x1, y1), (x2, y2)))
	for x in result_lines:
		for y in h_lines:
			if (abs(angle(x[0][0],x[0][1],x[1][0],x[1][1],y[0][0],y[0][1],y[1][0],y[1][1])) < 0.0005):
				cv2.line(img, x[0], x[1], (0, 0, 255), 2)	

	cv2.imshow("img1",img);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow("img2",draw_img);
	cv2.waitKey(0)
	cv2.imshow("img3",edges);
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	plt.subplot(131), plt.imshow(tf_cv_plt(img)),plt.title("Original Image"),plt.xticks([]),plt.yticks([])
	plt.subplot(132), plt.imshow(edges,'gray'),plt.title("Edges"),plt.xticks([]),plt.yticks([])
	plt.subplot(133), plt.imshow(tf_cv_plt(img_c)),plt.title("Hough"),plt.xticks([]),plt.yticks([])
	plt.tight_layout()
	#plt.show()	
	plt.savefig(str(t)+"_g.jpg")
	plt.close()
	t+=1
	'''
