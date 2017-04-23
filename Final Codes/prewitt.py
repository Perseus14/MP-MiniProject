import cv2
import numpy as np
import pickle

img1 = cv2.imread("../Perseus14/Codes/sari1.jpg") #First Image
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img = cv2.GaussianBlur(img1, (21, 21), 0) #for single column/shelf change the window size to 29
img = cv2.bilateralFilter(img, 19, 250, 250) #for single column/shelf change the window size to 21

#mask for sobel horizontal detector
#sobel_h = np.array([[-1,-4,-6,-4,-1], [-2,-8,-12,-8,-2], [0,0,0,0,0], [2,8,12,8,2], [1,4,6,4,1]])

prewitt_h = np.array([[-2,-2,-2,-2,-2], [-1,-1,-1,-1,-1], [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2]])
#prewitt_h = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])

#sobel_horizontal_edge_detector = cv2.filter2D(img, -1, sobel_h)
prewitt_horizontal_edge_detector = cv2.filter2D(img, -1, prewitt_h)

(y,x) = img.shape

average_saree_width = 120 #value obtained from hit and trial
start = average_saree_width*0.5

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert the image from grayscale to binary image using a threshold value
for i in range(x):
	for j in range(y):
		if(prewitt_horizontal_edge_detector.item(j,i) > 150):
			prewitt_horizontal_edge_detector.itemset((j,i), 255)			
		else:
			prewitt_horizontal_edge_detector.itemset((j,i), 0)

number_of_saris = 0
cnt = 5
edge_length = 0
#counting the number of horizontal edges by setting a threshold
for j in range(int(start), int(x), int(average_saree_width)):
	for i in range(y):
		val = prewitt_horizontal_edge_detector.item(y-i-1, j)
		if(val == 0):
			cnt += 1
			edge_length = 0
		if(val == 255):
			if(cnt >= 2):
				edge_length += 1
			else:
				cnt = 0
			if(prewitt_horizontal_edge_detector.item(y-i-2,j) == 0):
				cnt = 0
			if(edge_length == 2):
				number_of_saris += 1
	
print "The number of number_of_saris are", number_of_saris

cv2.imshow("Horizontal Prewitt Edge Detector", prewitt_horizontal_edge_detector)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("Save.jpg", sobel_horizontal_edge_detector)

