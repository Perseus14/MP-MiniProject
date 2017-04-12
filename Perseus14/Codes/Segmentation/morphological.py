import cv2
import numpy as np


import pickle

file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)

def display_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    

# Takes an image, convert it into grayscale, threshold it and makes 
# the darker region white and bright region black and we use morphology 
# close with a rectangular box of (2,8) and then using this image to find
# contours. In images where there is a shelf, there is dark region between
# the two sarees in gray scale image which we use to find seperation 
# between two sarees.
def find_lines2(i):
	img1 = b[i]                      
	display_image(img1)
	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	th, dst = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
	#cv2.imwrite(file[:-4]+"_1.jpg",dst)
	se = np.ones((2, 8), dtype='uint8')
	img = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, se)
	#cv2.imwrite(file[:-4]+"_2.jpg",img)
	#edges = cv2.Canny(img, 100, 200, apertureSize=3)
	display_image(img)
	im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img1, contours, -1, (0,255,0), 3)
	display_image(img1)
	#cv2.imwrite(file[:-4]+"_3.jpg",img1)

for i in xrange(0,10):	
	find_lines2(i)
