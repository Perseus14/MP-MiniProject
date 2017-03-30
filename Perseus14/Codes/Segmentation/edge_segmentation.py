import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
def tf_cv_plt(img):
	M=img.copy()
	for x in xrange(len(img)):
		for y in xrange(len(img[x])):
			M[x][y]=M[x][y][::-1]
	return M

file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
t=1
for img in b:
	#img = cv2.GaussianBlur(img, (3,3),0)
	'''
	kernel = np.ones((5, 5), np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
	edges = cv2.Canny(opening,200,400,True)
	plt.subplot(121), plt.imshow(tf_cv_plt(img)),plt.title("Original Image"),plt.xticks([]),plt.yticks([])
	plt.subplot(122), plt.imshow(edges,'gray'),plt.title("Edges"),plt.xticks([]),plt.yticks([])
	plt.tight_layout()
	#plt.savefig(str(t)+"_g.jpg")
	plt.show()	
	plt.close()
	t+=1
	'''
	lsd = cv2.createLineSegmentDetector(0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	#Detect lines in the image
	lines = lsd.detect(img)[0]#Position 0 of the returned tuple are the detected lines
	for line in lines:
		print line
	#Draw detected lines in the image
	drawn_img = lsd.drawSegments(img,lines)

	#Show image
	cv2.imshow("LSD",drawn_img )
	cv2.waitKey(0)
