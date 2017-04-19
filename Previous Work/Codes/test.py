import cv2
import numpy as np
import math


img = cv2.imread('sari2.jpg',1)
img1 = cv2.imread('sari2.jpg',0)


'''kernel = np.ones((3,3),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(img,100,750,apertureSize=3)

edges = cv2.GaussianBlur(edges, (3,3), 200)


minLineLength = 1500
maxLineGap = 20
lines = cv2.HoughLinesP(edges,1,np.pi/180,4,minLineLength,maxLineGap)

h=np.zeros((img.shape[0],img.shape[1]))
p=np.zeros((img.shape[0],img.shape[1]))

for i in range(0,lines.shape[0]):
	for x1,y1,x2,y2 in lines[i]:
		try:
			temp=abs(math.atan((x2-x1)/float(y2-y1)))
			if(not (temp>=0 and temp<1.2 or temp<3.14 and temp>1.9)):
				cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
				cv2.line(p,(x1,y1),(x2,y2),(255,255,255),1)
		except:
			continue	


'''
p=np.zeros((img.shape[0],img.shape[1]))
#ret,thresh = cv2.threshold(img,127,255,0)
im2, contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(p, contours, -1, (255,255,255), 3)

cv2.imshow('result', p)
cv2.waitKey(0)
cv2.destroyAllWindows()
