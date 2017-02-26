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

def hough_transform(img,edges):
	lines=cv2.HoughLines(edges,1,np.pi/180,0)
	try:
		for i in range(0,5):
			for rho,theta in lines[i]:
			#if( theta > 0.09 and theta < 1.48 or theta < 3.14 and theta > 1.66): 
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				cv2.line(img,(x1,y1),(x2,y2),(0,0,255),6)
			#cv2.line(h,(x1,y1),(x2,y2),(255,255,255),6)
	except:
		pass;
file_Name = "../../Data/input.pkl"
fileObj=open(file_Name,'rb')
b=pickle.load(fileObj)
t=1
for img in b:
	#img = cv2.GaussianBlur(img, (5,5), 200)
	edges = cv2.Canny(img,50,200)
	img_c = img.copy()
	hough_transform(img_c,edges);
	plt.subplot(131), plt.imshow(tf_cv_plt(img)),plt.title("Original Image"),plt.xticks([]),plt.yticks([])
	plt.subplot(132), plt.imshow(edges,'gray'),plt.title("Edges"),plt.xticks([]),plt.yticks([])
	plt.subplot(133), plt.imshow(tf_cv_plt(img_c)),plt.title("Hough"),plt.xticks([]),plt.yticks([])
	plt.tight_layout()
	#plt.show()	
	plt.savefig(str(t)+"_g.jpg")
	plt.close()
	t+=1
