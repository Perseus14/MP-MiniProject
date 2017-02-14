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
	img = cv2.GaussianBlur(img, (3,3), 200)
	edges = cv2.Canny(img,100,750)
	plt.subplot(121), plt.imshow(tf_cv_plt(img)),plt.title("Original Image"),plt.xticks([]),plt.yticks([])
	plt.subplot(122), plt.imshow(edges,'gray'),plt.title("Edges"),plt.xticks([]),plt.yticks([])
	plt.tight_layout()
	plt.savefig(str(t)+"_g.jpg")
	plt.close()
	t+=1
