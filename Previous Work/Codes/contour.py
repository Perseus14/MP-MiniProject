import numpy as np
import cv2

im = cv2.imread('sari1.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,20,255,0)
cv2.imshow('original', im)
img = im[:,:]

'''#clean up noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)'''

'''#canny edge detection
edges = cv2.Canny(opening,50,200,apertureSize=3)
cv2.imshow('edges',edges)
img_color = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)'''

#contour drawing
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,255), 3)

cv2.imshow('binary',thresh)
cv2.imshow('contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('countour.jpg',img);
cv2.imwrite('binary.jpg',thresh);