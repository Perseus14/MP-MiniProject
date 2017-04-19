import numpy as np
import cv2
im = cv2.imread('sari2.jpg')
img = cv2.imread('sari2.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

imgray = cv2.Canny(imgray, 500, 900)

ret, thresh = cv2.threshold(imgray, 100, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgray, contours, -1, (255,0,0), 1)

#cv2.drawContours(img, contours, 3, (0,255,0), 3)
#cnt = contours[4]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

#kernel = np.ones((3,3),np.uint8)
#img1 = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)

p=np.zeros((img.shape[0], img.shape[1]))                        
cv2.drawContours(p, contours, -1, (255,0,0), 1)

cv2.imshow("Img1", imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Img2", p)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output1.jpg", imgray)
cv2.imwrite("output2.jpg", p)
