'''
@author Aymen Ibrahim
'''

import cv2 as cv
import numpy as np

im = cv.imread('test_image_1.jpg')

# print im.shape
# cap = cv.VideoCapture(0)


gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

gray = cv.blur(gray,(3,3))

gray = cv.Sobel(gray,cv.CV_8U,1,0,ksize=3,scale=1,delta=0)

rt, gray  = cv.threshold(gray,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY)

element = cv.getStructuringElement(cv.MORPH_RECT,(17,3))

gray = cv.morphologyEx(gray,cv.MORPH_CLOSE ,element)

contours ,hierarchy = cv.findContours(gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

def verify_sizes(mr):
	error = 0.4
	#TODO aspect ratio of the plate for Ethiopia 
	aspect = 4.72
	#TODO min and max area for the plate with aspect * pixels
	p_min = 15 * aspect * 15 
	p_max = 125 * aspect * 125

	#respect ratio
	r_min = aspect - aspect*error
	r_max = aspect + aspect*error

	#contour area
	area = mr[0][0] * mr[0][1]
	r = mr[0][0] / mr[0][1]

	if (r<1):
		r = 1/r

		if (area < p_min or area > p_max) and (r < r_min or r > r_max):
			return False
		else:
			return True


rects = []
for cont in contours:
	mr = cv.minAreaRect(cont)
	if verify_sizes(mr):
		# print mr
		rects.append(mr)
	# print mr

# while True:
cv.imshow('test',gray)
0xFF & cv.waitKey()


cv.destroyAllWindows()