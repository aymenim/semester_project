'''
@author Aymen Ibrahim
'''

import cv2 as cv
import numpy as np
import random

im = cv.imread('test_image_3.jpg')
h , w = im.shape[:2]
# print im.shape
# cap = cv.VideoCapture(1)
# while True:
	
# 	ret , im = cap.read()

gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

gray = cv.blur(gray,(5,5))

gray = cv.Sobel(gray,cv.CV_8U,1,0,ksize=3,scale=1,delta=0, borderType=cv.BORDER_DEFAULT)

rt, gray  = cv.threshold(gray,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY )

element = cv.getStructuringElement(cv.MORPH_RECT,(17,3))

gray = cv.morphologyEx(gray,cv.MORPH_CLOSE ,element)

contours ,hierarchy = cv.findContours(gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

def verify_sizes(mr):
	print mr
	error = 0.4
	#TODO aspect ratio of the plate for Ethiopia 
	aspect = 4.7272
	#TODO min and max area for the plate with aspect * pixels
	p_min = 15 * aspect * 15 
	p_max = 125 * aspect * 125

	#respect ratio
	r_min = aspect - aspect*error
	r_max = aspect + aspect*error

	#contour area
	area = mr[1][0] * mr[1][1]
	try:
		r = mr[1][0] / mr[1][1]
	except:
		return False
	print "r" , r
	if (r<1):
		r = 1/r

		if (area < p_min or area > p_max) and (r < r_min or r > r_max):
			return False
		else:
			return True


rects = []
for cont in contours:
	mr = cv.minAreaRect(cont)
	if not verify_sizes(mr):
		# print mr
		rects.append(cont)

	# print mr

cv.drawContours(im,rects,-1,(0,0,255))

for rect in rects:
	mr = cv.minAreaRect(rect)
	print "mr >>", mr
	# cv.circle(im,(int(mr[0][1]) , int(mr[0][0])),50,(0,255,0))
	minSize = min(mr[0])
	minSize = int(minSize - minSize*0.5)
	# random.seed()
	mask = np.zeros((h+2, w+2), np.uint8)
	seed_pt = (int(mr[0][0]) , int(mr[0][1]))
	fixed_range = True
	connectivity = 4
	lo = 39
	hi = 38
	newMaskVal = 255
	numSeeds = 10
	mask[:] = 0
	flags = connectivity #+ cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY + (newMaskVal << 8)
	flags |= cv.FLOODFILL_FIXED_RANGE
	# for x in xrange(numSeeds):
	rand_pt = seed_pt #(int( seed_pt[0] + random.random() * minSize)  ,int( seed_pt[1] + random.random() * minSize) )
	print "<><><><>>> random",rand_pt
	try:
		cv.circle(im, rand_pt, 2, (0,  255 ,0), -1)
		cv.floodFill(im, mask, rand_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
	except :
		pass
# cv.drawContours(im,contours,-1,(255,0,0)) #blue
cv.drawContours(im,rects,-1,(0,0,255)) #red



cv.imshow('test',im)
0xFF & cv.waitKey()#(10)


cv.destroyAllWindows()