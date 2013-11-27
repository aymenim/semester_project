'''
@author Aymen Ibrahim
'''

import cv2 as cv
import numpy as np
import random ,time , math
from svm_train import *


model = SVM()
model.load('plates_svm.dat')
strt = time.time()
im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\test\Image001.jpg")
h , w = im.shape[:2]
print im.shape
# cap = cv.VideoCapture(1)
# while True:
	
# 	ret , im = cap.read()

gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

gray = cv.blur(gray,(7,7))
gray = cv.Sobel(gray,cv.CV_8U,1,0,ksize=3,scale=1,delta=0, borderType=cv.BORDER_DEFAULT)

rt, gray  = cv.threshold(gray,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY )

element = cv.getStructuringElement(cv.MORPH_RECT,(26,3))

gray = cv.morphologyEx(gray,cv.MORPH_CLOSE ,element)


contours ,hierarchy = cv.findContours(gray.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)


def verify_sizes(mr):
	x, y, w, h = cv.boundingRect(mr)
	error = 0.1
	#TODO aspect ratio of the plate for Ethiopia 
	aspect = 3
	#TODO min and max area for the plate with aspect * pixels
	p_min = 15 * aspect * 15 
	p_max = 125 * aspect * 125

	#respect ratio
	r_min = aspect - aspect*error
	r_max = aspect + aspect*error

	#contour area
	area = w * h
	try:
		r = float(w) / float(h)
	except:
		return False
	# print "r" , r
	if (r<1):
		return False
		# r = 1/r

	if (area < p_min or area > p_max) and (r < r_min or r > r_max):

		return False
	else:
		return True


rects = []
for cont in contours:
	if   verify_sizes(cont):
		rects.append(cont)

i = 0
for cont in rects:
	mr = cv.minAreaRect(cont)
	x, y, w12, h12 = cv.boundingRect(cont)
	# img_rotated = None
	try:
		r = mr[1][0] / mr[1][1]
		angle = mr[2]
	except:
		continue
	if r < 1:
		angle = angle + 90
	# if int(mr[2]) == 0 :

	rotmat = cv.getRotationMatrix2D(mr[0], angle ,1)
	img_rotated = cv.warpAffine(im, rotmat ,(w,h))
	img_crop = cv.getRectSubPix(img_rotated , (w12,h12) ,mr[0])
	result_resized = cv.resize ( img_crop ,(288,66),fx = 0 , fy = 0 , interpolation=cv.INTER_CUBIC)
	result_resized = cv.equalizeHist(cv.cvtColor(result_resized, cv.COLOR_BGR2GRAY))
	# hi = cv.cvtColor(result_resized , cv.COLOR_BGR2HSV)
	# hi_split = cv.split(hi)
	# hi_split[2] = cv.equalizeHist(hi_split[2])
	# hi = cv.merge(hi_split)
	# result_resized = cv.cvtColor(hi, cv.COLOR_HSV2BGR)


	sample = preprocess_hog([result_resized])
	digit = model.predict(sample)
	cv.rectangle(im, (x, y), (x+w12, y+h12), (255, 0, 0))

	if digit:
		cv.rectangle(im, (x, y), (x+w12, y+h12), (0, 255, 0))

		cv.putText(im, '%d'%mr[2], (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
		# cv.imshow('test1',img_rotated)
		print 'test_sliced'+ str(i) ,digit
		cv.imshow('test_sliced'+ str(i),result_resized)
	# cv.imwrite('test_sliced'+ str(i)+'.jpg',result_resized)
	i = int(i) + 1

print "finished in" , str(time.time() - strt) , "sec"
# cv.imshow('mask',mask)

# cv.drawContours(im,contours,-1,(0,0,255))
# cv.imshow('gray',gray)

cv.imshow('test',im)
0xFF & cv.waitKey()#(10)


cv.destroyAllWindows()