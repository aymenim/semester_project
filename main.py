'''
@author Aymen Ibrahim
'''

import cv2 as cv
import numpy as np
import random ,time , math
import numpy
im = cv.imread('test_image_4.jpg')
h , w = im.shape[:2]
# print im.shape
# cap = cv.VideoCapture(1)
# while True:
	
# 	ret , im = cap.read()

gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

gray = cv.blur(gray,(7,7))

gray = cv.Sobel(gray,cv.CV_8U,1,0,ksize=3,scale=1,delta=0, borderType=cv.BORDER_DEFAULT)

rt, gray  = cv.threshold(gray,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY )

element = cv.getStructuringElement(cv.MORPH_RECT,(21,3))

gray = cv.morphologyEx(gray,cv.MORPH_CLOSE ,element)

contours ,hierarchy = cv.findContours(gray.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

def verify_sizes(mr):
	# print mr
	error = 0.1
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
	# print "r" , r
	if (r<1):
		r = 1/r

	if (area < p_min or area > p_max) and (r < r_min or r > r_max):
		return False
	else:
		return True


rects = []
for cont in contours:
	mr = cv.minAreaRect(cont)
	x, y, w1, h1 = cv.boundingRect(cont)
	cv.rectangle(im, (x, y), (x+w1, y+h1), (0, 0, 255))
	cv.putText(im, '%d'%mr[2], (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
	# cv.rectangle(im, (int(mr[0][0]), int(mr[0][1])), (int(mr[0][0])+int(mr[1][0]), int(mr[0][1])+int(mr[1][1])), (0, 255, 0))

	# box = cv.cv.BoxPoints(mr)
	# box = np.int0(box)
	# cv.drawContours(im,[box],0,(0,0,255),2)
	if   verify_sizes(mr):
		# print mr
		rects.append(cont)

	# print mr
for cont in rects:
	mr = cv.minAreaRect(cont)
	x, y, w, h = cv.boundingRect(cont)
	cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0))
	cv.putText(im, '%d w: %f %f'%(mr[2],mr[0][0] , mr[0][1]), (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)

result = None
# gray.copy(result)
# cv.drawContours(result,rects,-1,(0,0,255))

fixed_range = True
connectivity = 4
lo = 38
hi = 39
flags = connectivity #+  cv.FLOODFILL_FIXED_RANGE  +  
flags |=cv.FLOODFILL_MASK_ONLY
flags |= cv.FLOODFILL_FIXED_RANGE
flags |= (255 << 8)
print type(rects[0])
mask = np.zeros((h+2, w+2), np.uint8)
mask[:] = 0
# for rect in rects:
# 	mr = cv.minAreaRect(rect)
# 	# print "mr >>", mr
# 	cv.circle(im,(int(mr[0][1]) , int(mr[0][0])),3,(0,255,0))
# 	minSize = min(mr[0])
# 	minSize = minSize - minSize*0.5
# 	# random.seed()
# 	seed_pt = (int(mr[0][0]) , int(mr[0][1]))
# 	# print "seed_pt" , seed_pt

# 	for x in xrange(10):
# 		rand_pt =  (int(mr[0][0] + random.random() % minSize - (minSize/2)) , int(mr[0][1]+ random.random() % minSize- (minSize/2)))
# 		# print "r",rand_pt
# 		try:
# 			cv.circle(im, rand_pt, 2, (0, 0, 255), -1)
# 			ret , rect1 = cv.floodFill(im, mask, rand_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
# 			# if rect1 != (0,0,0,0):
# 			# 	print rect1
# 			# 	x, y, w2, h2 = rect1
# 			# 	cv.rectangle(im, (x, y), (x+w2, y+h2), (0, 0, 255))
# 		except Exception as e:
# 			print "something wrong!",e

	# interest_points = []
	# # print ">>",mask.item(10,10)
	# for y , row in enumerate(mask):
	# 	for x , px in enumerate(row):
	# 		if px == 255:
	# 			interest_points.append((y,x))
	# box = cv.minAreaRect(numpy.array([interest_points], dtype=numpy.int32))
	# print box

	# # # for m in mask:
	# # interest_points ,hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
	# # cv.drawContours(mask,contours,-1,(255,0,0)) #blue
	# # print type(interest_points[0])
	# # print len(interest_points)
	# # # min_rect = cv.minAreaRect(interest_points)

	# if verify_sizes(box):
	# 	r = box[1][0] / box[1][1]
	# 	angle = box[2]
	# 	if r < 1:
	# 		angle  = 90 + angle

	# 	img_rotated = None
	# 	rotmat = cv.getRotationMatrix2D(box[0], angle ,1)
	# 	img_rotated = cv.warpAffine(im, rotmat ,(w,h))
	# 	print type(img_rotated)
	# 	cv.imshow('test',img_rotated)
	# 	height = box[1][1] ; width = box[1][0]
	# 	if r < 1:
	# 		#swap
	# 		height = box[1][0]
	# 		width = box[1][1]

	# 		img_crop = cv.getRectSubPix(img_rotated , (width,height) ,box[0])
	# 		result_resized = cv.resize ( img_crop ,(33,144),fx = 0 , fy = 0 , interpolation=cv.INTER_CUBIC)


	



# mask  = cv.bitwise_not(deskew(mask))
# cv.drawContours(mask,rects,-1,(255,0,255)) #red

	# contours ,hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
	# cv.drawContours(mask,contours,-1,(255,0,0)) #blue
	

# min_rect = cv.minAreaRect(interest_points)

# cv.imshow('mask',mask)
cv.imshow('test',im)
0xFF & cv.waitKey()#(10)


cv.destroyAllWindows()