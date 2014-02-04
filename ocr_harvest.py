import cv2 as cv
import numpy as np
import random ,time , math
import os
PLATE_DIR = r"SVM_data\sliced_images\plate"
def verify_sizes(pic):
    cols  ,rows  = pic.shape
	# char size
    aspect = 45.0 / 77.0
    charAspect =  cols / rows
    error = 0.35
    minHeight = 10
    maxHeight = 28

    minAspect = 0.2
    maxAspect = aspect + aspect * error

    # area of pixels
    area = np.count_nonzero(pic)

    #bb area
    bbArea = cols * rows

    # % of pixxels in area
    percPixels = area / bbArea

    if  rows >= minHeight and rows < maxHeight:
        return True
    else:
        return False
def projected_histogram(img , type = True):
    rows , cols = img.shape[:2]
    if type:
        sz = rows
    else:
        sz = cols
        img = np.rot90(img) 

    
    mhist = [np.count_nonzero(img[x]) for x in xrange(sz)]

    max_mhist = max(mhist)

    mod =  [mhist[i]/float(max_mhist) for i in xrange(sz)]

    

    # minV , maxV , minL , maxL = cv.minMaxLoc(mhist) 
    # cv.normalize

    return mod
         
def feature(img):

    low_sz = 10
    vhist = projected_histogram(img , True)
    hhist = projected_histogram(img, False)

    low = cv.resize( img, (low_sz,low_sz))

    out = [low[i][j] for i in xrange(low_sz) for j in xrange(low_sz)]

    return vhist + hhist + out

j = 0

for i in os.listdir(PLATE_DIR):
	print "working on plate", i
	if i[-3:] != "jpg": continue
	im = cv.imread(PLATE_DIR + "\\"+i)
	im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	bin = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 11)


	contours, heirs = cv.findContours( bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	try: heirs = heirs[0]
	except: heirs = []

	for cnt, heir in zip(contours, heirs):
				_, _, _, outer_i = heir
				# print outer_i
				if outer_i >= 0:
					continue
				x, y, w, h = cv.boundingRect(cnt)
				#print cv.boundingRect(cnt)
				# print (im[y:y+h , x:x+w]).shape
				if verify_sizes(im[y:y+h , x:x+w]):
					cv.imwrite('OCR_data\small'+ str(j)+'.jpg',cv.resize( im[y:y+h , x:x+w], (45,77)))
					print "pass" , j
					j = int(j) + 1
            
# cv.drawContours(im,contours,-1,(0,255,255))
# cv.imshow("threshold" , im)
# cv.imshow("threshold bin" , bin)

cv.waitKey()

