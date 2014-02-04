import cv2 as cv
import numpy as np
import random ,time , math


def verify_sizes(pic):
    cols  ,rows  = pic.shape
	# char size
    aspect = 45.0 / 77.0
    charAspect =  cols / rows
    error = 0.35
    minHeight = 15
    maxHeight = 28

    minAspect = 0.2
    maxAspect = aspect + aspect * error

    # area of pixels
    area = np.count_nonzero(pic)

    #bb area
    bbArea = cols * rows

    # % of pixxels in area
    percPixels = area / bbArea

    if percPixels < 0.8 and charAspect > minAspect and charAspect < maxAspect and rows >= minHeight and rows < maxHeight:
        return True
    else:
        return False
def projected_histogram(img , type= True):
    rows , cols , dim = img.shape
    if type:
        sz = rows 
    else:
        sz = cols
    mhist = np.zeros((1,sz),order = cv.CV_32F)

    for j in xrange(sz):

        if type :
            data =  img[j]
        else:
            data = img[j]

        mhist[j] = cv.count_nonzero(data)

    minV , maxV , minL , maxL = cv.minMaxLoc(mhist) 
    cv.normalize

    return mhist
            

	

im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\semester_project\SVM_data\sliced_images\plate\test_slicedImage0830.jpg")
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# im = cv.blur(im,(2,2))
# rt, bin  = cv.threshold(im,110,255, cv.THRESH_BINARY_INV )
bin = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 11)
# bin = cv.medianBlur(bin, 3)

contours, heirs = cv.findContours( bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
try: heirs = heirs[0]
except: heirs = []
i = 0
for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            # print outer_i
            if outer_i >= 0:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            print cv.boundingRect(cnt)
            # print (im[y:y+h , x:x+w]).shape
            # if verify_sizes(im[y:y+h , x:x+w]):
            cv.imshow("small"+str(i) , im[y:y+h , x:x+w] )
            cv.imwrite('OCR_data\small'+ str(i)+'.jpg',cv.resize( im[y:y+h , x:x+w], (10,10)))

            cv.rectangle(im, (x, y), (x+w, y+h), (255,0, 255))

            i = int(i) + 1
            
# cv.drawContours(im,contours,-1,(0,255,255))
cv.imshow("threshold" , im)
cv.imshow("threshold bin" , bin)

cv.waitKey()

