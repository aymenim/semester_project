import cv2 as cv
import numpy as np
import random ,time , math


def verify_sizes(pic):
	# char size
	aspect = 45.0 / 77.0
	

im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\semester_project\SVM_data\sliced_images\plate\test_slicedImage0830.jpg")
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# im = cv.blur(im,(2,2))
# rt, bin  = cv.threshold(im,110,255, cv.THRESH_BINARY_INV )
bin = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 11)
bin = cv.medianBlur(bin, 3)

contours, heirs = cv.findContours( bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
try: heirs = heirs[0]
except: heirs = []

for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            # print outer_i
            if outer_i >= 0:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            print cv.boundingRect(cnt)
            cv.rectangle(im, (x, y), (x+w, y+h), (255,0, 255))

cv.imshow("threshold" , im)
cv.imshow("threshold bin" , bin)

cv.waitKey()

