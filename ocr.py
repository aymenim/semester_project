import cv2 as cv
import numpy as np
import random ,time , math


im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\semester_project\SVM_data\sliced_images\plate\test_slicedtest_image_150.jpg")
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# rt, bin  = cv.threshold(im,110,255, cv.THRESH_BINARY_INV )
bin = cv.adaptiveThreshold(im, 155, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 11)
bin = cv.medianBlur(bin, 3)

contours, heirs = cv.findContours( bin.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
try: heirs = heirs[0]
except: heirs = []

for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(im, (x, y), (x+w, y+h), (0,0, 255))

cv.imshow("threshold" , im)
cv.imshow("threshold bin" , bin)

cv.waitKey()

