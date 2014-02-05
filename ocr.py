from operator import itemgetter, attrgetter
import cv2 as cv
import numpy as np
import random ,time , math
from ocr_svm_train import *

def verify_sizes2(pic):
    w  , h  = pic.shape

    error = 0.1
    #TODO aspect ratio of the plate for Ethiopia 
    aspect = 2


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

def verify_sizes(pic):
    cols  ,rows  = pic.shape
    
    minHeight = 20
    maxHeight = 60

    minWidth = 12
    maxWidth = 40

    if  cols >= minHeight and cols < maxHeight and rows >= minWidth and rows < maxWidth:
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
        

class CharacterPic(object):
    def __init__(self , pic , shape , val):
        self.pic = pic
        self.shape = shape # x ,y , w , h 
        self.val = val   

def compare_character(first , second):
    return first.shape[0] - second.shape[0]
chars = []
def main(im):
    global chars
    model = SVM()
    model.load('ocr_svm.dat')
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
        # print cv.boundingRect(cnt)
        # print (im[y:y+h , x:x+w]).shape
        if verify_sizes(im[y:y+h , x:x+w]):
            cv.rectangle(im, (x, y), (x+w, y+h), (255,0, 255))
            cv.imshow("small"+str(i) , im[y:y+h , x:x+w] )
            # cv.imwrite('OCR_data\small'+ str(i)+'.jpg',cv.resize( im[y:y+h , x:x+w], (10,10)))
            char = cv.resize( im[y:y+h , x:x+w], (45,77))
            # print str(char)
            sample = preprocess_hog([char])
            digit = model.predict(sample)
            print digit , "small"+str(i)
            chars.append(CharacterPic(char,cv.boundingRect(cnt) , digit))

        i = int(i) + 1
                
    # cv.drawContours(im,contours,-1,(0,255,255))
    cv.imshow("threshold" , im)
    cv.imshow("threshold bin" , bin)

    tmp = ""
    for s in  sorted(chars, cmp=compare_character):
        x =  s.val[0]
        if x < 10:
            tmp = tmp + str(int(x))
        elif x == 11 :
            tmp += "A"
        elif x == 12:
            tmp += " code2 "

        elif x == 13:
            tmp += " AA "

    print tmp
    # cv.waitKey()
    
if __name__ == '__main__':
    im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\semester_project\test_sliced0.jpg")
    cv.imshow("im", im)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    main(im)
    # print len(chars)
    cv.waitKey()

