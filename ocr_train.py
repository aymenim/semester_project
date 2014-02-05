import cv2 as cv
import numpy as np
import os
import random

PLATE_DIR = r"OCR_data"

ESCAPE_DIR = [r"undefined"]

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
fk = []

for i in os.listdir(PLATE_DIR):
	print "working on Folder", i
	# fout = open('feature.dat','a')

	if os.path.isdir(PLATE_DIR + "\\" + i) and not ESCAPE_DIR.count(i):
		for j in os.listdir(PLATE_DIR+"\\"+i):
			print "\tworking on image ", PLATE_DIR+"\\"+i+"\\"+j
			ret = feature(cv.cvtColor(cv.imread(PLATE_DIR+"\\"+i+"\\"+j) , cv.COLOR_BGR2GRAY))
			fk.append(i+ ","+ str(ret).replace('[',"").replace(']',"")+'\n')
			# fout.write(i+ ","+ str(ret).replace('[',"").replace(']',"")+'\n')
			# fout.flush()
	# fout.close
random.shuffle(fk)
with open('feature.dat','a') as fout:
	for x in fk:
		fout.write(x)

	

