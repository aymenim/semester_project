import os
import cv2 as cv
import numpy as np
import itertools as it
from numpy.linalg import norm
bin_n = 16 # Number of bins

plates , labels = [] , []

PLATE_DIR = r"SVM_data\sliced_images\plate"
NO_PLATE_DIR=r"SVM_data\sliced_images\no-plate"

def deskew(img):
	m = cv.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11'] / m['mu02']
	# M = np.float32([1,skew,-0.5*])
	M = np.float32([[1, skew, -0.5*288*skew], [0, 1, 0]])
	img = cv.warpAffine(img,M,(288, 66),flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
	return img
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv.SVM_RBF,
                            svm_type = cv.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv.SVM()

    def train(self, samples, responses):
        self.model = cv.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs)#, pad)
    return np.vstack(map(np.hstack, rows))

def load_data():
    global plates , labels
    for i in os.listdir(PLATE_DIR):
        print "working on plate", i
        plates.append(cv.imread(PLATE_DIR + "\\"+i))
        labels.append(1)
        

    for i in os.listdir(NO_PLATE_DIR):
        print "working on Non plate", i
        plates.append(cv.imread(NO_PLATE_DIR + "\\"+i))
        labels.append(0)
    print "final size: " + str(len(plates))
    #plates = np.array(plates)
    plates = np.array(plates)
    print len(plates), type(plates[0])
    labels = np.float32(labels)
    print "type ", type(plates)
def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

    vis = []
    for img, flag in zip(digits, resp == labels):
        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(4, vis)

def main():
    global plates , labels
    load_data()

    # print 'preprocessing...'
    # shuffle data
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(plates))
    print "shuffle: ",type(shuffle) , str(shuffle)
    plates, labels = plates[shuffle], labels[shuffle]

    samples = preprocess_hog(plates) # hog(plates)

    train_n = int(0.9*len(samples))
    cv.imshow('test set', mosaic(4, plates))

    plates_train, plates_test = np.split(plates, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, plates_test, samples_test, labels_test)
    cv.imshow('SVM test', vis)
    print 'saving SVM as "plates_svm.dat"...'
    model.save('plates_svm.dat')
    print 'saved SVM as "plates_svm.dat"'
    cv.waitKey(0)




# img = cv.imread('test_sliced0.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # rt, img  = cv.threshold(img,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY )

# cv.imshow('pre' , img)
# # cv.imshow('post' , hog (img))
# print hog(img)
# cv.waitKey()

if __name__ == '__main__':
    main()
