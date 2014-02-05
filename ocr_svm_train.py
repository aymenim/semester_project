import cv2 as cv
import numpy as np
import os
import random
import itertools as it
from numpy.linalg import norm


PLATE_DIR = r"OCR_data"

ESCAPE_DIR = [r"undefined"]
plates , labels = [] , []

class StatModel(object):
    class_n = 13
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


class MLP(StatModel):
    def __init__(self):
        self.params = dict( term_crit = (cv.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model = cv.ANN_MLP()
    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses
    def train(self, samples, responses):
        self.model = cv.ANN_MLP()
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, 100, self.class_n])
        self.model.create(layer_sizes)

        self.model.train(samples, np.float32(new_responses), None, params = self.params)
        

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


def preprocess_hog(plates):
    samples = []
    for img in plates:
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


def load_data():
    k = 0
    global plates , labels
    for i in os.listdir(PLATE_DIR):
    	print "working on Folder", i

        if os.path.isdir(PLATE_DIR + "\\" + i) and not ESCAPE_DIR.count(i):
            for j in os.listdir(PLATE_DIR+"\\"+i):
                print "\tworking on image ", PLATE_DIR+"\\"+i+"\\"+j
                plates.append(cv.imread(PLATE_DIR+"\\"+i+"\\"+j))
                labels.append(int(i))
                k += 1
    			
    print k

def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((14, 14), np.int32)
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
    return None #mosaic(4, vis)

def main():
    global plates , labels
    load_data()

    print 'preprocessing...'
    print len(plates)
    # # shuffle data
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(plates))
    plates = np.array(plates)
    labels = np.array(labels)

    print "shuffle: ",type(shuffle) , str(shuffle)
    print "plates", type(plates)
    plates, labels = plates[shuffle], labels[shuffle]

    samples = preprocess_hog(plates) # hog(plates)

    train_n = int(0.9*len(samples))
    # cv.imshow('test set', mosaic(4, plates))

    plates_train, plates_test = np.split(plates, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print 'training SVM...' 
    model = SVM(C=2.67, gamma=5.383 )#MLP() #
    print 'labels_train',type(labels_train[0])
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, plates_test, samples_test, labels_test)
    # cv.imshow('SVM test', vis)
    print 'saving SVM as "ocr_svm.dat"...'
    model.save('ocr_svm.dat')
    print 'saved SVM as "ocr_svm.dat"'
    # cv.waitKey(0)



if __name__ == '__main__':
    main()

	

