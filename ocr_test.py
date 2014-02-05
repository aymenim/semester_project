


import numpy as np
import cv2 as cv

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses
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
class LetterStatModel(object):
    class_n = 11
    train_ratio = 0.95

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

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


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv.ANN_MLP()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, 100, self.class_n])
        self.model.create(layer_sizes)

        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model.train(samples, np.float32(new_responses), None, params = params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)



img = r"C:\Users\Aymen Ibrahim\Documents\semester_project\OCR_data\7\small6.jpg"
model = MLP()
model.load('test.dat')
ret = feature(cv.cvtColor(cv.imread(img) , cv.COLOR_BGR2GRAY))
# t = str(ret).replace('[',"").replace(']',"")
print model.predict(np.array([np.array(ret)]))
print type(np.array(ret))
