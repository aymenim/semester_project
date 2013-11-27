import os
import cv2 as cv
import numpy as np

from svm_train import *

def main():
	model = SVM()
	model.load('plates_svm.dat')
	# im = cv.imread(r"SVM_data\sliced_images\no-plate\test_slicedtest_image_120.jpg")
	im = cv.imread(r"SVM_data\sliced_images\plate\test_slicedtest_image_90.jpg")

	sample = preprocess_hog([im])
	digit = model.predict(sample)
	print digit


if __name__ == '__main__':
	main()