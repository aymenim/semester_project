import os
import cv2 as cv
import numpy as np

from ocr_svm_train import *

def main():
	model = SVM()
	model.load('ocr_svm.dat')
	# model = MLP()
	# model.load('ocr_mlp.dat')
	# im = cv.imread(r"SVM_data\sliced_images\no-plate\test_slicedtest_image_120.jpg")
	im = cv.imread(r"C:\Users\Aymen Ibrahim\Documents\semester_project\OCR_data\11\small140.jpg")

	sample = preprocess_hog([im])
	digit = model.predict(sample)
	print digit


if __name__ == '__main__':
	main()