import cv2

lk_params = dict(winSize=(15,15) , maxLevel = 2 , criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10), criteria= (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))
feature_params = dict(maxCorners=500,qualityLevel=0.01,minDistance=10)

class LKTracker(object):
	"""docstring for LKTracker"""
	def __init__(self, imnames):
		self.imnames = imnames
		self.features = []
		self.tracks = []
		self.current_frame = 0

		