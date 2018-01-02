# Auther : Rishikesh
# Date	: 29 Dec 2017
# import the necessary packages
from collections import OrderedDict
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import numpy as np
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def turn_aspect_ratio(x1,x2,x3):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(x1, x2)
	B = dist.euclidean(x2, x3)

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	#C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	# ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return A/B

def open_mouth_detection(x1,x2,x3,x4):
	A = dist.euclidean(x1, x2)
	B = dist.euclidean(x3, x4)

	return A/B

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


cap = cv2.VideoCapture(0)
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
event = "none"
event2 = "none"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    rame = imutils.resize(frame, width=450)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(rame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array of face
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)

    	# 0.6 right and 1.6 left threshold
    	ratio = turn_aspect_ratio(shape[1],shape[28],shape[17])
    	lips_ratio = open_mouth_detection(shape[62],shape[66],shape[49],shape[59])
    	if lips_ratio>0.32:
    		event2="Mouth Open"
    	else :
    		event2="Mouth Close"

    	cv2.line(rame, tuple(shape[62]), tuple(shape[66]), (180, 42, 220), 2)
    	cv2.line(rame, tuple(shape[49]), tuple(shape[59]), (180, 42, 220), 2)
    	cv2.line(rame, tuple(shape[1]), tuple(shape[28]), (19, 199, 109), 2)
    	cv2.line(rame, tuple(shape[28]), tuple(shape[17]), (19, 199, 109), 2)
    	if ratio<0.6:
    		event="Right turn"
    	elif ratio>1.6:
    		event="Left turn"
    	else :
    		event="none"
    	# convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	cv2.rectangle(rame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    	# show the face number
    	cv2.putText(rame, "Face #{}".format(i + 1), (x - 10, y - 10),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	cv2.putText(rame, "Ratio: {}--{}".format(event,event2), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
    	for (x, y) in shape:
    		cv2.circle(rame, (x, y), 1, (0, 0, 255), -1)



    # Display the resulting frame
    cv2.imshow('frame',rame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
