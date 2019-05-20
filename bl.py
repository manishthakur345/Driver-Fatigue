# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
import dlib
import cv2


def shape2NP(shape):
	point = np.zeros((68,2), dtype = 'int')

	for i in range(0,68):
		point[i] = (shape.part(i).x, shape.part(i).y)
	return point


def rect2Box(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)




def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively

(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

#print(face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"])
# start the video stream thread

vs = cv2.VideoCapture(args['video'])

time.sleep(1.0)
fctr = 0
fctrThresh = 15
flag = False
# loop over frames from the video stream
while True:
	ret, frame = vs.read()

	if not ret:
		break

	width = 600
	height = int ((width / (frame.shape[1])) * frame.shape[0])
	frame = cv2.resize(frame, (width, height))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gr = gray.copy()
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		(x, y, w, h) = rect2Box(rect)
		shape = predictor(gray, rect)
		shape = shape2NP(shape)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
		#cv2.drawContours(frame, shape[48:59], -1, (0, 0, 255), 1)
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			fctr += 1

		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			if fctr > fctrThresh:
				#cv2.putText(frame, 'Drowiness Alert', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
				flag = True
			else:
				fctr = 0;
			# reset the eye frame counter
			COUNTER = 0
		if flag :
			cv2.putText(frame, 'Drowiness Alert', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(3) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
