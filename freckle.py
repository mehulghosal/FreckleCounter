import cv2
import dlib
import numpy as np
from imutils import face_utils
from os import getcwd

cwd = getcwd()

# uses haar face detection built into cv2
# returns rectangle with coordinates of recognized face 
def faceDetect(img):
	haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
	# as scaleFactor gets smaller, the smaller the rectangle gets
	faces_rects = haar_cascade_face.detectMultiScale(img, scaleFactor = 1.005, minNeighbors = 5);
	print(faces_rects)
	for (x,y,w,h) in faces_rects:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	display(img)
	return faces_rects

# displays given image
def display(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def B(image):
	p = cwd + "/data/shape_predictor_68_face_landmarks.dat"
	gray = img
	detector = dlib.get_frontal_face_detector()
	rects = detector(gray, 0)
	predictor = dlib.shape_predictor(p)
	
	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

	display(img)


if __name__ == '__main__':
	img = cv2.imread(cwd + '/images/control1.jpg', 1)
	# faceDetect(img)
	B(img)