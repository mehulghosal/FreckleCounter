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

def features(image):
	p = cwd + "/data/shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	rects = detector(image, 0)
	predictor = dlib.shape_predictor(p)
	
	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
		for (x, y) in shape:
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

	fill(image, shape, 36, 42) #left eye
	fill(image, shape, 42, 48) #left eye
	fill(image, shape, 48, 60) #mouth + lips
	fill(image, shape, 30, 36) #bottom of nose

	display(image)
	return image

def fill(image, shape, start, end):
	feature = shape[start:end].reshape((-1,1,2))
	cv2.fillPoly(image, [feature], (0,255,255))

if __name__ == '__main__':
	img1 = cv2.imread(cwd + '/images/control1.jpg', 1)
	img2 = cv2.imread(cwd + '/images/freckles1.jpg', 1)
	# faceDetect(img)
	img1 = features(img1)
	img2 = features(img2)
