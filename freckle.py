import cv2
import numpy as np

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

if __name__ == '__main__':
	img = cv2.imread('freckles1.jpg')
	faceDetect(img)