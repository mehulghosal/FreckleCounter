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
def display(img, title = "image"):
	cv2.imshow(title, img)
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
	fill(image, shape, 17, 40, sub=True, end1=22, start2=36) # left eyebrow to eye ???

	# display(image)
	return image

# draws polygon given set number of points [start-1, end]
# sub is for sublisting and splicing - so if i want to fill in the space between eyebores and eyes
def fill(image, shape, start, end, sub = False, end1 = 0, start2 = 0):
	if sub == False:
		feature = shape[start:end].reshape((-1,1,2))
		cv2.fillPoly(image, [feature], (0,255,255))
	else:
		feature = (shape[start:end1]).reshape((-1,1,2)) + (shape[start2:end]).reshape((-1,1,2))
		cv2.fillPoly(image, [feature], (0,255,255))


# not going to use this - im gonna keep it here for future reference on corners
def corners(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	dst = cv2.cornerHarris(gray,15,21,0.04)
	image[dst>0.01*dst.max()]=[0,0,255]
	display(image)
	return image

if __name__ == '__main__':
	img1 = cv2.imread(cwd + '/images/control1.jpg', 1)
	img2 = cv2.imread(cwd + '/images/freckles1.jpg', 1)
	# faceDetect(img)
	img1 = features(img1)
	display(img1)

	# img2 = corners(features(img2))
