import cv2, dlib, _thread
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
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

# im tired and nothing is working so im going to manually convert rgb --> hsv
# pass in numpy array with three values rep rgb vals
# returns 3 element list with hsv color represengin same color
def rgb2hsv(color):
	cPrime = color * (1/255)
	cmax = max(cPrime)
	cmin = min(cPrime)
	delta = cmax - cmin
	hsv = np.array([0,0,cmax], float)
	if cmax == cPrime[0]:
		hsv[0] = 60 * (((cPrime[1] - cPrime[2])/delta)%6)
	elif cmax == cPrime[1]:
		hsv[0] = 60 * (((cPrime[2] - cPrime[0])/delta) + 2)
	elif cmax == cPrime[2]:
		hsv[0] = 60 * (((cPrime[0] - cPrime[1])/delta) + 4)

	if not cmax == 0:
		hsv[1] = delta/cmax

	return hsv


def findFeatures(image):
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
			cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

	fill(image, shape, 42, 48) #right eye
	fill(image, shape, 48, 60) #mouth + lips
	fill(image, shape, 30, 36) #bottom of nose
	fill(image, shape, 17, 40, sub=True, l = [shape[17], shape[18], shape[19], shape[20],shape[21],shape[39], shape[40], shape[41], shape[36]]) # left eye
	fill(image, shape, 17, 40, sub=True, l = [shape[22],shape[23],shape[24], shape[25], shape[26], shape[45], shape[42]]) # left eye
 
	# display(image)
	return image

# draws polygon given set number of points [start-1, end]
# sub is for sublisting and splicing - so if if want to fill in the space between eyebores and eyes
def fill(image, shape, start, end, sub = False, l = []):
	if sub == False:
		feature = shape[start:end].reshape((-1,1,2))
		cv2.fillPoly(image, [feature], (0,255,255))
	else:
		feature = np.array(l).reshape((-1,1,2))
		cv2.fillPoly(image, [feature], (0,255,255))

	return feature

def colors(img):
	average = img.mean(axis=0).mean(axis=0)
	pixels = np.float32(img.reshape(-1, 3))

	n_colors = 5
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS

	_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	_, counts = np.unique(labels, return_counts=True)
	dominant = palette[np.argmax(counts)]

	avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

	indices = np.argsort(counts)[::-1]   
	freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
	rows = np.int_(img.shape[0]*freqs)

	dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
	for i in range(len(rows) - 1):
	    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

	fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12,6))

	ugh = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(dominant)
	hold = cv2.cvtColor(ugh, cv2.COLOR_RGB2HSV)

	# ax0.imshow(hold)
	# # ax0.imshow(avg_patch)
	# ax0.set_title('dom but im fucking with it')
	# ax0.axis('off')
	# ax1.imshow(dom_patch)
	# ax1.set_title('Dominant colors')
	# ax1.axis('off')
	# ax2.imshow(img)
	# plt.show(fig)

	return dominant, palette.astype(int), ugh, hold

# looks through and counts the occurances of colors in a range around the dominant color
def a(img, color, ran):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# lowBound = 

if __name__ == '__main__':
	img1 = cv2.imread(cwd + '/images/control1.jpg', 1)
	img2 = cv2.imread(cwd + '/images/freckles1.jpg', 1)
	# faceDetect(img)
	img2 = findFeatures(img2)
	# _thread.start_new_thread(display, (img2,))
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
	dominant, palette, h1, h2 = colors(img2)
	dominant = dominant.astype(int)
	# for img2, dominant color is rgb(188, 135, 111); bgr(111, 135, 188)


	print(dominant)
	print("\n")
	print(palette)