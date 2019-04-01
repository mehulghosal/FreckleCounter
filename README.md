# FreckleCounter
counts the number of freckles on a face


* first need to isolate face from image
* basically, remove features used to detect faces - eyes, mouth
	* using dlib: https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg
* look for color differences