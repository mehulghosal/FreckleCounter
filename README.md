# FreckleCounter
counts the number of freckles on a face


* first need to isolate face from image
* basically, remove features used to detect faces - eyes, mouth
	* using dlib: https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg

* first instinct was to use corner detection to maybe find freckles - lol horrifying
* look for color differences
	* so 4:39 am me is saying: scan the face
		* ignore hair color (established by sampling above forehead)
		* ignore facial features - marked in yellow (0,255,255)
		* establish average skin color
		* look for stuff darker than skin color