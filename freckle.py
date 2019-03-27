import cv2
import numpy as np

img = cv2.imread('freckles1.jpg', 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()