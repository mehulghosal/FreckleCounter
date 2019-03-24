import cv2 as cv
import numpy as np

img = cv.imread('freckles1.jpg')
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()