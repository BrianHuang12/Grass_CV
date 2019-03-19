import cv2
import numpy as np
from matplotlib import pyplot as plt

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

image = cv2.imread("testimage3.jpg")

blur = cv2.blur(image, (2,2))

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('before v', hsv_image)
h, s, v = cv2.split(hsv_image)
v.fill(200)
hsv_image = cv2.merge([h, s, v])
blurred = cv2.GaussianBlur(hsv_image, (3, 3), 0)
edges = auto_canny(blurred)
# edges = cv2.bitwise_not(edges)

cv2.imshow('edges', edges)

lower = np.array([20,14,100])
upper = np.array([96,255,255])

mask = cv2.inRange(hsv_image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
output =cv2.bitwise_and(image, image, mask = edges)
output = cv2.bitwise_not(output)
cv2.imshow('output', output)

dilation_size = 4
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size +1, 2 * dilation_size + 1))
dst = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

cv2.imshow('mask', mask)
cv2.imshow('image no blur', hsv_image)
cv2.imshow('image', blur)
cv2.imshow('dst', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()