import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("picture")
args = parser.parse_args()

start_time = time.time()

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

image = cv2.imread(args.picture)
blur = cv2.blur(image, (2,2))

# get rid of shadows
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
v.fill(200)
hsv_image = cv2.merge([h, s, v])

# blur image for edge detection
blurred = cv2.GaussianBlur(hsv_image, (3, 3), 0)
edges = auto_canny(blurred)
edges = cv2.bitwise_not(edges)

cv2.imshow('edges', edges)

# grass color array, MAY NEED TO BE ADJUSTED
lower = np.array([20,14,100])
upper = np.array([96,255,255])

# calculate mask and apply to image
mask = cv2.inRange(hsv_image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
grass = output
bw_image = cv2.cvtColor(grass, cv2.COLOR_HSV2BGR)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
new_image = bw_image[:]
threshold = 1
h,b = grass.shape[:2]    
(thresh, im_bw) = cv2.threshold(bw_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('cv2.threshold', im_bw)

# bitwise or with calculated edges and calculated mask
output =cv2.bitwise_or(im_bw, im_bw, mask = edges)
cv2.imshow('Output of bitwise_or', output)

# trying to smooth image, not really sure what im doing here
dilation_size = 4
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size +1, 2 * dilation_size + 1))
dst = cv2.morphologyEx(output, cv2.MORPH_CLOSE, element)

cv2.imshow('Finaloutput after morphology', dst)

print("%s seconds" % (time.time()-start_time))

k = cv2.waitKey(0)
if k =='q':
	cv2.destroyAllWindows()