import numpy as np
from PIL import Image
import cv2


image = np.asarray(Image.open('test.PNG'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pixelfreq = np.zeros(256)
im = Image.fromarray(image)
im.show()


# finds the frequency distribution of all pixels in the image in range [0, 255]
for p in range(len(image)):
    pixelfreq[image[p]] += 1

# this will take a threshold value and convert a grayscale image into a binary image depending on the threshold.
def binarize(img, thresh):
    
    binim = np.ones((1008,756))
    for u in range(1008):
        for v in range(756):
            if image[u][v] <= thresh:
                binim[u][v] = 255
            else:
                binim[u][v] = 0
    return binim

def entropy(freq, Threshold):
    N = len(image)**2
    bin = len(freq)
    eleft = 0.0
    eright = 0.0

#find p(i) distribution of a given pixel
    p = np.zeros(bin).astype(np.float)
    sumall = 0
    for i in range(bin):
        p[i] = freq[i]/N
        sumall += p[i]
# normalize the probability distribution
    for i in range(bin):
        p[i] /= sumall

# finds the entropy to the left of Threshold, and the right of Threshold, returns their sum
    for i in range(bin):
        if i < Threshold:
            if freq[i]>0:
                eleft += -(p[i])*np.log10(p[i])
        else:
            if freq[i] >0:
                eright += -(p[i])*np.log10(p[i])
    return eright + eleft


entrop = np.ones(256)
# stores all possible entropy values for Threshold between [0-255]
for k in range(256):
    entrop[k] = entropy(pixelfreq, k)

# finds the index that has the threshold value for the minimum entropy
min = entrop.argmin()
newmin = Image.fromarray(binarize(image, min))
newmin.show()
