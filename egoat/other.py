#############
# LIBRARIES
#############
import numpy as np
import cv2
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy.optimize import leastsq
from scipy.stats.mstats import gmean
from scipy.signal import argrelextrema
from scipy.stats import entropy
from scipy.signal import savgol_filter
import time


#############
# PROGRAM
#############
if __name__ == '__main__':

    #-----------------------------------
    ## 1. Create Chromaticity Vectors ##
    #-----------------------------------

    # Get Image
    start_time = time.time()
    img = cv2.imread('testimage3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    img = cv2.GaussianBlur(img, (5,5), 0)

    # Separate Channels
    r, g, b = cv2.split(img) 

    im_sum = np.sum(img, axis=2)
    im_mean = gmean(img, axis=2)

    # Create "normalized", mean, and rg chromaticity vectors
    #  We use mean (works better than norm). rg Chromaticity is
    #  for visualization
    n_r = np.ma.divide( 1.*r, g )
    n_b = np.ma.divide( 1.*b, g )

    mean_r = np.ma.divide(1.*r, im_mean)
    mean_g = np.ma.divide(1.*g, im_mean)
    mean_b = np.ma.divide(1.*b, im_mean)





    #-----------------------
    ## 2. Take Logarithms ##
    #-----------------------

    l_rg = np.ma.log(n_r)
    l_bg = np.ma.log(n_b)

    log_r = np.ma.log(mean_r)
    log_g = np.ma.log(mean_g)
    log_b = np.ma.log(mean_b)

    ##  rho = np.zeros_like(img, dtype=np.float64)
    ##
    ##  rho[:,:,0] = log_r
    ##  rho[:,:,1] = log_g
    ##  rho[:,:,2] = log_b

    rho = cv2.merge((log_r, log_g, log_b))

    #----------------------------
    ## 3. Rotate through Theta ##
    #----------------------------
    u = 1./np.sqrt(3)*np.array([[1,1,1]]).T
    I = np.eye(3)

    tol = 1e-15

    P_u_norm = I - u.dot(u.T)
    U_, s, V_ = np.linalg.svd(P_u_norm, full_matrices = False)

    s[ np.where( s <= tol ) ] = 0.

    U = np.dot(np.eye(3)*np.sqrt(s), V_)
    U = U[ ~np.all( U == 0, axis = 1) ].T

    # Columns are upside down and column 2 is negated...?
    U = U[::-1,:]
    U[:,1] *= -1.

    ##  TRUE ARRAY:
    ##
    ##  U = np.array([[ 0.70710678,  0.40824829],
    ##                [-0.70710678,  0.40824829],
    ##                [ 0.        , -0.81649658]])

    chi = rho.dot(U) 

    e = np.array([[np.cos(np.radians(np.linspace(1, 180, 180))), \
                   np.sin(np.radians(np.linspace(1, 180, 180)))]])

    gs = chi.dot(e)

    prob = np.array([np.histogram(gs[...,i], bins='scott', density=True)[0] 
                      for i in range(np.size(gs, axis=3))])

    eta = np.array([entropy(p, base=2) for p in prob])

    theta_min = np.radians(np.argmin(eta))

    print('Min Angle: ', np.degrees(theta_min))

    e = np.array([[-1.*np.sin(theta_min)],
                  [np.cos(theta_min)]])

    gs_approx = chi.dot(e)
    print('%s seconds' % (time.time()-start_time))
        # Visualize Grayscale Approximation --> DEBUGGING
    plt.imshow(gs_approx.squeeze(), cmap='gray')
    plt.title('Grayscale Approximation')
    plt.show()
