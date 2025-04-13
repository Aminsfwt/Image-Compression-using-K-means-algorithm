import numpy as np
import matplotlib.pyplot as plt
from kmeans import *

# Load an image of a bird
original_img = plt.imread('e:/ML/Intro to Deep Learning/Labs/Codes/Course 3/Week 1/Image Compresion/bird_small.png')

#transform the matrix original_img into a two-dimensional matrix.
# Normalize the image by Divide by 255 so that all values are in the range 0 - 1
original_img = original_img / 255
# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Run K-Means algorithm on this data
# values of K (no of clusters (colors)) and max_iters here & initial_centroids
K = 16                       
max_iters = 10                
initial_centroids = kMeans_init_centroids(X_img, K) 

# Run K-Means and return centroids (k) and clossest centroid for each cols of X_img (idx)
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

# Represent image in terms of indices
# X_recovered is a matrix of size m x 3 where each row is the RGB value of the pixel
X_recovered = centroids[idx, :] 
# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)

# Display original image and compressed image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

# Display original image
ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()

# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
plt.show()
