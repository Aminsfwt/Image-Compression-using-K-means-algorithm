import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *


#GRADED FUNCTION: find_closest_centroids
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i in range(X.shape[0]): # iterate over each example
        # Compute the distance between the example and each centroid
        distance = []
        for j in range(centroids.shape[0]):# iterate over each centroid
            # Compute the distance between the example and centroid j
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            """
            #A more verbose but equivalent way to write this would be:
            difference = X[i] - centroids[j]  # Vector difference
            squared_diff = difference ** 2    # Square each element
            sum_squares = np.sum(squared_diff)  # Sum the squares
            norm_ij = np.sqrt(sum_squares)    # Take the square root
            """
            distance.append(norm_ij)
            
        idx[i] = np.argmin(distance)    
    ### END CODE HERE ###
    
    return idx

# GRADED FUNCTION: compute_centpods
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis = 0)
    ### END CODE HERE ## 
    
    return centroids

# GRADED FUNCTION: run_kMeans
def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """  
    # Initialize values (extract rows and cols from the dataset)
    m, n = X.shape
    K = initial_centroids.shape[0] # number of clusters
    centroids = initial_centroids  # Initialize centroids
    idx = np.zeros(m)
    
    # Run K-Means
    for i in range(max_iters):  
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
             
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K) 
    return centroids, idx

# You do not need to modify this part

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples (shuffle the dataset row randomly)
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples from the shuffled dataset rows as centroids 
    centroids = X[randidx[:K]]
    
    return centroids