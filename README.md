# Image-Compression-using-K-means algorithm

This project from unsupervised learning coursera

In this exercise, you will apply K-means to image compression.

In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green, and blue intensity values. This encoding is often referred to as the RGB encoding.
Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of colors to 16.
This reduction makes it possible to represent (compress) the photo efficiently.
Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image, you now need to store only the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).

In this part, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image.

Concretely, you will treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3- 3-dimensional RGB space.
Once you have computed the cluster centroids on the image, you will then use the 16 colors to replace the pixels in the original image.
![bird_small](https://github.com/user-attachments/assets/53f03ceb-1a5d-40d1-8712-d242a9de49fd)
