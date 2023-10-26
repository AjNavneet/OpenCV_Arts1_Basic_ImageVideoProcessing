import cv2  # Import OpenCV library
import numpy as np  # Import numpy library
import os  # Import the os module for file operations
from .admin import output_folder  # Import the output folder path

class Smoothing:
    def __init__(self, path):
        """
        Initialize the Smoothing class with an image path.

        :param path: Path to the input image for smoothing operations.
        """
        self.image_path = path
        self.image = cv2.imread(self.image_path)

    def averaging_numpy(self):
        """
        Perform image smoothing using the numpy implementation of average blur in OpenCV.

        This method applies a 5x5 average filter using numpy and saves the smoothed image.
        """
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(self.image, -1, kernel)

        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "averaged_numpy.jpg"), dst)

    def average_bluring(self):
        """
        Perform image smoothing using the OpenCV average blur.

        This method applies a 5x5 average blur filter using OpenCV and saves the smoothed image.
        """
        blur = cv2.blur(self.image, (5, 5))
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self image)
        cv2.imwrite(os.path.join(output_folder, "averaged_blur.jpg"), blur)

    def gaussian_blur(self):
        """
        Perform image smoothing using Gaussian blur in OpenCV.

        This method applies a Gaussian blur filter with a kernel size of 5x5 and saves the smoothed image.
        """
        blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "gaussian_blur.jpg"), blur)

    def median_blur(self):
        """
        Perform image smoothing using median blur in OpenCV.

        This method applies a median blur filter with a kernel size of 5x5 and saves the smoothed image.
        It is highly effective in removing salt and pepper noise.
        """
        median = cv2.medianBlur(self.image, 5)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "median_blur.jpg"), median)

    def bilateral_blur(self):
        """
        Perform image smoothing using bilateral blur in OpenCV.

        This method applies bilateral filtering to the image, which is highly effective in noise removal while keeping edges sharp.
        Parameters 'd', 'sigmaColor', and 'sigmaSpace' control the filtering characteristics.
        """
        blur = cv2.bilateralFilter(self.image, 9, 75, 75)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "bilateral_blur.jpg"), blur)
