import cv2  # Import OpenCV library
import numpy as np  # Import numpy for array operations
from .admin import output_folder  # Import the output folder path
import os  # Import the os module for file operations

class Morphological:
    def __init__(self, path):
        """
        Initialize the Morphological class with an image path.

        :param path: Path to the input image for morphological operations.
        """
        self.image_path = path
        self.image = cv2.imread(self.image_path, 0)  # Read the input image in grayscale

    def erode(self):
        """
        Perform erosion, which erodes away the boundaries of the foreground object.

        Always try to keep the foreground in white.
        """
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(self.image, kernel, iterations=1)  # Apply erosion
        cv2.imwrite(os.path.join(output_folder, "erosed.jpg"), erosion)  # Save the eroded image

    def dilate(self):
        """
        Perform dilation, which increases the white region in the image or increases the size of the foreground object.
        """
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(self.image, kernel, iterations=1)  # Apply dilation
        cv2.imwrite(os.path.join(output_folder, "dilated.jpg"), dilation)  # Save the dilated image

    def opening(self):
        """
        Perform opening, which is erosion followed by dilation. It is useful for removing noise.
        """
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)  # Apply opening
        cv2.imwrite(os.path.join(output_folder, "opened.jpg"), opening)  # Save the opened image

    def closing(self):
        """
        Perform closing, which is dilation followed by erosion. It is useful for closing small holes inside
        the foreground objects or small black points on the object.
        """
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)  # Apply closing
        cv2.imwrite(os.path.join(output_folder, "closing.jpg"), closing)  # Save the closed image

    def get_structuring_element(self):
        """
        Get a structuring element for morphological operations if required.

        It provides examples of creating rectangular and elliptical structuring elements.
        """
        rect_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Create a rectangular structuring element
        print(rect_element)
        elliptical_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Create an elliptical structuring element
        print(elliptical_element)
