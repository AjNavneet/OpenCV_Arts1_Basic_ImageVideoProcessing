import cv2  # Import OpenCV library
from .admin import output_folder  # Import the output folder path
import os  # Import the os module for file operations

class CannyEdgeDetection:
    def __init__(self, path):
        """
        Initialize the CannyEdgeDetection class with an image path.

        :param path: Path to the input image for edge detection.
        """
        self.image_path = path

    def canny_edge(self):
        """
        Perform Canny edge detection on the input image.

        Second and third arguments are our minVal and maxVal.
        minVal and maxVal are used for Hysteresis Thresholding.
        """
        img = cv2.imread(self.image_path, 0)  # Read the input image in grayscale
        edges = cv2.Canny(img, 50, 200)  # Perform Canny edge detection
        cv2.imwrite(os.path.join(output_folder, "edges.jpg"), edges)  # Save the detected edges
