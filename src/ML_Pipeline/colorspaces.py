import cv2  # Import OpenCV library
import numpy as np  # Import numpy for array operations
from .admin import output_folder  # Import the output folder path
import os  # Import the os module for file operations

def check_color_spaces():
    """
    Print a list of color space flags available in OpenCV.
    """
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print(flags)

class ColorSpaces:
    """
    The ColorSpaces class provides methods for working with color spaces, primarily converting between BGR, Gray, and HSV color spaces.
    """
    def __init__(self, path):
        """
        Initialize the ColorSpaces class with an image path.

        :param path: Path to the input image.
        """
        self.image_path = path
        self.image = cv2.imread(self.image_path)  # Read the input image

    def convert_bgr_hsv(self):
        """
        Convert the BGR image to HSV color space and save the result.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)  # Convert to HSV
        cv2.imwrite(os.path.join(output_folder, "bgr2hsv.jpg"), hsv)  # Save the result

    def convert_bgr_gray(self):
        """
        Convert the BGR image to grayscale and save the result.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        cv2.imwrite(os.path.join(output_folder, "bgr2gray.jpg"), gray)  # Save the result

    def track_blue(self):
        """
        Track the blue color in the BGR image. Convert to HSV and use a mask to detect blue.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)  # Convert to HSV

        lower_blue = np.array([25, 50, 50])  # Define lower blue color range
        upper_blue = np.array([130, 255, 255])  # Define upper blue color range

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image to detect blue
        res = cv2.bitwise_and(self.image, self.image, mask=mask)

        cv2.imwrite(os.path.join(output_folder, "detected_blue.jpg"), res)  # Save the result
