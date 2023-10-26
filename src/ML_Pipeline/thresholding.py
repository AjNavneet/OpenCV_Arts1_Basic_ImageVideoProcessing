import cv2  # Import OpenCV library
from .admin import output_folder  # Import the output folder path
import os  # Import the os module for file operations

class Thresholding:
    def __init__(self, path):
        """
        Initialize the Thresholding class with the path to the input image.

        :param path: Path to the input image for thresholding.
        """
        self.image_path = path
        self.image = cv2.imread(self.image_path, 0)  # Read the input image in grayscale

    def simple_thres(self):
        """
        Perform simple thresholding on the image.

        If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value.
        """
        ret, thresh1 = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_folder, "simple_thres.jpg"), thresh1)

    def adaptive_thres(self):
        """
        Perform adaptive thresholding on the image.

        This method is suitable when an image has different lighting conditions in different areas. The algorithm
        determines the threshold for a pixel based on a small region around it.

        Threshold methods:
        - cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighborhood area minus the constant C.
        - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a Gaussian-weighted sum of the neighborhood values minus
        the constant C.
        """
        img = cv2.medianBlur(self.image, 5)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(os.path.join(output_folder, "adaptive_thres.jpg"), th2)

    def otsu_binarization(self):
        """
        Perform Otsu's binarization on the image.

        Otsu's method determines an optimal global threshold value from the image histogram. It is useful for
        bimodal images with two distinct peaks in the histogram.
        """
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(output_folder, "otsu_binarized_thres.jpg"), th3)
