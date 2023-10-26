from .admin import output_folder  # Import the output folder path
import os  # Import the os module for file operations
import cv2  # Import OpenCV for image processing

class AirthmeticOperations:
    def __init__(self, path01, path02):
        """
        Initialize the AirthmeticOperations class with two image paths.

        :param path01: Path to the first image.
        :param path02: Path to the second image.
        """
        self.image_path_src = path01
        self.image_path_dest = path02

    def add(self):
        """
        Perform image addition and save the result.

        cv2.addWeighted(img1, wt1, img2, wt2, gammaValue)
        gammaValue: Measurement of light
        """
        image_src = cv2.imread(self.image_path_src)  # Read the first image
        image_dest = cv2.imread(self.image_path_dest)  # Read the second image

        # Resize both images to the same dimensions before adding
        # Take (1050, 1610) as the size we need to resize
        image_src = cv2.resize(image_src, (1050, 1610))
        image_dest = cv2.resize(image_dest, (1050, 1610))

        # Perform image addition with specified weights and save the result
        weightedSum = cv2.addWeighted(image_src, 0.5, image_dest, 0.4, 0)
        cv2.imwrite(os.path.join(output_folder, "added.jpg"), weightedSum)

    def substract(self):
        """
        Perform pixel-wise subtraction of one image from another and save the result.
        """
        image_src = cv2.imread(self.image_path_src)  # Read the first image
        image_dest = cv2.imread(self.image_path_dest)  # Read the second image

        # Resize both images to the same dimensions before subtracting
        # Take (1050, 1610) as the size we need to resize
        image_src = cv2.resize(image_src, (1050, 1610))
        image_dest = cv2.resize(image_dest, (1050, 1610))

        # Perform pixel-wise subtraction and save the result
        remaining_image = cv2.subtract(image_src, image_dest)
        cv2.imwrite(os.path.join(output_folder, "subtracted.jpg"), remaining_image)
