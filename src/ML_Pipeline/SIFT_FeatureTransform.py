import cv2  # Import OpenCV library
import os  # Import the os module for file operations
from .admin import output_folder  # Import the output folder path

class SIFT:
    def __init__(self, path):
        """
        Initialize the SIFT class with an image path.

        :param path: Path to the input image for SIFT operations.
        """
        self.image_path = path
        self.image = cv2.imread(self.image_path)

    def drawKeypoints(self):
        """
        Detect and draw SIFT keypoints on the image.

        This method converts the image to grayscale, detects SIFT keypoints, and draws them on the image.
        The result is saved as "sift_keypoints.jpg".
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv2.drawKeypoints(gray, kp, self.image)
        cv2.imwrite(os.path.join(output_folder, "sift_keypoints.jpg"), img)

    def match(self, image_src, image_dest):
        """
        Perform feature matching using SIFT descriptors between two images.

        This method converts the source and destination images to grayscale, detects SIFT features, and matches descriptors.
        It then draws and saves the first 150 matches between the two images.

        :param image_src: Path to the source image.
        :param image_dest: Path to the destination image.
        """
        img1 = cv2.imread(image_src)
        img2 = cv2.imread(image_dest)

        # Convert images to grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        # Detect SIFT features in both images
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # Create a feature matcher
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        # Match descriptors of both images
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw and save the first 150 matches
        matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:150], img2, flags=2)
        cv2.imwrite(os.path.join(output_folder, "matched_imaged.jpg"), matched_img)
