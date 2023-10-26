import numpy as np  # Import numpy for array operations
import cv2 as cv  # Import OpenCV

class CornerDetection:
    def __init__(self, path):
        """
        Initialize the CornerDetection class with an image path.

        :param path: Path to the input image for corner detection.
        """
        self.image_path = path

    def detect(self):
        """
        Detect corners in the input image using the Harris corner detection algorithm.

        For arguments for cornerHarris:
          img - Input image. It should be grayscale and float32 type.
          blockSize - It is the size of the neighborhood considered for corner detection.
          ksize - Aperture parameter of the Sobel derivative used.
          k - Harris detector free parameter in the equation.
        """
        img = cv.imread(self.image_path)  # Read the input image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert the image to grayscale
        gray = np.float32(gray)  # Convert grayscale to float32

        dst = cv.cornerHarris(gray, 2, 3, 0.04)  # Perform Harris corner detection
        # Result is dilated for marking the corners, not important
        dst = cv.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark detected corners in red
        cv.imshow('dst', img)  # Display the image with detected corners
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()  # Close the window on pressing the 'Esc' key
