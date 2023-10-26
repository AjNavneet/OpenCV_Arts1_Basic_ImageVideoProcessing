import cv2  # Import OpenCV library
import sys  # Import sys module for system-related operations
import os  # Import the os module for file operations
from .admin import output_folder  # Import the output folder path

class ReadWriteDisplay:
    def __init__(self, image_path):
        """
        Initialize the ReadWriteDisplay class with an image path.

        :param image_path: Path to the image to be read, written, and displayed.
        """
        self.image_path = image_path

    def read(self):
        """
        Read the image from the provided image path.

        This method demonstrates reading images using cv2.imread() with various flags.
        """
        image = cv2.imread(self.image_path)
        shape = image.shape
        # shape goes by: (rows, cols, channels)
        print("Shape of the image:", shape)

        # Reading image with flags
        grayscale = cv2.imread(self.image_path, 0)
        # shape goes by: (rows, cols)
        shape = grayscale.shape
        print("Shape of the grayscale image:", shape)

    def write(self):
        """
        Write the image to the output folder.

        This method reads the image and then writes it to the specified output folder.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            sys.exit("Could not read the image.")
        cv2.imwrite(os.path.join(output_folder, "write_output.jpg"), image)  # Save the image

    def show(self):
        """
        Display the image using OpenCV window.

        This method displays the image in a window named "Display". Press the 'Esc' key to close the window.
        """
        image = cv2.imread(self.image_path)
        cv2.imshow("Display", image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
