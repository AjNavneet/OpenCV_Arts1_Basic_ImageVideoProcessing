import cv2  # Import OpenCV library
import os  # Import the os module for file operations
from .admin import output_folder  # Import the output folder path

class VideoAnalytics:
    def __init__(self, video_path):
        """
        Initialize the VideoAnalytics class with the path to the video.

        :param video_path: Path to the video to be processed.
        """
        self.video_path = video_path

    def process(self):
        """
        Capture frames from the video and save them as individual image files.

        - `cv2.VideoCapture` takes the video path to be read. You can use `0` as an argument to directly feed camera input.
        - Frames are read one at a time from the video, and the `ret` variable is checked. If it's `True`, the frame is
          saved as an image file. If it's `False`, the loop breaks as there are no more frames.
        - Image files are saved in the specified output folder with filenames like 'frame0.jpg', 'frame1.jpg', etc.
        """
        cam = cv2.VideoCapture(self.video_path)
        basepath = os.path.join(output_folder, 'data')  # Set the base directory for saving frames
        try:
            if not os.path.exists(basepath):
                os.makedirs(basepath)  # Create the directory if it doesn't exist
        except OSError:
            print('Error: Creating directory of data')

        currentframe = 0
        while True:
            # Read one frame at a time
            ret, frame = cam.read()
            if ret:
                name = os.path.join(basepath, 'frame' + str(currentframe) + '.jpg')
                print('Creating...' + name)
                cv2.imwrite(name, frame)  # Save the frame as an image file
                currentframe += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()
