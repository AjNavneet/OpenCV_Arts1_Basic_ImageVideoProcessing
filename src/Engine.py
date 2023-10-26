from ML_Pipeline.airthmetic_operations import *
from ML_Pipeline.colorspaces import *
from ML_Pipeline.corner_detection import *
from ML_Pipeline.edge_detection import *
from ML_Pipeline.face_mouth_detection import *
from ML_Pipeline.hough_line_circle_detection import *
from ML_Pipeline.matcher import *
from ML_Pipeline.morphology import *
from ML_Pipeline.read_write import *
from ML_Pipeline.admin import input_folder
from ML_Pipeline.SIFT_FeatureTransform import *
from ML_Pipeline.smoothing import *
from ML_Pipeline.template_matching import *
from ML_Pipeline.thresholding import *
from ML_Pipeline.video_processing import *
import os

# Initialize the image path for reading and writing
image_path = os.path.join(input_folder, "./test.jpg")

# Create an object for reading and writing images
read_write_obj = ReadWriteDisplay(image_path)
read_write_obj.show()  # Display the image
read_write_obj.read()  # Read the image
read_write_obj.write()  # Write the image

# Arithmetic operations on images
image_path_src = os.path.join(input_folder, "test.jpg")
image_path_dest = os.path.join(input_folder, "test02.jpg")
airthmetic_obj = AirthmeticOperations(image_path_src, image_path_dest)
airthmetic_obj.add()  # Perform addition
airthmetic_obj.substract()  # Perform subtraction

# Color spaces and changing color spaces
image_path_src = os.path.join(input_folder, "blue_cap.jpg")
color_spaces_obj = ColorSpaces(image_path_src)
color_spaces_obj.convert_bgr_gray()  # Convert to grayscale
color_spaces_obj.convert_bgr_hsv()  # Convert to HSV color space
color_spaces_obj.track_blue()  # Track blue color in the image

# Image thresholding
image_path_src = os.path.join(input_folder, "blue_cap.jpg")
thres_obj = Thresholding(image_path_src)
thres_obj.simple_thres()  # Simple thresholding
thres_obj.adaptive_thres()  # Adaptive thresholding
thres_obj.otsu_binarization()  # Otsu's binarization

# Smoothing images
image_path_src = os.path.join(input_folder, "test.jpg")
smoothing_object = Smoothing(image_path_src)
smoothing_object.averaging_numpy()  # Averaging using NumPy
smoothing_object.median_blur()  # Median blur
smoothing_object.gaussian_blur()  # Gaussian blur
smoothing_object.bilateral_blur()  # Bilateral blur
smoothing_object.average_bluring()  # Average blurring

# Morphological transformations
image_path_src = os.path.join(input_folder, "test.jpg")
morpho_obj = Morphological(image_path_src)
morpho_obj.erode()  # Erosion
morpho_obj.dilate()  # Dilation
morpho_obj.opening()  # Morphological opening
morpho_obj.closing()  # Morphological closing
morpho_obj.get_structuring_element()  # Get structuring element

# Canny edge detection
image_path_src = os.path.join(input_folder, "test.jpg")
canny_edge_obj = CannyEdgeDetection(image_path_src)
canny_edge_obj.canny_edge()  # Perform Canny edge detection

# Template matching
image_path_src = os.path.join(input_folder, "test.jpg")
template_obj = TemplateMatching(image_path_src, image_path_src)
template_obj.template_matching()  # Template matching

# Multi-scale template matching
template_obj.multiscale_template_matching()  # Multi-scale template matching

# Hough Transforms for Line and Circle detection
image_path_src = os.path.join(input_folder, "test.jpg")
hough_object = HoughLineCircleDetection(image_path_src)
hough_object.line_detection()  # Detect lines using Hough transform
hough_object.circle_detection()  # Detect circles using Hough transform

# Video processing
video_path = os.path.join(input_folder, "test_video.mp4")
video_obj = VideoAnalytics(video_path)
video_obj.process()  # Process the video

# Harris corner detection
image_path_src = os.path.join(input_folder, "test.jpg")
corner_obj = CornerDetection(image_path_src)
corner_obj.detect()  # Detect corners using Harris corner detection

# SIFT feature detection
image_path_src = os.path.join(input_folder, "test.jpg")
sift_obj = SIFT(image_path_src)
sift_obj.drawKeypoints()  # Draw SIFT keypoints
sift_obj.match(image_path_src, image_path_src)  # Match SIFT features

# Feature matching using FLANN and brute-force orb
image_path_src = os.path.join(input_folder, "test.jpg")
image_path_dest = os.path.join(input_folder, "test.jpg")
matcher_obj = Matcher(image_path_src, image_path_dest)
matcher_obj.brute_force_matcher()  # Brute-force feature matching
matcher_obj.flann_matcher()  # FLANN feature matching

# Face and eye detection
face_detection_obj = detect()  # Perform face and eye detection
