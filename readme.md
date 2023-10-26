# OpenCV art 1 - Basics

## Business Objective

OpenCV (Open-Source Computer Vision Library) is an open-source library that includes several hundreds of computer vision algorithms. OpenCV has a modular structure, which means that the package includes several shared or static libraries. OpenCV is a huge open-source library for computer vision, machine learning, and image processing. It can process images and videos to identify objects, faces, or even the handwriting of a human. When integrated with various libraries, such as "NumPy," a highly optimized library for numerical operations, the number of weapons increases in your Arsenal, i.e., whatever operations one can do in NumPy can be combined with OpenCV.

---

## Data Description

In this project, we will use three sample images (jpg) and a video (mp4) as our input data and perform various operations on top of it.

---

## Aim

The project aims to understand OpenCV and build some applications of how to use the OpenCV library.

## Tech Stack

- Language: `Python`
- Libraries: `numpy`, `matplotlib`, `cv2(OpenCV)`

---

## Approach

1. Importing the required libraries.
2. Perform reading, writing, and displaying an image.
3. Perform arithmetic operations like addition and subtraction on images.
4. Create a function for color spacing and conversion.
5. Create a function for Image thresholding.
6. Create a function for Image smoothing.
7. Create a function for Morphological Transformation.
8. Perform edge detection of an image with Canny Edge Detection.
9. Perform template matching and multi-scale template matching.
10. Create a function for Hough Transformation.
11. Perform video processing.
12. Perform Harris Corner detection.
13. Perform feature Detection and Extraction using SIFT.
14. Create a function for feature matching with Flann and Brute force.
15. Perform face and eye detection.

---

## Modular Code Overview

Once you unzip the `modular_code.zip` file you can find the following folders within it.

1. **Input**
2. **src**
3. **Output**

---

### Input folder

It contains all the data that we have for analysis. Here we have three sample images and one video image for performing different operations using OpenCV.

---

### Source folder

This is the most important folder of the project. This folder contains all the modularized code for all the above steps in a modularized manner. This folder consists of:

- `Engine.py`
- `ML_Pipeline`

The `ML_pipeline` is a folder that contains all the functions put into different python files which are appropriately named. These python functions are then called inside the `engine.py` file.

---

### Output folder

The output folder contains output images generated after running all the functions created. There are approximately 25 output images and one data folder which has frames after video processing.

---

## Concepts Explored

1. Understand OpenCV.
2. Read, write and display images.
3. Perform arithmetic operations like addition and subtraction on images.
4. Color spacing and conversion.
5. Image thresholding.
6. Image smoothing.
7. Morphological Transformation.
8. Canny Edge Detection.
9. Template matching and multi-scale template matching.
10. Hough Transformation.
11. Video processing.
12. Harris Corner detection.
13. Feature Detection and Extraction using SIFT (Scale-Invariant Feature Transform).
14. Feature matching with Flann and Brute force.
15. Face and eye detection.

---

