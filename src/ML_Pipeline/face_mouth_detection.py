import cv2 as cv  # Import OpenCV library
from .admin import output_folder, input_folder  # Import the output and input folders
import os  # Import the os module for file operations

def detectAndDisplay(frame, face_cascade, eyes_cascade):
    """
    Detect faces and eyes in a video frame using Haar Cascade Classifiers.

    :param frame: Input video frame.
    :param face_cascade: Haar Cascade Classifier for face detection.
    :param eyes_cascade: Haar Cascade Classifier for eyes detection.
    """
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    frame_gray = cv.equalizeHist(frame_gray)  # Equalize the histogram for better results

    # Detect faces using the face cascade
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)  # Draw an ellipse around the face
        faceROI = frame_gray[y:y + h, x:x + w]

        # In each detected face, detect eyes using the eyes cascade
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)  # Draw circles around the eyes

    cv.imshow('Capture - Face detection', frame)  # Display the frame with face and eye detection

def detect():
    # Load Haar Cascade Classifiers for eyes and faces
    eyes_cascade_name = os.path.join(input_folder, "haarcascade_eye.xml")
    face_cascade_name = os.path.join(input_folder, "haarcascade_frontalface_default.xml")

    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!) Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!) Error loading eyes cascade')
        exit(0)

    # Read the video stream from the default camera (ID 0)
    cap = cv.VideoCapture(0)
    if not cap.isOpened:
        print('--(!) Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()  # Read a frame from the video stream
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, face_cascade, eyes_cascade)  # Perform face and eye detection on the frame
        if cv.waitKey(10) == 27:
            break
