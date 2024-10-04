import cv2
import numpy as np
from datetime import datetime

# Load Haarcascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Function to preprocess the frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Histogram equalization for better contrast
    return gray

# Function to detect and draw rectangles around faces and eyes
def detect_and_draw(frame):
    gray = preprocess_frame(frame)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    detections = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
            detections.append((x, y, w, h, ex, ey, ew, eh))
    
    return frame, detections

# Function to add timestamp to the frame
def add_timestamp(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Capture frames from the camera
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Open a file to store detected objects data
with open("detected_faces_eyes.txt", "w") as file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and eyes, and draw rectangles
        frame, detections = detect_and_draw(frame)
        
        # Add timestamp to the frame
        frame = add_timestamp(frame)
        
        # Store detected faces and eyes data with timestamp
        for detection in detections:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp}, Face: ({detection[0]}, {detection[1]}, {detection[2]}, {detection[3]}), Eye: ({detection[4]}, {detection[5]}, {detection[6]}, {detection[7]})\n")
        
        # Write the frame into the file 'output.avi'
        out.write(frame)
        
        # Display the resulting frame
        cv2.imshow('Face and Eye Detection', frame)
        
        # Break the loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the capture and video writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

import os

if not os.path.exists('videos'):
    os.makedirs('videos')

out = cv2.VideoWriter('videos/output.avi', fourcc, 20.0, (640, 480))
