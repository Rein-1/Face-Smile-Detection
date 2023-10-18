# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:52:53 2023

@author: Rein (Mark Renier Mercado)
"""

import cv2

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start video capture ('0' for the default camera)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)

    # Draw rectangle around smiles
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display output
    cv2.imshow('Video', frame)

    # Stop if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
video_capture.release()
cv2.destroyAllWindows()
