from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np

# Load OpenCV pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process the video feed
def detect_expression():
    cap = cv2.VideoCapture(0)  # Start capturing from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Here you can integrate a model to predict emotions based on the face region

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(detect_expression(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'detection/index.html')
