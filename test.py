import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Check if the result is not None before printing
    if result is not None:
            print('gesture recognition result: {}'.format(result.gestures))
    else:
        # If no gesture is recognized, print a default message
        cv2.putText(output_image, 'No gesture recognized', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=VisionRunningMode.LIVE_STREAM,result_callback=print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)
frame_timestamp_ms = int(round(time.time() * 1000))
# Initialize the webcam
cap = cv2.VideoCapture("rtsp://admin:Facenet2022@192.168.1.2:554/cam/realmonitor?channel=1&subtype=1")

timestamp = 0

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Convert the frame to a format that the recognizer can process
        # (Note: You might need to adjust this part depending on how your recognizer expects the data)

        # Send live image data to perform gesture recognition
        recognizer.recognize_async(mp_image, timestamp)

        cv2.imshow("MediaPipe Model", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()