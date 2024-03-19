import time

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult


class GestureDetector(object):
    def __init__(self):
        self.model = self.__build_model_mediapipe()

    @staticmethod
    def __build_model_mediapipe():
        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        recognizer = vision.GestureRecognizer.create_from_options(options)
        return recognizer

    def __detect(self, bgr):
        # Image must convert from RGB to BGR
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr)
        result = self.model.recognize(mp_image)
        return mp_image, result

    # def __draw_gesture_batch(self, mp_images, results):
    #     for mp_image, result in zip(mp_images, results):
    #         return self.__draw_gesture(mp_image, result)

    @staticmethod
    def __draw_gesture(mp_image, result):
        image = mp_image.numpy_view()
        annotated_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        label = "Undetected"

        if len(result.gestures) > 0:
            top_gesture = result.gestures[0][0]
            label = '{}: {:.2f}'.format(top_gesture.category_name, top_gesture.score)

            hand_landmarks = result.hand_landmarks[0]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        cv2.putText(
            annotated_image,
            label,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            org=(20, 50),
            color=(0, 255, 0),
            thickness=2
        )

        annotated_image = cv2.resize(annotated_image, (720, 450), interpolation=cv2.INTER_LINEAR)

        return annotated_image

    def get_gesture(self, bgr):
        mp_image, result = self.__detect(bgr)
        return self.__draw_gesture(mp_image, result)


if __name__ == '__main__':
    video = cv2.VideoCapture()
    # Cam Marketing
    # video.open("rtsp://admin:Facenet2022@192.168.1.2:554/cam/realmonitor?channel=1&subtype=1")

    # Cam AI
    video.open("rtsp://admin:abcd1234@192.168.1.12:554/cam/realmonitor?channel=1&subtype=1")

    # Cam Integrated
    # video.open(0)

    # Video
    # video.open('./data/video/video_2024-03-13_16-27-50.mp4')

    if not video.isOpened():
        raise IOError("Cannot open webcam")

    fps = video.get(cv2.CAP_PROP_FPS)
    print('frames per second =', fps)

    model = GestureDetector()

    while True:
        t0 = time.time()
        ret, frame = video.read()
        # res, encoded = cv2.imencode('.jpg', frame)
        # frame_mjpeg = encoded.tobytes()
        if ret:
            t1 = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            annotated_image = model.get_gesture(frame)
            t2 = time.time()
            print(f'Time to process frame: {(t2 - t1):.2f} seconds')
            cv2.imshow('out', annotated_image)
            cv2.waitKey(1)
        else:
            break
