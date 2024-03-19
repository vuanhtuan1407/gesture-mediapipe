import time

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))


def __build_model():
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    return recognizer


def __display_image_with_gesture_and_label(mp_image, result):
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

        # plt.title(f'{top_gesture.category_name}({top_gesture.score * 100}%)')
        # plt.imshow(annotated_image)
        # plt.show()

    cv2.putText(
        annotated_image,
        label,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        org=(20, 50),
        color=(0, 255, 0),
        thickness=2
    )
    cv2.imshow('out', annotated_image)
    cv2.waitKey(1)


def gesture_predict(frame):
    # Convert frame into right colors
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Build Model
    model = __build_model()

    # Load image (transfer from cv2 Image to mp Image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Predict
    result = model.recognize(image=mp_image)

    # Display image with prediction gesture and label
    __display_image_with_gesture_and_label(mp_image=mp_image, result=result)


if __name__ == '__main__':
    video = cv2.VideoCapture()
    # video.open("http://admin:Facenet2022@192.168.1.2:8001")
    # video.open(0)
    video.open('./data/video/video_2024-03-13_16-27-50.mp4')
    # time.sleep(2)
    if not video.isOpened():
        raise IOError("Cannot open webcam")

    fps = video.get(cv2.CAP_PROP_FPS)
    print('frames per second =', fps)

    minutes = 0
    seconds = 28
    frame_id = int(fps * (minutes * 60 + seconds))
    print('frame id =', frame_id)
    t0 = time.time()

    while True:
        t1 = time.time()
        # print(t1-t0)
        ret, frame = video.read()
        # res, encoded = cv2.imencode('.jpg', frame)
        # frame_mjpeg = encoded.tobytes()
        if ret:
            t1 = time.time()
            gesture_predict(frame)
            t2 = time.time()
            print('frame time =', t2 - t1)
        # elif ret and int((t1-t0)) % 3 != 0:
        #     continue
        else:
            break

    video.release()
    cv2.destroyAllWindows()
