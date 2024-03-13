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


def build_model(model_type: str = "mediapipe"):
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=1)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    return recognizer


def display_image_with_gesture_and_label(mp_image, result):
    image = mp_image.numpy_view()
    if len(result.gestures) == 0:
        print("Undetected gesture")
    top_gesture = result.gestures[0][0]
    hand_landmarks = result.hand_landmarks

    annotated_image = image.copy()

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

    plt.title(f'{top_gesture.category_name}({top_gesture.score * 100}%)')
    plt.imshow(annotated_image)
    plt.show()


def gesture_predict(image, model_type: str = "mediapipe"):
    # Build Model
    model = build_model(model_type=model_type)

    # Load image (transfer from cv2 Image to mp Image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Predict
    result = model.recognize(image=mp_image)

    # Display image with prediction gesture and label
    display_image_with_gesture_and_label(mp_image=mp_image, result=result)


if __name__ == "__main__":
    image_path = './data/photo_6293834231222221931_y.jpg'
    image = cv2.imread(image_path)
    gesture_predict(image)
