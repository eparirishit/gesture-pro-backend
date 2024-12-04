import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load the Random Forest model and labels
labels = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h", "i": "i", 
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "q", "r": "r", 
    "s": "s", "t": "t", "u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
    "1": "Back Space", "2": "Clear", "3": "Space", "4": ""
}

with open("./ASL_model.p", "rb") as f:
    model = pickle.load(f)

rf_model = model["model"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)

def process_frame_for_prediction(frame_content):
    """
    Process a single frame and predict the character.
    """
    global hands, rf_model, labels

    # Decode the frame content
    nparr = np.frombuffer(frame_content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame with Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_image = hands.process(frame_rgb)
    hand_landmarks = processed_image.multi_hand_landmarks

    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            x_coordinates = [landmark.x for landmark in hand_landmark.landmark]
            y_coordinates = [landmark.y for landmark in hand_landmark.landmark]
            min_x, min_y = min(x_coordinates), min(y_coordinates)

            normalized_landmarks = [
                (landmark.x - min_x, landmark.y - min_y)
                for landmark in hand_landmark.landmark
            ]
            sample = np.asarray(normalized_landmarks).reshape(1, -1)
            predicted_character = rf_model.predict(sample)[0]
            return labels[predicted_character]
    return ""
