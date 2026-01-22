import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
import time


# mediapipe objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)


# path for raw data and objects for landmark data
DATA_DIR = "./dataset/alphabet"
data = []
labels = []


# time elapsed
start_time = time.time()


# iterate over all directories, check if its valid, then iterate through contents, process, calculate and store landmark coordinates to objects
for dir_path in os.listdir(DATA_DIR):

    # checking if dir is valid
    if not os.path.isdir(os.path.join(DATA_DIR, dir_path)):
        continue
    
    # iterating over dir contents
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_path)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # processing images and calculating landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # storing landmark coordinates and labels for current image
                    data_aux.append(x)
                    data_aux.append(y)
                
            # storing all parsed coordinates and labels with sufficient features (avoids 'inhomogeneous shape' errors with numpy)
            if (len(data_aux) == 42):
                data.append(data_aux)
                labels.append(dir_path)


# dumping all data from coordinate and label objects into .pickle file for the model
f = open("data.pickle", "wb")
pickle.dump({
    "data": data,
    "labels": labels
    }, f)
f.close()


# confirm success and print elapsed time
print(f"\nLandmarks created, time elapsed: {(time.time() - start_time):.2f}s\n")
