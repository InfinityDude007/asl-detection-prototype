import pickle
import cv2
import mediapipe as mp
import numpy as np


# mediapipe objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


# load stored model artifact
model_dict = pickle.load(open("./artifacts/model.p", "rb"))
model = model_dict["model"]


# setup video capture feed
capture = cv2.VideoCapture(0)

while True:

    # setup processing for frames from video feed
    ret, frame = capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    # auxiliary arrays for data and drawing coordinates
    data_aux = []
    x_aux = []
    y_aux = []


    # iteratively process a frame from capture feed and draw landmarks on top of it
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )


        # iteratively process a frame from capture feed, calculate and store its landmarks
        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):

                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                data_aux.append(x)
                data_aux.append(y)
                
                x_aux.append(x)
                y_aux.append(y)


        # use model to classify frame (only if feature length from the frame matches the training shape)
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = prediction[0]


            # draw model prediction onto frame
            x1 = int(min(x_aux) * W)
            y1 = int(min(y_aux) * H)
            x2 = int(max(x_aux) * W)
            y2 = int(max(y_aux) * H)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                4
            )

            cv2.putText(
                frame,
                predicted_char,
                (x1, y1-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )


    # display video feed and allow user to quit
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# release video capture feed
capture.release()
cv2.destroyAllWindows()
