import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

data = []
label = input("Enter the label for this recording (e.g., ONE, TWO, A, B): ")
frame_count = 0
max_frames = 1000

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from webcam")
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            data.append(landmarks + [label])
            frame_count += 1

            cv2.putText(frame, f'Captured: {frame_count}/{max_frames}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if frame_count >= max_frames:
                break

    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow('Hand Landmark Capture', resized_frame)

    if cv2.waitKey(1) & 0xFF == 27 or frame_count >= max_frames:
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv(f'{label}_dataset.csv', index=False)
print(f'Dataset for {label} saved as {label}_dataset.csv')
