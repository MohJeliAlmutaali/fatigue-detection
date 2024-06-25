import streamlit as st
import cv2
import dlib
import numpy as np
import pygame
import threading
from firebase_config import initialize_firebase
from firebase_admin import db  # Pastikan Anda telah mengimpor db

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Function to play the alarm sound
def play_alarm():
    pygame.mixer.music.load('alarm.mp3')
    pygame.mixer.music.play(-1)  # Play in a loop

# Function to stop the alarm sound
def stop_alarm():
    pygame.mixer.music.stop()

def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def main():
    st.title("Fatigue Detection System")

    initialize_firebase()

    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Set up webcam capture
    cap = cv2.VideoCapture(0)
    queue = np.zeros(30, dtype=int).tolist()
    alarm_playing = False

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for i, rect in enumerate(rects):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            landmarks = predictor(gray, rect)
            landmarks = landmarks_to_np(landmarks)

            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            d1 = np.linalg.norm(landmarks[37] - landmarks[41])
            d2 = np.linalg.norm(landmarks[38] - landmarks[40])
            d3 = np.linalg.norm(landmarks[43] - landmarks[47])
            d4 = np.linalg.norm(landmarks[44] - landmarks[46])
            d_mean = (d1 + d2 + d3 + d4) / 4
            d5 = np.linalg.norm(landmarks[36] - landmarks[39])
            d6 = np.linalg.norm(landmarks[42] - landmarks[45])
            d_reference = (d5 + d6) / 2
            d_judge = d_mean / d_reference
            print(d_judge)

            flag = int(d_judge < 0.25)
            queue = queue[1:len(queue)] + [flag]

            if sum(queue) > len(queue) / 2:
                cv2.putText(frame, "WARNING !", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if not alarm_playing:
                    threading.Thread(target=play_alarm).start()
                    alarm_playing = True
                    db.reference('status').set({'fatigue': True})
            else:
                cv2.putText(frame, "SAFE", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False
                    db.reference('status').set({'fatigue': False})

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        frame_placeholder.image(frame, channels="RGB")

    cap.release()
    stop_alarm()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
