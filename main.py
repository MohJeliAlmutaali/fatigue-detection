import dlib
import cv2
import numpy as np
import streamlit as st
import pygame
import time

# Fungsi konversi format landmarks
def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

predictor_path = "./shape_predictor_68_face_landmarks.dat" # Sesuaikan path Anda
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Inisialisasi antrian deret waktu
queue = np.zeros(30, dtype=int).tolist()

# Inisialisasi pygame untuk memainkan suara
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# Fungsi utama untuk memproses video dan menampilkan hasilnya
def main():
    st.title("Deteksi Kelelahan dengan Dlib dan OpenCV")

    # Membaca frame video dari webcam
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for i, rect in enumerate(rects):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

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
            queue = queue[1:] + [flag]

            if sum(queue) > len(queue) / 2:
                cv2.putText(frame, "WARNING !", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                # Hidupkan alarm jika tidak sedang berbunyi
                if not pygame.mixer.get_busy():
                    alarm_sound.play()
            else:
                cv2.putText(frame, "SAFE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                # Matikan alarm jika berbunyi
                if pygame.mixer.get_busy():
                    pygame.mixer.stop()

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == 27:  # Tekan "Esc" untuk keluar
            break

    video_capture.release()

if __name__ == "__main__":
    main()
