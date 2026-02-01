import cv2
import numpy as np
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/emotion_model.h5", compile=False)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haar/haarcascade_frontalface_default.xml"
)

# Buffers
emotion_buffer = deque(maxlen=3)     # smoothing
history_buffer = deque(maxlen=150)    # ~5 sec history (30 FPS)

# FPS
prev_time = time.time()

cap = cv2.VideoCapture(0)

print("✅ Emotion Detector Running (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- LOW LIGHT CHECK ----------
    brightness = np.mean(gray_frame)
    low_light = brightness < 60

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]

        resized = cv2.resize(face, (64, 64))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 64, 64, 1)

        preds = model.predict(reshaped, verbose=0)[0]
        confidence = np.max(preds)
        if confidence < 0.45:
            continue
        top_idx = np.argmax(preds)
        emotion = EMOTIONS[top_idx]

        if emotion == "Neutral" and np.max(preds) < 0.55:
            continue


        # ---------- SMOOTHING ----------
        emotion_buffer.append(emotion)
        smooth_emotion = Counter(emotion_buffer).most_common(1)[0][0]

        history_buffer.append(smooth_emotion)

        # ---------- DRAW FACE BOX ----------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ---------- EMOTION TEXT ----------
        cv2.putText(
            frame,
            f"{smooth_emotion}",
            (x, y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ---------- CONFIDENCE BAR ----------
        bar_x = x
        bar_y = y - 20
        bar_width = int(w * confidence)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + 10), (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"{int(confidence * 100)}%",
            (x + w + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # ---------- FPS ----------
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ---------- LOW LIGHT WARNING ----------
    if low_light:
        cv2.putText(
            frame,
            "Low light may reduce accuracy",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ---------- EMOTION HISTORY ----------
    if len(history_buffer) > 0:
        recent = list(history_buffer)[-5:]
        history_text = " → ".join(recent)
        cv2.putText(
            frame,
            f"Recent: {history_text}",
            (10, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

    cv2.imshow("EmotionAI - Advanced Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

