import cv2
import numpy as np
from pathlib import Path
import tensorflow.lite as tflite

BASE = Path(__file__).resolve().parent.parent

cascade_path = BASE / "models" / "haarcascade_frontalface_default.xml"
model_path   = BASE / "models" / "GenderClass_06_03-20-08.tflite"

# -------------------------
# LOAD HAAR
# -------------------------
face_cascade = cv2.CascadeClassifier(str(cascade_path))
if face_cascade.empty():
    raise Exception(f"Failed to load Haar cascade from: {cascade_path}")

# -------------------------
# LOAD GENDER MODEL
# -------------------------
interpreter = tflite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model input shape:", input_details[0]["shape"])
print("Model output shape:", output_details[0]["shape"])

IN_H = int(input_details[0]["shape"][1])
IN_W = int(input_details[0]["shape"][2])

# -------------------------
# CAMERA
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera not accessible")

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face = frame[y:y+h, x:x+w]

        # resize לפי מה שהמודל דורש
        face_resized = cv2.resize(face, (IN_W, IN_H))
        face_resized = face_resized.astype(np.float32) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], face_resized)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        score0 = float(output[0])
        score1 = float(output[1])

        # מי יותר גדול
        if score0 > score1:
            predicted = "CLASS 0"
        else:
            predicted = "CLASS 1"

        print("Raw model output:", output)
        print("score0:", score0, "score1:", score1, "->", predicted)
        print("-------------")

        cv2.putText(
            frame,
            f"{score0:.2f} / {score1:.2f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Haar + Gender Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
