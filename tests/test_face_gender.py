import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow INFO logs

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ===== Paths =====
BASE = Path(__file__).resolve().parent.parent
FACE_MODEL = BASE / "models" / "haarcascade_frontalface_default.xml"
GENDER_MODEL = BASE / "models" / "GenderClass_06_03-20-08.tflite"

# ===== Load face detector (Haar) =====
face_cascade = cv2.CascadeClassifier(str(FACE_MODEL))
if face_cascade.empty():
    raise FileNotFoundError(f"Failed to load Haar cascade from: {FACE_MODEL}")

# ===== Load gender classification model (TFLite) =====
interpreter = tf.lite.Interpreter(model_path=str(GENDER_MODEL))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model input shape:", input_details[0]["shape"])
print("Model output shape:", output_details[0]["shape"])
print("Press 'q' to quit")

# ===== Camera =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Crop face
        face = frame[y:y + h, x:x + w]

        # Preprocess for gender model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=0)  # (1, 224, 224, 3)

        # Inference
        interpreter.set_tensor(input_details[0]["index"], face)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])  # (1, 2)

        # Convert output to label
        idx = int(np.argmax(output[0]))
        label = "Male" if idx == 1 else "Female"

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face + Gender Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
