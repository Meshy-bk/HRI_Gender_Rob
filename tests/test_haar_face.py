import cv2
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
cascade_path = str(BASE / "models" / "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise Exception(f"Failed to load Haar cascade from: {cascade_path}")

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

    cv2.imshow("Haar Face Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
