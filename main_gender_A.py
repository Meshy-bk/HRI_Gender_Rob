import cv2
import numpy as np
import time
import sqlite3
from datetime import datetime

from adafruit_servokit import ServoKit
import tflite_runtime.interpreter as tflite

# =========================
# CONFIG
# =========================
ENABLE_HAND = True          # False = דיבאג בלי להזיז סרווים
DB_PATH = "gender_log.db"

# class0 -> female, class1 -> male
CLASS0_LABEL = "female"
CLASS1_LABEL = "male"

MARGIN = 0.10        # 0.08-0.15 לפי יציבות בפועל
STABLE_N = 5
COOLDOWN_FRAMES = 20

CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "GenderClass_06_03-20-08.tflite"

# =========================
# DB (SQLite)
# =========================
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS detections ( id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, gender TEXT NOT NULL, conf REAL, score0 REAL, score1 REAL, diff REAL, action TEXT)""")
conn.commit()

def db_log(gender, conf, score0, score1, diff, action):
    cur.execute("INSERT INTO detections (ts, gender, conf, score0, score1, diff, action) VALUES (?, ?, ?, ?, ?, ?, ?)", (datetime.now().isoformat(), gender, conf, score0, score1, diff, action))
    conn.commit()

# =========================
# SERVO CONFIG
# =========================
kit = ServoKit(channels=16)

THUMB_CH  = 0
INDEX_CH  = 1
MIDDLE_CH = 2
RING_CH   = 3
PINKY_CH  = 4

OPEN  = 20
CLOSE = 160

def clamp_angle(a):
    return max(0, min(180, int(a)))

def set_fingers(thumb, index, middle, ring, pinky):
    if not ENABLE_HAND:
        print(f"[DEBUG] set_fingers: {thumb},{index},{middle},{ring},{pinky}")
        return
    kit.servo[THUMB_CH].angle  = clamp_angle(thumb)
    kit.servo[INDEX_CH].angle  = clamp_angle(index)
    kit.servo[MIDDLE_CH].angle = clamp_angle(middle)
    kit.servo[RING_CH].angle   = clamp_angle(ring)
    kit.servo[PINKY_CH].angle  = clamp_angle(pinky)

def neutral():
    set_fingers(CLOSE, CLOSE, CLOSE, CLOSE, CLOSE)

def thumb_up():
    set_fingers(OPEN, CLOSE, CLOSE, CLOSE, CLOSE)

def pinky_up():
    set_fingers(CLOSE, CLOSE, CLOSE, CLOSE, OPEN)

# =========================
# LOAD CASCADE + MODEL
# =========================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Haar cascade not loaded. Check haarcascade_frontalface_default.xml path")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_shape = input_details[0]["shape"]
IN_H = int(in_shape[1])
IN_W = int(in_shape[2])

print("Model input shape:", in_shape)
print("Model output shape:", output_details[0]["shape"])
print("Mapping: class0=female, class1=male (verified by our tests)")
print("Keys: q=quit, n=neutral, t=thumb_up, p=pinky_up")

def preprocess_face(face_bgr):
    face = cv2.resize(face_bgr, (IN_W, IN_H))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def predict_gender(face_bgr):
    x = preprocess_face(face_bgr)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_details[0]["index"])[0]  # (2,)
    score0 = float(y[0])  # class0
    score1 = float(y[1])  # class1
    diff = abs(score0 - score1)

    if diff < MARGIN:
        return "unknown", max(score0, score1), score0, score1, diff

    if score0 >= score1:
        return CLASS0_LABEL, score0, score0, score1, diff   # female
    else:
        return CLASS1_LABEL, score1, score0, score1, diff   # male

# =========================
# CAMERA LOOP + STABILITY
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try VideoCapture(1) or check /dev/video0")

last_gender = None
stable_count = 0
cooldown = 0

neutral()
time.sleep(0.5)
db_log("system", None, None, None, None, "start")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    gender_now = None
    conf_now = None
    score0 = None
    score1 = None
    diff = None

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        face = frame[y:y+h, x:x+w]
        gender_now, conf_now, score0, score1, diff = predict_gender(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText( frame, f"{gender_now} conf={conf_now:.2f} s0={score0:.2f} s1={score1:.2f} diff={diff:.2f}",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),2,)

    if cooldown > 0:
        cooldown -= 1

    if gender_now in ("male", "female") and cooldown == 0:
        if gender_now == last_gender:
            stable_count += 1
        else:
            last_gender = gender_now
            stable_count = 1

        if stable_count >= STABLE_N:
            if gender_now == "male":
                print("Male stable -> THUMB UP")
                thumb_up()
                db_log(gender_now, conf_now, score0, score1, diff, "thumb_up")
            else:
                print("Female stable -> PINKY UP")
                pinky_up()
                db_log(gender_now, conf_now, score0, score1, diff, "pinky_up")

            stable_count = 0
            cooldown = COOLDOWN_FRAMES

    cv2.imshow("Gender Detection (Option A)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("n"):
        neutral()
        db_log("manual", None, None, None, None, "neutral")
    if key == ord("t"):
        thumb_up()
        db_log("manual", None, None, None, None, "thumb_up")
    if key == ord("p"):
        pinky_up()
        db_log("manual", None, None, None, None, "pinky_up")

cap.release()
cv2.destroyAllWindows()
neutral()
db_log("system", None, None, None, None, "stop")
conn.close()
