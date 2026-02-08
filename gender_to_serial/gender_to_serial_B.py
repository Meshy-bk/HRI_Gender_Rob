import time
import cv2
import numpy as np
import serial
import glob
import tflite_runtime.interpreter as tflite

MODEL_PATH = "GenderClass_06_03-20-08.tflite"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

CMD_MALE = "THUMB"     # <-- להחליף למה שבאמת עובד אצלכם
CMD_FEMALE = "PINKY"   # <-- להחליף למה שבאמת עובד אצלכם

BAUD = 115200
STABLE_N = 5

def find_port():
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    return ports[0] if ports else None

def main():
    port = find_port()
    if not port:
        raise RuntimeError("No ttyACM/ttyUSB found (Option B not detected).")

    ser = serial.Serial(port, BAUD, timeout=0.5)
    time.sleep(2)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    last = None
    stable = 0
    cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        gender = None
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, fw, fh = faces[0]
            face = frame[y:y+fh, x:x+fw]
            face = cv2.resize(face, (w, h)).astype(np.float32) / 255.0
            face = np.expand_dims(face, axis=0)

            interpreter.set_tensor(inp["index"], face)
            interpreter.invoke()
            score = float(interpreter.get_tensor(out["index"]).flatten()[0])
            gender = "male" if score > 0.5 else "female"

            cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)
            cv2.putText(frame, f"{gender} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if cooldown > 0:
            cooldown -= 1

        if gender in ("male","female") and cooldown == 0:
            if gender == last:
                stable += 1
            else:
                last = gender
                stable = 1

            if stable >= STABLE_N:
                cmd = CMD_MALE if gender == "male" else CMD_FEMALE
                ser.write((cmd + "\n").encode())
                cooldown = 20  
                stable = 0

        cv2.imshow("Gender -> Serial", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
