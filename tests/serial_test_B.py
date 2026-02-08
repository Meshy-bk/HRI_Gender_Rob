import serial
import time

PORT = "/dev/ttyUSB0"   # או /dev/ttyACM0 לפי מה שמופיע אצלך
BAUD = 115200           # לפעמים 9600; אם לא עובד ננסה לשנות

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# דוגמה: שולחים מחרוזות פקודה (תלוי בפרוטוקול של היד)
for cmd in [b"THUMB_UP\n", b"PINKY_UP\n", b"OPEN\n", b"CLOSE\n"]:
    print("sending:", cmd)
    ser.write(cmd)
    time.sleep(1)
