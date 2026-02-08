from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)
ch = 0  # נתחיל מערוץ 0, אחר כך ממפים לכל אצבע

for angle in [0, 45, 90, 135, 180, 90]:
    print("channel", ch, "->", angle)
    kit.servo[ch].angle = angle
    time.sleep(1)
