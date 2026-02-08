from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

# טווחים עדינים כדי לא לשבור משהו. תשני אם צריך.
SAFE_MIN = 20
SAFE_MAX = 160
SAFE_MID = 90

def set_angle(ch: int, angle: int):
    kit.servo[ch].angle = angle
    time.sleep(0.4)

def wiggle(ch: int):
    print(f"\nWIGGLE channel {ch}")
    set_angle(ch, SAFE_MID)
    set_angle(ch, SAFE_MIN)
    set_angle(ch, SAFE_MAX)
    set_angle(ch, SAFE_MID)

def main():
    print("Finger mapping tool (PCA9685/ServoKit)")
    print("Make sure HAND POWER is connected and stable.")
    print("Commands:")
    print("  w <ch>  -> wiggle channel (test movement)")
    print("  a <ch> <angle> -> set angle manually (0-180)")
    print("  scan    -> wiggle channels 0..15 one by one (you watch which finger moves)")
    print("  q       -> quit\n")

    while True:
        try:
            cmd = input(">> ").strip().lower()
        except KeyboardInterrupt:
            break

        if cmd in ("q", "quit", "exit"):
            break

        if cmd == "scan":
            for ch in range(16):
                wiggle(ch)
                ans = input("Did something move? (enter label e.g. thumb/index/middle/ring/pinky/none): ").strip().lower()
                print(f"channel {ch} -> {ans}")
            continue

        parts = cmd.split()
        if not parts:
            continue

        if parts[0] == "w" and len(parts) == 2:
            ch = int(parts[1])
            wiggle(ch)
            continue

        if parts[0] == "a" and len(parts) == 3:
            ch = int(parts[1])
            ang = int(parts[2])
            ang = max(0, min(180, ang))
            print(f"Set channel {ch} to {ang}")
            kit.servo[ch].angle = ang
            continue

        print("Unknown command. Try: w 0 | a 0 90 | scan | q")

if __name__ == "__main__":
    main()
