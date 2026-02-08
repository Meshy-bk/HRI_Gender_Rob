import serial
import time
import sys
import glob

def auto_find_port():
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    return ports[0] if ports else None

def main():
    port = auto_find_port()
    if not port:
        print("No /dev/ttyACM* or /dev/ttyUSB* found. Option B not detected.")
        sys.exit(1)

    baud_candidates = [115200, 9600, 57600, 38400, 19200]
    print("Found port:", port)

    for baud in baud_candidates:
        try:
            print(f"Trying baud {baud}...")
            ser = serial.Serial(port, baud, timeout=0.5)
            time.sleep(2)  # gives controller time to reset
            ser.reset_input_buffer()

            print("Type commands and press Enter. (q to quit)")
            print("Examples: M , F , N , 0 90 180 90 0 , thumb 160 , etc.")
            while True:
                cmd = input(">> ").strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    ser.close()
                    return
                ser.write((cmd + "\n").encode("utf-8", errors="ignore"))
                time.sleep(0.1)
                # read any response
                resp = ser.read(ser.in_waiting or 1)
                if resp:
                    print("<<", resp.decode("utf-8", errors="ignore"))
        except Exception as e:
            print("Failed at baud", baud, ":", e)

if __name__ == "__main__":
    main()
