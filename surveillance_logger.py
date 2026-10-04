"""Task 8: webcam surveillance that saves snapshots and logs high-confidence events.

When a target class is detected with confidence > 0.65 the script:
  - saves the frame to alerts/YYYYMMDD_HHMMSS.jpg
  - appends a line to events.log (Timestamp, Class, Confidence, BBox)
  - shows a red "LOGGING EVENT" banner on the live stream

Keys:  q = quit
"""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

ALERT_DIR = "alerts"
LOG_FILE = "events.log"
TARGET_CLASSES = {"person", "cell phone"}
CONF_THRESHOLD = 0.65
COOLDOWN_SECONDS = 2.0      # minimum gap between two logged events
BANNER_SECONDS = 1.0        # how long the on-screen indicator stays up


def init_log():
    os.makedirs(ALERT_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w") as f:
            f.write("Timestamp,Class,Confidence,BBox(x1;y1;x2;y2)\n")


def unique_snapshot_path(stamp):
    path = os.path.join(ALERT_DIR, f"{stamp}.jpg")
    n = 1
    while os.path.exists(path):          # two events in the same second
        path = os.path.join(ALERT_DIR, f"{stamp}_{n}.jpg")
        n += 1
    return path


def log_event(frame, hits):
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(unique_snapshot_path(stamp), frame)
    with open(LOG_FILE, "a") as f:
        for name, conf, (x1, y1, x2, y2) in hits:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')},{name},{conf:.2f},{x1};{y1};{x2};{y2}\n")
    print(f"[{now:%H:%M:%S}] logged {len(hits)} detection(s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    init_log()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    last_logged = 0.0
    banner_until = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=0.25, verbose=False)[0]
            annotated = result.plot()

            hits = []
            for box in result.boxes:
                name = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if name in TARGET_CLASSES and conf > CONF_THRESHOLD:
                    hits.append((name, conf, tuple(map(int, box.xyxy[0].tolist()))))

            now = time.time()
            if hits and now - last_logged >= COOLDOWN_SECONDS:
                log_event(annotated, hits)
                last_logged = now
                banner_until = now + BANNER_SECONDS

            if now < banner_until:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 0, 255), -1)
                cv2.putText(annotated, "LOGGING EVENT", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Surveillance (q = quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
