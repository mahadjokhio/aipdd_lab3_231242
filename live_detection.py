"""Task 7: real-time webcam object detection with an FPS counter.

Keys:  q = quit    s = save the current frame to captures/
"""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Live YOLO detection on a webcam")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights file")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    return p.parse_args()


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def save_frame(frame, folder="captures"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
    cv2.imwrite(path, frame)
    print("Saved", path)


def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check the camera index or close other apps using it.")

    fps = 0.0
    try:
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame, stopping.")
                break

            result = model(frame, conf=args.conf, verbose=False)[0]
            annotated = result.plot()

            # smooth the FPS value so it doesn't flicker
            instant = 1.0 / max(time.perf_counter() - start, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
            draw_fps(annotated, fps)

            cv2.imshow("Live Detection (q = quit, s = save)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_frame(annotated)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
