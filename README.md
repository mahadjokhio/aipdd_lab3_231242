# Lab 03: Computer Vision Prototyping & Object Detection Pipeline

AI Project Design and Development (AI-316) | Mahad, 231242

## Setup
```
pip install -r requirements.txt
```

## Files
| File | Tasks |
|------|-------|
| `lab03_tasks_1_to_6.ipynb` | Tasks 1 to 6 (preprocessing, color-space thresholding, edge detection, YOLO benchmark, class filtering, ROI intrusion) |
| `live_detection.py` | Task 7: live webcam detection with FPS (`q` quits, `s` saves a frame) |
| `surveillance_logger.py` | Task 8: event-triggered snapshots in `alerts/` and entries in `events.log` |
| `images/` | Test images used by the notebook |
| `results/` | Output images saved by the notebook |

## Run
```
jupyter notebook lab03_tasks_1_to_6.ipynb
python live_detection.py
python surveillance_logger.py
```
