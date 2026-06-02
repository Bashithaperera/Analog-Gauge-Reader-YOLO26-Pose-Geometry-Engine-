# Analog Gauge Reader (YOLO26-Pose + Geometry Engine)

A real-time computer vision pipeline that automates manual meter readings by converting analog dial positions into precise digital data streams. This system eliminates human error, digitizes legacy equipment metrics, and logs pressure/vacuum readings dynamically.

## Key Features
* **Dynamic Self-Calibration:** Tracks the needle pivot, tip, and scale markers (`Min` and `Max`) to handle camera movement, tilt, and rotation.
* **Clockwise Trigonometric Engine:** Seamlessly handles circular wrap-arounds and filters out dead-zone snapping behavior.
* **Dropout Stabilization:** Uses historical metric caching to prevent text flicker during extreme motion blur or frame dropouts.

---

## Prerequisites

Before running the script, ensure you have Python 3.8+ installed along with a CUDA-compatible environment (highly recommended for real-time performance on an NVIDIA GPU).

### Dependencies

Install the required Python packages using pip:

```bash
pip install ultralytics opencv-python numpy torch
```

## How to Run

Open gauge_reader.py and verify the execution configuration parameters at the bottom of the script match your target gauge bounds:
```
Python
if __name__ == "__main__":
    MODEL_WEIGHTS = "best.pt"
    INPUT_VIDEO   = "inference.mp4"
    
    # Configure your gauge scale limits here
    DIAL_MIN_VALUE = 0.0
    DIAL_MAX_VALUE = 10.0

```
Run the pipeline script from your terminal:
```
Bash
python gauge_reader.py
```
Press 'q' at any point on the keyboard to exit the active visualization video stream early.

## Output

The processed frame results containing vector tracking overlays and digital readouts will be compiled and automatically saved to:
final/final_gauge_output.mp4
