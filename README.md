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
Note: Open gauge_reader.py and verify the execution configuration parameters at the bottom of the script match your target gauge bounds (minimum and maximum marker values)

To run the pipeline out of the box, organize your project files as follows:

├── final/                      # Auto-generated output directory
├── best.pt                     # Custom trained YOLO26-Pose weights
├── gauge_reader.py             # Main production Python file
└── inference.mp4               # Input test video
```
