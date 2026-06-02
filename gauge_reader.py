import os
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ultralytics import YOLO

class AnalogGaugeReader:
    """
    A real-time computer vision pipeline to digitize analog gauge readings
    using YOLOv8-Pose keypoint tracking and angular linear interpolation.
    """
    def __init__(
        self, 
        model_path: str, 
        min_value: float = 0.0, 
        max_value: float = 10.0,
        conf_threshold: float = 0.25
    ) -> None:
        """Initializes the gauge reader with model weights, scale thresholds, and configuration."""
        print(f"Loading YOLO-Pose model from: {model_path}...")
        self.model = YOLO(model_path)
        self.min_value = min_value
        self.max_value = max_value
        self.conf_threshold = conf_threshold
        
        # State tracking variables to mitigate frame dropouts/motion blur
        self.last_valid_value = 0.0
        self.last_valid_angle = 0.0

    @staticmethod
    def _get_absolute_angle(center: List[float], point: List[float]) -> float:
        """
        Calculates the absolute Cartesian angle of a vector stretching from center to point.
        Inverts the Y-axis calculation to compensate for standard image pixel space 
        where row indices increase downwards.
        """
        return math.atan2(center[1] - point[1], point[0] - center[0])

    def _process_geometry(
        self, 
        needle_center: List[float], 
        needle_tip: List[float], 
        min_marker: List[float], 
        max_marker: List[float]
    ) -> Tuple[float, float]:
        """
        Executes the clockwise trigonometric engine to determine relative needle sweep
        and interpolates the raw numeric metric value.
        """
        theta_min = self._get_absolute_angle(needle_center, min_marker)
        theta_max = self._get_absolute_angle(needle_center, max_marker)
        theta_needle = self._get_absolute_angle(needle_center, needle_tip)
        
        # Determine clock-wise sweep intervals using modulo 2PI boundary handling
        total_arc = (theta_min - theta_max) % (2 * math.pi)
        current_arc = (theta_min - theta_needle) % (2 * math.pi)
        
        # Bottom dead-zone clipping optimization
        if current_arc > total_arc:
            if (current_arc - total_arc) < ((2 * math.pi - total_arc) / 2):
                current_arc = total_arc  # Snap to maximum boundary
            else:
                current_arc = 0.0        # Snap to minimum boundary
                
        relative_angle_deg = math.degrees(current_arc)
        travel_ratio = current_arc / total_arc
        calculated_value = self.min_value + (travel_ratio * (self.max_value - self.min_value))
        
        return calculated_value, relative_angle_deg

    def process_video(self, source_path: str, output_dir: str = "final") -> str:
        """
        Streams, processes, overlays analytics graphics frame-by-frame,
        and saves the compiled final tracking stream output file.
        """
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f" Error: Unable to open or parse source video file at: {source_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_gauge_output.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing: {width}x{height} @ {fps} FPS | Target: {total_frames} Frames.")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            annotated_frame = frame.copy()
            
            # Predict single frame array instance
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, device=0)[0]
            
            needle_center: Optional[List[float]] = None
            needle_tip: Optional[List[float]] = None
            min_marker: Optional[List[float]] = None
            max_marker: Optional[List[float]] = None
            
            if results.boxes is not None and results.keypoints is not None and len(results.keypoints.xy) > 0:
                for idx, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    instance_kpts = results.keypoints.xy[idx].tolist()
                    
                    if len(instance_kpts) == 0 or instance_kpts[0] == [0.0, 0.0]:
                        continue
                        
                    if class_name == "Needle" and len(instance_kpts) >= 2:
                        needle_center = instance_kpts[0]
                        needle_tip = instance_kpts[1]
                    elif class_name == "Min":
                        min_marker = instance_kpts[0]
                    elif class_name == "Max":
                        max_marker = instance_kpts[0]
                    elif class_name == "Gauge":
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)

            # Compute geometry if all key elements are detected
            if needle_center and needle_tip and min_marker and max_marker:
                real_time_value, relative_angle_deg = self._process_geometry(
                    needle_center, needle_tip, min_marker, max_marker
                )
                # Cache parameters for state fallback defense
                self.last_valid_value = real_time_value
                self.last_valid_angle = relative_angle_deg
                
                # Render tracking visual elements
                cx, cy = int(needle_center[0]), int(needle_center[1])
                tx, ty = int(needle_tip[0]), int(needle_tip[1])
                mx, my = int(min_marker[0]), int(min_marker[1])
                kx, ky = int(max_marker[0]), int(max_marker[1])
                
                cv2.line(annotated_frame, (cx, cy), (mx, my), (255, 0, 0), 2)     # Blue: Center -> Min
                cv2.line(annotated_frame, (cx, cy), (kx, ky), (0, 165, 255), 2)   # Orange: Center -> Max
                cv2.line(annotated_frame, (cx, cy), (tx, ty), (0, 0, 255), 3)     # Red: Needle Vector
                
                cv2.circle(annotated_frame, (cx, cy), 6, (255, 0, 255), -1)       # Purple Pin
                cv2.circle(annotated_frame, (tx, ty), 6, (0, 255, 255), -1)       # Yellow Tip
                cv2.circle(annotated_frame, (mx, my), 6, (0, 255, 0), -1)         # Green Min
                cv2.circle(annotated_frame, (kx, ky), 6, (0, 0, 255), -1)         # Red Max
            else:
                # Dropout stabilization fallback injection
                real_time_value = self.last_valid_value
                relative_angle_deg = self.last_valid_angle

            # Apply UI dashboard overlay data texts
            cv2.putText(annotated_frame, f"VALUE: {real_time_value:.2f}", (30, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"ANGLE: {relative_angle_deg:.1f} DEG", (30, 110), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
            
            out.write(annotated_frame)
            cv2.imshow("Analog Gauge Reader - Active Stream", annotated_frame)
            
            if frame_count % 30 == 0:
                print(f" ⏳ Status: Compiled Frame {frame_count}/{total_frames}...")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠ Stream execution interrupted manually by user shortcut command.")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return output_path


if __name__ == "__main__":
    # ==================================================
    # CONFIGURATION RUNTIME ENTRY POINT
    # ==================================================
    MODEL_WEIGHTS = "best.pt"
    INPUT_VIDEO   = "inference.mp4"
    
    # Update these values to match the calibration metric on your dial face bounds
    DIAL_MIN_VALUE = 0.0
    DIAL_MAX_VALUE = 10.0
    # ==================================================

    reader = AnalogGaugeReader(
        model_path=MODEL_WEIGHTS, 
        min_value=DIAL_MIN_VALUE, 
        max_value=DIAL_MAX_VALUE
    )
    
    try:
        final_output = reader.process_video(source_path=INPUT_VIDEO, output_dir="final")
        print(f"\n Pipeline processing completed successfully!\n Output saved: {final_output}\n")
    except Exception as error:
        print(f"\n Pipeline runtime exception encountered: {error}\n")
