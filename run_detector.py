import cv2
import argparse
import os
from typing import Optional
from owldetector import Owlv2Detector, OwlViTDetector

def render_detections_on_frame(frame, detections):
    """
    Draw bounding boxes and labels on the frame for each detection.

    Args:
        frame (ndarray): The image frame.
        detections (list of dict): Each dict contains 'label', 'box', 'score'.

    Returns:
        frame (ndarray): The frame with rendered detections.
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label_text = det["label"]
        confidence = det["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_text}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def _pick_fourcc_for_extension(path: str) -> int:
    """Return a sensible FOURCC based on file extension (Windows-friendly)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext in (".avi",):
        return cv2.VideoWriter_fourcc(*"MJPG")
    if ext in (".mkv",):
        # Fall back to mp4v; MKV support depends on backend. User can change extension if needed.
        return cv2.VideoWriter_fourcc(*"mp4v")
    # Unknown extension: default to MJPG in AVI-style encoding
    return cv2.VideoWriter_fourcc(*"MJPG")


def main(detector_type, model_name, objects, threshold, frame_width, frame_height, video_device_id, output_file=None, fps: Optional[float] = None):
    if detector_type == 'owlv2':
        detector = Owlv2Detector(model_name=model_name, objects=objects, threshold=threshold)
    elif detector_type == 'owlvit':
        detector = OwlViTDetector(model_name=model_name, objects=objects, threshold=threshold)
    else:
        raise ValueError("Invalid detector type. Choose 'owlv2' or 'owlvit'.")

    cap = cv2.VideoCapture(video_device_id)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    cap.set(3, frame_width)
    cap.set(4, frame_height)

    frame_skip = 5
    frame_count = 0
    out = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_height, frame_width = frame.shape[:2]

        if output_file and not out:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            use_fps = fps if fps and fps > 0 else (cap_fps if cap_fps and cap_fps > 0 else 20.0)

            fourcc = _pick_fourcc_for_extension(output_file)
            out = cv2.VideoWriter(output_file, fourcc, use_fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Warning: Could not open VideoWriter for '{output_file}'. Disabling video saving.")
                output_file = None
                out = None
            else:
                print(f"Saving video to '{output_file}' at {use_fps:.2f} FPS, size {frame_width}x{frame_height}.")

        if frame_count % frame_skip == 0:
            detections = detector.detect_objects(frame)

        frame_with_detections = render_detections_on_frame(frame, detections)

        if out is not None:
            out.write(frame_with_detections)

        cv2.imshow("Object Detection", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select detector type.')
    parser.add_argument('-d', '--detector', type=str, 
                        choices=['owlv2', 'owlvit'], default='owlvit',
                        help='Type of detector to use: owl2 or owlvit (default: owlvit)')
    parser.add_argument('-m', "--model-name", 
        help=f"Pretrained model name (default for owlvit: {OwlViTDetector.DEFAULT_MODEL_NAME}, for owlv2: {Owlv2Detector.DEFAULT_MODEL_NAME})"
    )
    parser.add_argument('-o',
        "--objects", 
        nargs="+", 
        default=["a person", "human face", "a hand"], 
        help="List of objects to detect (e.g., '\"a person\" \"human face\" \"a hand\"')"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15)"
    )
    parser.add_argument('-fw',
        "--frame-width",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument('-fh',
        "--frame-height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument('-vd',
        "--video-device-id",
        type=int,
        default=0,
        help="Video capture device index (default: 0)"
    )
    parser.add_argument('-lvd', '--list-video-devices',
        action='store_true',
        help="List available video capture devices and exit"
    )
    parser.add_argument('-of',
                        '--output-file', 
                        help="Save the video of the detection to a file. Use .mp4 (mp4v) or .avi (MJPG), e.g. output.mp4")
    parser.add_argument('--fps', type=float, default=None,
                        help="Frames per second for output file. If omitted, use camera FPS or 20.0 fallback.")
    args = parser.parse_args()

    if args.list_video_devices:
        print("Available video devices:")
        i = 0
        while True:
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"  Device id {i}: Available")
                    cap.release()
                else:
                    break
                i+=1
            except:
                break
        exit(0)

    main(args.detector, args.model_name, args.objects, args.threshold, args.frame_width, args.frame_height, args.video_device_id, args.output_file, args.fps)