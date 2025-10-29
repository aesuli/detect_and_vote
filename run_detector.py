import cv2
import argparse
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

def main(detector_type, model_name, objects, threshold, frame_width, frame_height, video_device_id):
    if detector_type == 'owlv2':
        detector = Owlv2Detector(model_name=model_name, objects=objects, threshold=threshold)
    elif detector_type == 'owlvit':
        detector = OwlViTDetector(model_name=model_name,objects=objects, threshold=threshold)
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
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        if frame_count%frame_skip==0:
            detections = detector.detect_objects(frame)

        frame_with_detections = render_detections_on_frame(frame, detections)

        cv2.imshow("Object Detection", frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
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
        default=["a person", "human face", "bicycle"], 
        help="List of objects to detect (e.g., '\"a person\" \"human face\" bicycle')"
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

    main(args.detector, args.model_name, args.objects, args.threshold, args.frame_width, args.frame_height, args.video_device_id)