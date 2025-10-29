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

def main(detector_type, model_name, target_objects, threshold):
    if detector_type == 'owlv2':
        detector = Owlv2Detector(model_name=model_name, target_objects=target_objects, threshold=threshold)
    elif detector_type == 'owlvit':
        detector = OwlViTDetector(model_name=model_name,target_objects=target_objects, threshold=threshold)
    else:
        raise ValueError("Invalid detector type. Choose 'owlv2' or 'owlvit'.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

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
    parser.add_argument(
        "--target-objects", 
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
    args = parser.parse_args()

    main(args.detector, args.model_name, args.target_objects, args.threshold)