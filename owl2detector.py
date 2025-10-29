import cv2
import torch
import argparse
from transformers import Owlv2ForObjectDetection, Owlv2Processor

class Owlv2Detector:
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", target_objects=None):
        """
        Initialize the ObjectDetector with a model and the objects to detect.

        Args:
            model_name (str): The name of the pretrained model to use.
            target_objects (list): List of object labels to detect.
        """
        self.model_name = model_name
        self.target_objects = target_objects if target_objects else []
        
        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        """
        Load the Owlv2 model and processor from the Hugging Face Hub.

        Returns:
            model (Owlv2ForObjectDetection): The loaded model.
            processor (Owlv2Processor): The processor for image preprocessing.
        """
        print(f"Loading model: {self.model_name}")
        model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
        model = model.to(self.device)
        processor = Owlv2Processor.from_pretrained(self.model_name)
        return model, processor

    def detect_objects(self, frame):
        """
        Detect objects in the provided frame.

        Args:
            frame (ndarray): The input frame from the webcam.

        Returns:
            detections (list of dict): Each dict contains 'label', 'box', 'score'.
        """
        texts = [self.target_objects]
        inputs = self.processor(images=frame, text=texts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )[0]

        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                "label": self.target_objects[label.item()],
                "box": [int(i) for i in box.tolist()],
                "score": score.item()
            })
        return detections


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Object detection with Owlv2.")
    parser.add_argument(
        "--target-objects", 
        nargs="+", 
        default=["a person", "human face", "bicycle"], 
        help="List of objects to detect (e.g., '\"a person\" \"human face\" bicycle')"
    )
    parser.add_argument(
        "--model-name", 
        default="google/owlv2-base-patch16-ensemble", 
        help="Pretrained model name (default: 'google/owlv2-base-patch16-ensemble')"
    )
    return parser.parse_args()


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


def main():
    args = parse_args()

    detector = Owlv2Detector(model_name=args.model_name, target_objects=args.target_objects)

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


if __name__ == "__main__":
    main()
