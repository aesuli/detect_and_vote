from abc import ABC, abstractmethod
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor,  OwlViTForObjectDetection, OwlViTProcessor


class OwlDetector(ABC):
    """Abstract base class for OWL-based object detectors."""
    
    DEFAULT_MODEL_NAME = None  # Must be defined by subclasses
    
    def __init__(self, model_name=None, objects=None, threshold=0.15):
        """
        Initialize the ObjectDetector with a model and the objects to detect.

        Args:
            model_name (str): The name of the pretrained model to use.
            objects (list): List of object labels to detect.
            threshold (float): Detection confidence threshold.
        """
        if model_name is None:
            self.model_name = self.DEFAULT_MODEL_NAME
        else:
            self.model_name = model_name

        self.objects = objects if objects else []

        self.threshold = threshold
        
        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model, self.processor = self.load_model_and_processor()

    @abstractmethod
    def load_model_and_processor(self):
        """
        Load the model and processor from the Hugging Face Hub.
        Must be implemented by subclasses.

        Returns:
            tuple: (model, processor)
        """
        pass

    def set_objects(self, objects):
        """
        Update the list of objects to detect without reloading the model.

        Args:
            objects (list): New list of object labels to detect.
        """
        self.objects = objects if objects else []
        print(f"Updated detection objects: {self.objects}")

    def set_threshold(self, threshold):
        """
        Update the detection confidence threshold without reloading the model.

        Args:
            threshold (float): New detection confidence threshold.
        """
        self.threshold = float(threshold)
        print(f"Updated detection threshold: {self.threshold}")

    def detect_objects(self, frame):
        """
        Detect objects in the provided frame.

        Args:
            frame (ndarray): The input frame from the webcam.

        Returns:
            detections (list of dict): Each dict contains 'label', 'box', 'score'.
        """
        texts = [self.objects]
        inputs = self.processor(images=frame, text=texts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([frame.shape[:2]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold
        )[0]

        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                "label": self.objects[label.item()],
                "box": [int(i) for i in box.tolist()],
                "score": score.item()
            })
        return detections


class OwlViTDetector(OwlDetector):
    """OWL-ViT object detector implementation."""
    
    DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"
    
    def load_model_and_processor(self):
        """
        Load the OWL-ViT model and processor from the Hugging Face Hub.

        Returns:
            tuple: (model, processor)
        """
        print(f"Loading model: {self.model_name}")
        model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        processor = OwlViTProcessor.from_pretrained(self.model_name)
        model = model.to(self.device)
        return model, processor


class Owlv2Detector(OwlDetector):
    """OWL-v2 object detector implementation."""
    
    DEFAULT_MODEL_NAME = "google/owlv2-base-patch16-ensemble"

    def load_model_and_processor(self):
        """
        Load the OWL-v2 model and processor from the Hugging Face Hub.

        Returns:
            tuple: (model, processor)
        """
        print(f"Loading model: {self.model_name}")
        model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
        model = model.to(self.device)
        processor = Owlv2Processor.from_pretrained(self.model_name, use_fast=True)
        return model, processor
