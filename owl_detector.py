from abc import ABC, abstractmethod
import os

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

    @staticmethod
    def _from_pretrained_cached_first(loader, model_name, **kwargs):
        """Use an existing Hugging Face cache without making a metadata request.

        If the cache is incomplete and offline mode was not explicitly requested,
        fall back to the normal Hub-enabled loader so first-time setup still works.
        """
        try:
            return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
        except OSError as cache_error:
            offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
            offline = offline or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}
            if offline:
                raise RuntimeError(
                    f"Model '{model_name}' is not completely cached and offline mode is enabled. "
                    "Connect once to download the missing model/processor files."
                ) from cache_error
            print(f"Local cache for {model_name} is incomplete; downloading missing files from Hugging Face.")
            return loader.from_pretrained(model_name, **kwargs)

    def detect_objects(self, frame):
        """
        Detect objects in the provided frame.

        Args:
            frame (ndarray): The input frame from the webcam.

        Returns:
            detections (list of dict): Each dict contains 'label', 'box', 'score'.
        """
        # Snapshot mutable settings once so concurrent UI updates cannot desync
        # text prompts from label mapping in the same inference pass.
        objects = list(self.objects) if self.objects else []
        if not objects:
            return []
        threshold = float(self.threshold)

        texts = [objects]
        inputs = self.processor(images=frame, text=texts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([frame.shape[:2]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            label_idx = int(label.item())
            if label_idx < 0 or label_idx >= len(objects):
                # Defensive guard against transient or malformed label indices.
                continue
            detections.append({
                "label": objects[label_idx],
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
        model = self._from_pretrained_cached_first(OwlViTForObjectDetection, self.model_name)
        processor = self._from_pretrained_cached_first(OwlViTProcessor, self.model_name)
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
        model = self._from_pretrained_cached_first(Owlv2ForObjectDetection, self.model_name)
        model = model.to(self.device)
        processor = self._from_pretrained_cached_first(Owlv2Processor, self.model_name, use_fast=True)
        return model, processor
