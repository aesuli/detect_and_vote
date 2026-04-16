import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import json
import math
import random
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cherrypy
import cv2
import numpy as np
from cherrypy.process.plugins import SignalHandler

from owl_detector import Owlv2Detector, OwlViTDetector


class ObjectDetectionApp:
    """CherryPy app for detection, region counting, and quiz voting."""

    REGION_SLOT_VOTING_MODE = "region_slots"
    OBJECT_LIST_VOTING_MODE = "object_lists"
    DEFAULT_VOTE_DURATION_SEC = 30
    DEFAULT_PAUSE_DURATION_SEC = 10
    DEFAULT_PRE_QUESTION_COUNTDOWN_SEC = 3
    DEFAULT_VOTE_WINDOW_SIZE = 50

    def __init__(self, questions_path: str = os.path.join("data", "test.jsonl")):
        self.detector_type = "owlvit"
        self.model_name = None
        self.voting_mode = self.REGION_SLOT_VOTING_MODE
        self.region_vote_objects = ["a person"]
        self.answer_objects: Dict[int, List[str]] = {1: ["the palm of an open hand"], 2: ["a hand closed in a fist"]}
        self.objects = list(self.region_vote_objects)
        self.threshold = 0.15
        self.frame_width = 960
        self.frame_height = 540
        self.video_device_id = 0
        self.frame_skip = 4

        self.detector = None
        self.cap = None
        self.running = False
        self.lock = threading.Lock()

        self.current_frame = None
        self.current_frame_with_detections = None
        self.last_detections: List[Dict] = []
        self.last_assignments: List[Dict] = []

        self.regions: List[Dict] = self._build_default_regions()
        self.next_region_id = len(self.regions) + 1
        self.region_masks: Dict[int, np.ndarray] = {}
        self._regions_customized: bool = False
        self._rebuild_region_masks_locked()

        self.latest_region_counts: Dict[int, int] = {}
        self.latest_slot_counts: Dict[int, int] = {1: 0, 2: 0}
        self.slot_colors: Dict[int, str] = {1: "#0066cc", 2: "#ffcc00"}
        self.count_history = deque(maxlen=600)

        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.app_root, "data")
        self.questions_path = self._resolve_data_path(questions_path)
        self.questions = self._load_questions(self.questions_path)
        self.question_order: List[int] = []
        self.question_cursor = 0

        self.vote_duration_sec = self.DEFAULT_VOTE_DURATION_SEC
        self.pause_duration_sec = self.DEFAULT_PAUSE_DURATION_SEC
        self.pre_question_countdown_sec = self.DEFAULT_PRE_QUESTION_COUNTDOWN_SEC
        self.vote_window_size = self.DEFAULT_VOTE_WINDOW_SIZE
        self.shuffle_answers = True
        self.vote_results = deque(maxlen=self.vote_window_size)

        self.voting_active = False
        self.voting_phase = "idle"  # idle | countdown | question | pause
        self.phase_end_ts = 0.0
        self.current_question: Optional[Dict] = None
        self.last_vote_result: Optional[Dict] = None

        self._capture_thread = None
        self._initialize_detector()

    def _build_default_regions(self) -> List[Dict]:
        sixth_w = max(1, int(self.frame_width) // 6)
        third_h = max(1, self.frame_height//3)


        return [
            {
                "id": 1,
                "answer_slot": 1,
                "criterion": "overlap",
                "points": [
                    [sixth_w*4, third_h], 
                    [sixth_w*5, third_h], 
                    [sixth_w*5, third_h*2], 
                    [sixth_w*4, third_h*2]
                    ],
            },
            {
                "id": 2,
                "answer_slot": 2,
                "criterion": "overlap",
                "points": [
                    [sixth_w, third_h], 
                    [sixth_w*2, third_h], 
                    [sixth_w*2, third_h*2], 
                    [sixth_w, third_h*2]
                    ],
            },
        ]

    def _initialize_detector(self):
        try:
            if self.detector_type == "owlv2":
                self.detector = Owlv2Detector(
                    model_name=self.model_name,
                    objects=self.objects,
                    threshold=self.threshold,
                )
            elif self.detector_type == "owlvit":
                self.detector = OwlViTDetector(
                    model_name=self.model_name,
                    objects=self.objects,
                    threshold=self.threshold,
                )
            else:
                raise ValueError("Invalid detector type")
        except Exception as exc:
            print(f"Error initializing detector: {exc}")
            self.detector = None

    def _initialize_video_capture(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video device {self.video_device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def _load_questions(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            print(f"Questions file not found at {path}. Voting will start with an empty pool.")
            return []

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except Exception as exc:
            print(f"Failed to load questions from {path}: {exc}")
            return []

        try:
            payload = self._read_question_payload_from_text(text, source_name=os.path.basename(path))
        except Exception as exc:
            print(f"Failed to parse questions from {path}: {exc}")
            return []

        return self._normalize_questions(payload)

    def _resolve_data_path(self, source_path: str) -> str:
        if os.path.isabs(source_path):
            return source_path
        return os.path.join(self.app_root, source_path)

    def _supported_question_extension(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in {".json", ".jsonl"}

    def _read_question_payload_from_text(self, text: str, source_name: str = "") -> Any:
        source_name = str(source_name or "").strip().lower()

        if source_name.endswith(".jsonl"):
            return self._parse_jsonl(text)

        try:
            return json.loads(text)
        except Exception:
            return self._parse_jsonl(text)

    def _parse_jsonl(self, text: str) -> List[Dict]:
        parsed = []
        for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            parsed.append(item)
        return parsed

    def _normalize_single_question(self, item: Dict, index: int) -> Optional[Dict]:
        if not isinstance(item, dict):
            return None

        question = str(item.get("question", "")).strip()
        answers = item.get("answers", [])
        correct_answer = str(item["correct_answer"]).strip() if item.get("correct_answer") else None

        if not isinstance(answers, list) or len(answers) != 2:
            return None

        a0 = str(answers[0]).strip()
        a1 = str(answers[1]).strip()
        if not question or not a0 or not a1:
            return None

        try:
            extra_time = int(item.get("extra_time", 0))
        except (TypeError, ValueError):
            extra_time = 0

        more_info = str(item["more_info"]).strip() if item.get("more_info") else None

        return {
            "id": int(item.get("id", index + 1)),
            "question": question,
            "answers": [a0, a1],
            "extra_time": max(0, extra_time),
            "correct_answer": correct_answer,
            "more_info": more_info,
        }

    def _normalize_questions(self, payload: Any) -> List[Dict]:
        if isinstance(payload, dict):
            items = payload.get("questions", [])
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        normalized = []
        for i, item in enumerate(items):
            normalized_item = self._normalize_single_question(item, i)
            if normalized_item is None:
                continue
            normalized.append(normalized_item)

        return normalized

    def _list_question_sources(self) -> List[Dict[str, Any]]:
        if not os.path.isdir(self.data_dir):
            return []

        files = []
        selected_default = os.path.basename(self.questions_path)
        for name in sorted(os.listdir(self.data_dir), key=lambda item: (0 if item.lower().endswith(".jsonl") else 1, item.lower())):
            abs_path = os.path.join(self.data_dir, name)
            if not os.path.isfile(abs_path) or not self._supported_question_extension(abs_path):
                continue
            files.append(
                {
                    "name": name,
                    "selected": name == selected_default,
                }
            )
        return files

    def _normalize_selected_source_names(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            raw_value = str(value)
            if "," in raw_value or "\n" in raw_value:
                candidates = [part.strip() for part in re.split(r"[,\n]", raw_value) if part.strip()]
            else:
                candidates = [value]

        selected = []
        seen = set()
        for candidate in candidates:
            name = os.path.basename(str(candidate or "").strip())
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            abs_path = os.path.join(self.data_dir, name)
            if os.path.isfile(abs_path) and self._supported_question_extension(abs_path):
                selected.append(name)
        return selected

    def _normalize_voting_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == self.OBJECT_LIST_VOTING_MODE:
            return self.OBJECT_LIST_VOTING_MODE
        return self.REGION_SLOT_VOTING_MODE

    def _normalize_object_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            items = value.splitlines()
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = []

        cleaned = []
        seen = set()
        for item in items:
            label = str(item or "").strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(label)
        return cleaned

    def _normalize_answer_object_lists(self, value: Any) -> Dict[int, List[str]]:
        if isinstance(value, dict):
            raw_slot_1 = value.get("1", value.get(1, []))
            raw_slot_2 = value.get("2", value.get(2, []))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            raw_slot_1 = value[0]
            raw_slot_2 = value[1]
        else:
            raw_slot_1 = []
            raw_slot_2 = []

        slot_1 = self._normalize_object_list(raw_slot_1)
        slot_2 = self._normalize_object_list(raw_slot_2)

        overlap = {label.casefold() for label in slot_1}.intersection(label.casefold() for label in slot_2)
        if overlap:
            raise ValueError("Answer 1 and Answer 2 object lists must not contain the same object.")

        return {1: slot_1, 2: slot_2}

    def _active_detector_objects(self, voting_mode: str, region_vote_objects: List[str], answer_objects: Dict[int, List[str]]) -> List[str]:
        if voting_mode == self.OBJECT_LIST_VOTING_MODE:
            merged = []
            seen = set()
            for slot in (1, 2):
                for label in answer_objects.get(slot, []):
                    key = label.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(label)
            return merged
        return list(region_vote_objects)

    def _detected_label_vote_slot_locked(self, label: Any) -> Optional[int]:
        normalized = str(label or "").strip().casefold()
        if not normalized:
            return None
        for slot in (1, 2):
            if normalized in {item.casefold() for item in self.answer_objects.get(slot, [])}:
                return slot
        return None

    def _ensure_question_order(self):
        if not self.questions:
            self.question_order = []
            self.question_cursor = 0
            return
        if self.question_cursor >= len(self.question_order):
            self.question_order = list(range(len(self.questions)))
            random.shuffle(self.question_order)
            self.question_cursor = 0

    def _next_question(self) -> Optional[Dict]:
        self._ensure_question_order()
        if not self.question_order:
            return None
        idx = self.question_order[self.question_cursor]
        self.question_cursor += 1
        return self.questions[idx]

    def _build_active_question(self, question: Dict) -> Dict:
        active_question = dict(question)
        answers = list(question.get("answers", []))
        if self.shuffle_answers and len(answers) == 2:
            random.shuffle(answers)
        active_question["answers"] = answers
        return active_question

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(value)

    def _read_json_body(self) -> Dict:
        raw = cherrypy.request.body.read()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _rebuild_region_masks_locked(self):
        masks = {}
        if self.frame_width <= 0 or self.frame_height <= 0:
            self.region_masks = masks
            return

        for region in self.regions:
            pts = region.get("points", [])
            if len(pts) < 3:
                continue
            poly = np.array(pts, dtype=np.int32)
            mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            masks[region["id"]] = mask

        self.region_masks = masks

    def _region_answer_slot(self, region: Dict) -> int:
        raw_slot = region.get("answer_slot", region.get("label", 1))
        try:
            if isinstance(raw_slot, str):
                raw_slot = raw_slot.strip().lower().replace("answer", "").strip()
            slot = int(raw_slot)
        except (TypeError, ValueError):
            slot = 1
        return slot if slot in (1, 2) else 1

    def _answer_slot_name_locked(self, slot: int) -> str:
        if self.current_question:
            answers = self.current_question.get("answers", [])
            if len(answers) >= slot:
                return str(answers[slot - 1])
        return f"Answer {slot}"

    def _display_slot_counts_locked(self) -> Dict[str, int]:
        return {
            self._answer_slot_name_locked(slot): int(self.latest_slot_counts.get(slot, 0))
            for slot in (1, 2)
        }

    def _normalize_color(self, color: Any, fallback: str) -> str:
        value = str(color or "").strip()
        if len(value) == 7 and value.startswith("#"):
            hex_part = value[1:]
            if all(ch in "0123456789abcdefABCDEF" for ch in hex_part):
                return value.lower()
        return fallback

    def _slot_color_hex_locked(self, slot: int) -> str:
        default = "#0066cc" if slot == 1 else "#ffcc00"
        return self._normalize_color(self.slot_colors.get(slot, default), default)

    def _slot_color_bgr_locked(self, slot: int) -> Tuple[int, int, int]:
        color = self._slot_color_hex_locked(slot)
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        return (blue, green, red)

    def _display_slot_colors_locked(self) -> Dict[str, str]:
        return {str(slot): self._slot_color_hex_locked(slot) for slot in (1, 2)}

    def _box_inside_polygon(self, box: List[int], points: List[List[int]]) -> bool:
        x1, y1, x2, y2 = box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        poly = np.array(points, dtype=np.int32)
        for corner in corners:
            if cv2.pointPolygonTest(poly, corner, False) < 0:
                return False
        return True

    def _overlap_ratio(self, box: List[int], region_mask: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        x1 = max(0, min(self.frame_width - 1, x1))
        x2 = max(0, min(self.frame_width - 1, x2))
        y1 = max(0, min(self.frame_height - 1, y1))
        y2 = max(0, min(self.frame_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = region_mask[y1:y2, x1:x2]
        box_area = float((x2 - x1) * (y2 - y1))
        if box_area <= 0:
            return 0.0
        intersection = float(np.sum(roi))
        return intersection / box_area

    def _assign_detections_locked(self, detections: List[Dict]) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
        assignments = []
        region_counts: Dict[int, int] = {}
        slot_counts: Dict[int, int] = {1: 0, 2: 0}

        for region in self.regions:
            region_counts[region["id"]] = 0

        for det in detections:
            best_region = None
            best_score = 0.0

            for region in self.regions:
                points = region.get("points", [])
                if len(points) < 3:
                    continue

                criterion = region.get("criterion", "overlap")
                score = 0.0
                if criterion == "inside":
                    if self._box_inside_polygon(det["box"], points):
                        score = 1.0
                else:
                    mask = self.region_masks.get(region["id"])
                    if mask is None:
                        continue
                    score = self._overlap_ratio(det["box"], mask)

                if score > best_score:
                    best_score = score
                    best_region = region

            assigned = {
                "label": det["label"],
                "box": det["box"],
                "score": det["score"],
                "region_id": None,
                "vote_slot": None,
                "vote_display_name": None,
                "counted": False,
                "assignment_reason": "outside_vote_area",
                "region_answer_slot": None,
                "region_display_name": None,
                "region_score": 0.0,
            }
            if best_region is not None and best_score > 0:
                if self.voting_mode == self.OBJECT_LIST_VOTING_MODE:
                    answer_slot = self._detected_label_vote_slot_locked(det["label"])
                    assigned["assignment_reason"] = "label_not_mapped" if answer_slot is None else "counted"
                else:
                    answer_slot = self._region_answer_slot(best_region)
                    assigned["assignment_reason"] = "counted"
                assigned["region_id"] = best_region["id"]
                assigned["region_score"] = round(best_score, 4)
                if answer_slot in (1, 2):
                    assigned["vote_slot"] = answer_slot
                    assigned["vote_display_name"] = self._answer_slot_name_locked(answer_slot)
                    assigned["counted"] = True
                    assigned["region_answer_slot"] = answer_slot
                    assigned["region_display_name"] = assigned["vote_display_name"]
                    region_counts[best_region["id"]] += 1
                    slot_counts[answer_slot] = slot_counts.get(answer_slot, 0) + 1

            assignments.append(assigned)

        return assignments, region_counts, slot_counts

    def _render_detections(self, frame: np.ndarray, assignments: List[Dict]) -> np.ndarray:
        for det in assignments:
            x1, y1, x2, y2 = det["box"]
            assigned_text = det.get("vote_display_name") or det.get("region_display_name") or det.get("assignment_reason") or "unassigned"
            vote_slot = det.get("vote_slot", det.get("region_answer_slot"))
            color = self._slot_color_bgr_locked(vote_slot) if vote_slot else (120, 120, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{det['label']} {det['score']:.2f} -> {assigned_text}"
            cv2.putText(
                frame,
                caption,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return frame

    def _update_history_locked(self):
        now = time.time()
        snapshot = {
            "ts": now,
            "counts": {1: int(self.latest_slot_counts.get(1, 0)), 2: int(self.latest_slot_counts.get(2, 0))},
        }
        self.count_history.append(snapshot)

    def _finalize_vote_locked(self):
        if not self.current_question:
            return

        answers = self.current_question.get("answers", [])
        if len(answers) != 2:
            return

        a0, a1 = answers[0], answers[1]
        c0 = int(self.latest_slot_counts.get(1, 0))
        c1 = int(self.latest_slot_counts.get(2, 0))

        if c0 == c1:
            voted = None
        elif c0 > c1:
            voted = a0
        else:
            voted = a1

        correct_answer = self.current_question.get("correct_answer")
        is_correct = None
        if correct_answer:
            is_correct = voted == correct_answer
            self.vote_results.append(1 if is_correct else 0)

        self.last_vote_result = {
            "question": self.current_question,
            "counts": {a0: c0, a1: c1},
            "voted_answer": voted,
            "is_correct": is_correct,
        }

    def _question_extra_time_sec(self, question: Optional[Dict]) -> int:
        if not question:
            return 0
        try:
            return max(0, int(question.get("extra_time", 0)))
        except (TypeError, ValueError):
            return 0

    def _start_question_phase_locked(self):
        if not self.current_question:
            self._stop_voting_locked(clear_last_result=False)
            return
        total_duration = max(3, self.vote_duration_sec + self._question_extra_time_sec(self.current_question))
        self.voting_phase = "question"
        self.phase_end_ts = time.time() + total_duration

    def _start_next_question_locked(self):
        q = self._next_question()
        if not q:
            self._stop_voting_locked(clear_last_result=False)
            return

        self.current_question = self._build_active_question(q)
        if self.pre_question_countdown_sec > 0:
            self.voting_phase = "countdown"
            self.phase_end_ts = time.time() + self.pre_question_countdown_sec
            return

        self._start_question_phase_locked()

    def _tick_voting_locked(self):
        if not self.voting_active:
            return

        now = time.time()
        if now < self.phase_end_ts:
            return

        if self.voting_phase == "question":
            self._finalize_vote_locked()
            self.voting_phase = "pause"
            self.phase_end_ts = now + self.pause_duration_sec
            return

        if self.voting_phase == "countdown":
            self._start_question_phase_locked()
            return

        if self.voting_phase == "pause":
            self._start_next_question_locked()

    def _set_voting_config_locked(self, vote_duration: int, pause_duration: int, pre_question_countdown: int, window_size: int, shuffle_answers: bool):
        self.vote_duration_sec = vote_duration
        self.pause_duration_sec = pause_duration
        self.pre_question_countdown_sec = pre_question_countdown
        self.shuffle_answers = bool(shuffle_answers)
        if window_size != self.vote_window_size:
            self.vote_window_size = window_size
            self.vote_results = deque(list(self.vote_results), maxlen=self.vote_window_size)

    def _stop_voting_locked(self, clear_last_result: bool = False):
        self.voting_active = False
        self.voting_phase = "idle"
        self.phase_end_ts = 0.0
        self.current_question = None
        if clear_last_result:
            self.last_vote_result = None

    def _start_voting_locked(self):
        self.voting_active = True
        self.last_vote_result = None
        self._start_next_question_locked()

    def _capture_loop(self):
        try:
            self._initialize_video_capture()
        except RuntimeError as exc:
            print(f"Video initialization failed: {exc}")
            self.running = False
            return

        frame_count = 0
        detections: List[Dict] = []

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_skip = max(1, int(self.frame_skip))
            if frame_count % frame_skip == 0 and self.detector is not None:
                try:
                    detections = self.detector.detect_objects(frame)
                except Exception as exc:
                    print(f"Detector inference error: {exc}")
                    detections = []

            with self.lock:
                h, w = frame.shape[:2]
                if (w, h) != (self.frame_width, self.frame_height):
                    self.frame_width = w
                    self.frame_height = h
                    if not self._regions_customized:
                        self.regions = self._build_default_regions()
                        self.next_region_id = len(self.regions) + 1
                    self._rebuild_region_masks_locked()

                assignments, region_counts, slot_counts = self._assign_detections_locked(detections)
                self.last_detections = detections
                self.last_assignments = assignments
                self.latest_region_counts = region_counts
                self.latest_slot_counts = slot_counts
                self._update_history_locked()
                self._tick_voting_locked()

                self.current_frame = frame.copy()
                self.current_frame_with_detections = None

            frame_count += 1

        if self.cap:
            self.cap.release()

    def start_capture(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop_capture(self):
        if not self.running:
            return
        self.running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3)

    def _frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        if frame is None:
            return None
        ok, data = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return data.tobytes()

    def _voting_state_locked(self) -> Dict:
        now = time.time()
        time_left = 0
        if self.voting_active and self.phase_end_ts > now:
            time_left = max(0, int(math.ceil(self.phase_end_ts - now)))

        accuracy = None
        if len(self.vote_results) > 0:
            accuracy = round(100.0 * sum(self.vote_results) / len(self.vote_results), 2)

        return {
            "active": self.voting_active,
            "phase": self.voting_phase,
            "time_left_sec": time_left,
            "vote_duration_sec": self.vote_duration_sec,
            "pause_duration_sec": self.pause_duration_sec,
            "pre_question_countdown_sec": self.pre_question_countdown_sec,
            "window_size": self.vote_window_size,
            "shuffle_answers": self.shuffle_answers,
            "recent_scored_votes": len(self.vote_results),
            "accuracy_percent": accuracy,
            "current_question": self.current_question,
            "last_vote_result": self.last_vote_result,
            "question_pool_size": len(self.questions),
        }

    def _to_json_safe(self, value: Any):
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _build_state_payload_locked(self) -> Dict:
        payload = {
            "status": "success",
            "frame": {
                "width": self.frame_width,
                "height": self.frame_height,
            },
            "detector": {
                "type": self.detector_type,
                "model_name": self.model_name,
                "threshold": self.threshold,
                "objects": self.objects,
                "region_objects": self.region_vote_objects,
                "answer_objects": {
                    "1": list(self.answer_objects.get(1, [])),
                    "2": list(self.answer_objects.get(2, [])),
                },
                "voting_mode": self.voting_mode,
                "frame_skip": self.frame_skip,
            },
            "regions": self.regions,
            "assignments": self.last_assignments,
            "region_counts": self.latest_region_counts,
            "slot_counts": self.latest_slot_counts,
            "slot_colors": self._display_slot_colors_locked(),
            "display_counts": self._display_slot_counts_locked(),
            "history": list(self.count_history),
            "voting": self._voting_state_locked(),
        }
        return self._to_json_safe(payload)

    @cherrypy.expose
    def index(self):
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "templates",
            "web_count_and_vote.html",
        )
        try:
            with open(template_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            raise cherrypy.HTTPError(500, f"Failed to load index template: {exc}")

    @cherrypy.expose
    def video_feed(self):
        cherrypy.response.headers["Content-Type"] = "multipart/x-mixed-replace; boundary=frame"

        def generate():
            while True:
                with self.lock:
                    frame = self.current_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                jpeg = self._frame_to_jpeg(frame)
                if jpeg is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )

        return generate()

    video_feed._cp_config = {"response.stream": True}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def state_json(self):
        try:
            with self.lock:
                return self._build_state_payload_locked()
        except Exception as exc:
            return {"status": "error", "message": f"Failed to build state payload: {exc}"}

    @cherrypy.expose("state.json")
    @cherrypy.tools.json_out()
    def state_json_alias(self):
        try:
            with self.lock:
                return self._build_state_payload_locked()
        except Exception as exc:
            return {"status": "error", "message": f"Failed to build state payload: {exc}"}

    @cherrypy.expose("settings/apply")
    @cherrypy.tools.json_out()
    def apply_settings(self):
        payload = self._read_json_body()
        try:
            detector = payload.get("detector", self.detector_type)
            model_name = payload.get("model_name", self.model_name)
            voting_mode = self._normalize_voting_mode(payload.get("voting_mode", self.voting_mode))
            region_vote_objects = self._normalize_object_list(payload.get("region_objects", payload.get("objects", self.region_vote_objects)))
            answer_objects = self._normalize_answer_object_lists(payload.get("answer_objects", self.answer_objects))
            threshold = float(payload.get("threshold", self.threshold))
            frame_skip = int(payload.get("frame_skip", self.frame_skip))

            if frame_skip < 1:
                raise ValueError("'frame_skip' must be >= 1")

            if voting_mode == self.OBJECT_LIST_VOTING_MODE:
                if not answer_objects.get(1) or not answer_objects.get(2):
                    raise ValueError("Provide at least one object for both Answer 1 and Answer 2 in object-list mode.")
            elif not region_vote_objects:
                raise ValueError("Provide at least one detector object in region-based mode.")

            active_objects = self._active_detector_objects(voting_mode, region_vote_objects, answer_objects)
            if not active_objects:
                raise ValueError("No valid detection objects provided.")

            with self.lock:
                model_reload = detector != self.detector_type or model_name != self.model_name
                self.detector_type = detector
                self.model_name = model_name
                self.voting_mode = voting_mode
                self.region_vote_objects = region_vote_objects
                self.answer_objects = {
                    1: list(answer_objects.get(1, [])),
                    2: list(answer_objects.get(2, [])),
                }
                self.objects = active_objects
                self.threshold = threshold
                self.frame_skip = frame_skip

                if model_reload:
                    self._initialize_detector()
                elif self.detector is not None:
                    self.detector.set_objects(active_objects)
                    self.detector.set_threshold(threshold)
                else:
                    self._initialize_detector()

            return {"status": "success", "message": "Detector settings applied."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @cherrypy.expose("regions/set")
    @cherrypy.tools.json_out()
    def set_regions(self):
        payload = self._read_json_body()
        incoming = payload.get("regions", [])
        if not isinstance(incoming, list):
            return {"status": "error", "message": "regions must be a list"}

        regions = []
        for item in incoming:
            if not isinstance(item, dict):
                continue
            points = item.get("points", [])
            if not isinstance(points, list) or len(points) < 3:
                continue

            clean_points = []
            for p in points:
                if not isinstance(p, list) or len(p) != 2:
                    continue
                x = int(p[0])
                y = int(p[1])
                clean_points.append([x, y])

            if len(clean_points) < 3:
                continue

            answer_slot = self._region_answer_slot(item)
            if answer_slot not in (1, 2):
                continue

            criterion = str(item.get("criterion", "overlap")).strip().lower()
            if criterion not in ("inside", "overlap"):
                criterion = "overlap"

            incoming_id = item.get("id")
            if incoming_id is None:
                rid = self.next_region_id
                self.next_region_id += 1
            else:
                rid = int(incoming_id)
                self.next_region_id = max(self.next_region_id, rid + 1)

            regions.append(
                {
                    "id": rid,
                    "answer_slot": answer_slot,
                    "criterion": criterion,
                    "points": clean_points,
                }
            )

        with self.lock:
            self.regions = regions
            self._regions_customized = True
            self._rebuild_region_masks_locked()

        return {
            "status": "success",
            "message": f"Saved {len(regions)} regions.",
            "regions": regions,
        }

    @cherrypy.expose("slot-colors/set")
    @cherrypy.tools.json_out()
    def set_slot_colors(self):
        payload = self._read_json_body()
        incoming = payload.get("slot_colors", {})
        if not isinstance(incoming, dict):
            return {"status": "error", "message": "slot_colors must be an object"}

        with self.lock:
            self.slot_colors[1] = self._normalize_color(incoming.get("1", incoming.get(1)), self._slot_color_hex_locked(1))
            self.slot_colors[2] = self._normalize_color(incoming.get("2", incoming.get(2)), self._slot_color_hex_locked(2))
            slot_colors = self._display_slot_colors_locked()

        return {
            "status": "success",
            "message": "Answer colors updated.",
            "slot_colors": slot_colors,
        }

    @cherrypy.expose("voting/config")
    @cherrypy.tools.json_out()
    def voting_config(self):
        payload = self._read_json_body()
        try:
            vote_duration = int(payload.get("vote_duration_sec", self.vote_duration_sec))
            pause_duration = int(payload.get("pause_duration_sec", self.pause_duration_sec))
            pre_question_countdown = int(payload.get("pre_question_countdown_sec", self.pre_question_countdown_sec))
            window_size = int(payload.get("window_size", self.vote_window_size))
            shuffle_answers = self._coerce_bool(payload.get("shuffle_answers", self.shuffle_answers))

            if vote_duration <= 5:
                raise ValueError("vote_duration_sec must be > 5")
            if pause_duration < 0:
                raise ValueError("pause_duration_sec must be >= 0")
            if pre_question_countdown < 0:
                raise ValueError("pre_question_countdown_sec must be >= 0")
            if window_size < 1:
                raise ValueError("window_size must be >= 1")

            with self.lock:
                self._set_voting_config_locked(vote_duration, pause_duration, pre_question_countdown, window_size, shuffle_answers)

            return {"status": "success", "message": "Voting configuration updated."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @cherrypy.expose("voting/questions")
    @cherrypy.expose("voting_questions")
    @cherrypy.tools.json_out()
    def voting_questions(self, **_params):
        content_type = str(cherrypy.request.headers.get("Content-Type", "")).lower()

        if content_type.startswith("application/json"):
            payload = self._read_json_body()
            if not isinstance(payload, (dict, list)):
                return {"status": "error", "message": "Provide a JSON array or object with 'questions'."}
            normalized = self._normalize_questions(payload)
        else:
            selected_raw = cherrypy.request.params.get("selected_files")
            getall = getattr(cherrypy.request.params, "getall", None)
            if callable(getall):
                try:
                    all_selected = getall("selected_files")
                    if all_selected:
                        selected_raw = all_selected
                except Exception:
                    pass

            selected_names = self._normalize_selected_source_names(selected_raw)
            if not selected_names:
                return {"status": "error", "message": "Select at least one question file from the checklist."}

            normalized = []
            for name in selected_names:
                source_path = os.path.join(self.data_dir, name)
                normalized.extend(self._load_questions(source_path))

            uploaded = cherrypy.request.params.get("questions_file")
            upload_name = str(getattr(uploaded, "filename", "") or "").strip()
            upload_stream = getattr(uploaded, "file", None)
            if upload_name and upload_stream is not None:
                raw_bytes = upload_stream.read()
                if raw_bytes:
                    decoded = raw_bytes.decode("utf-8-sig", errors="replace")
                    payload = self._read_question_payload_from_text(decoded, source_name=upload_name)
                    normalized.extend(self._normalize_questions(payload))

        # Keep IDs sequential after merging multiple sources.
        for idx, question in enumerate(normalized, start=1):
            question["id"] = idx

        with self.lock:
            self.questions = normalized
            self.question_order = []
            self.question_cursor = 0

        return {"status": "success", "message": f"Loaded {len(normalized)} questions."}

    @cherrypy.expose("voting/question_sources")
    @cherrypy.expose("voting_question_sources")
    @cherrypy.tools.json_out()
    def voting_question_sources(self):
        return {
            "status": "success",
            "sources": self._list_question_sources(),
        }

    @cherrypy.expose("voting/start")
    @cherrypy.tools.json_out()
    def voting_start(self):
        with self.lock:
            if not self.questions:
                return {"status": "error", "message": "Question pool is empty."}
            self._start_voting_locked()
        return {"status": "success", "message": "Voting started."}

    @cherrypy.expose("voting/stop")
    @cherrypy.tools.json_out()
    def voting_stop(self):
        with self.lock:
            self._stop_voting_locked(clear_last_result=False)
        return {"status": "success", "message": "Voting stopped."}


def app_exit(app: ObjectDetectionApp):
    print("\nShutting down...")
    app.stop_capture()
    cherrypy.engine.exit()


def main():
    app = ObjectDetectionApp()
    app.start_capture()
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    config = {
        "/": {
            "tools.sessions.on": True,
            "tools.auth_basic.on": False,
        },
        "/static": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": static_dir,
        }
    }

    cherrypy.config.update({"server.socket_port": 8080})
    cherrypy.tree.mount(app, "/", config)

    print("Starting Count & Vote app on http://localhost:8080")
    print("Press Ctrl+C to stop the server")

    signal_handler = SignalHandler(cherrypy.engine)
    exit_fun = lambda: app_exit(app)
    signal_handler.handlers["SIGTERM"] = exit_fun
    signal_handler.handlers["SIGHUP"] = exit_fun
    signal_handler.handlers["SIGQUIT"] = exit_fun
    signal_handler.handlers["SIGINT"] = exit_fun
    signal_handler.subscribe()

    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    main()
