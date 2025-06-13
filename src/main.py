import os
import argparse
import math
import collections
from dataclasses import dataclass, field
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR, NAS

# --- Constants ---
DEFAULT_TARGET_CLASSES = ["backpack", "handbag", "suitcase"]
DEFAULT_PERSON_CLASS = "person"
DEFAULT_MASK_DECAY = 1
DEFAULT_MASK_INIT = 128
DEFAULT_MASK_THRESH = 0
DEFAULT_STATIONARY_IOU = 0.75
DEFAULT_FPS = 10.0

# --- Helper Functions ---
def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

# --- Config ---
@dataclass
class AppConfig:
    target_fps: float = DEFAULT_FPS
    history_len_seconds: float = 7.0
    min_frames_initial_movement_check_seconds: float = 1.0
    initial_movement_centroid_threshold_pixels: int = 25
    stationary_check_window_duration_sec: float = 1.0
    stationary_check_min_presence_ratio: float = 0.7
    stationary_iou_threshold: float = DEFAULT_STATIONARY_IOU
    person_class_name: str = DEFAULT_PERSON_CLASS
    mask_decay_rate: int = DEFAULT_MASK_DECAY
    mask_initial_value: int = DEFAULT_MASK_INIT
    mask_threshold: int = DEFAULT_MASK_THRESH
    target_object_classes: list[str] = field(default_factory=lambda: DEFAULT_TARGET_CLASSES)
    history_len_frames: int = field(init=False)
    min_frames_initial_movement_check: int = field(init=False)
    stationary_check_window_frames: int = field(init=False)
    stationary_check_min_samples_in_window: int = field(init=False)
    display_video: bool = False

    def __post_init__(self):
        self.history_len_frames = int(self.target_fps * self.history_len_seconds)
        self.min_frames_initial_movement_check = int(self.target_fps * self.min_frames_initial_movement_check_seconds)
        self.stationary_check_window_frames = int(self.target_fps * self.stationary_check_window_duration_sec)
        self.stationary_check_min_samples_in_window = int(self.stationary_check_window_frames * self.stationary_check_min_presence_ratio)

# --- Detection Model ---
class DetectionModel:
    """Handles model loading and inference."""
    def __init__(self, model_type: str, model_name: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type == "YOLO":
            return YOLO(self.model_name)
        elif self.model_type == "RTDETR":
            return RTDETR(self.model_name)
        elif self.model_type == "NAS":
            return NAS(self.model_name)
        raise ValueError(f"Invalid model type: {self.model_type}")

    def get_model_class_names(self):
        return self.model.names

    def track_objects(self, frame):
        try:
            return self.model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")
        except Exception:
            return self.model(frame, verbose=False)

# --- Tracked Object ---
class TrackedObject:
    """Tracks a single object's state and history."""
    def __init__(self, track_id: int, class_name: str, initial_bbox, initial_centroid, first_seen_frame: int, config: AppConfig):
        self.track_id = track_id
        self.config = config
        self.bboxes = collections.deque(maxlen=config.history_len_frames)
        self.centroids = collections.deque(maxlen=config.history_len_frames)
        self.frames_seen_in = collections.deque(maxlen=config.history_len_frames)
        self.class_names = collections.deque(maxlen=config.history_len_frames)
        self.last_seen_frame = first_seen_frame
        self.first_seen_frame = first_seen_frame
        self.initial_centroids_for_movement_check = collections.deque(maxlen=config.min_frames_initial_movement_check)
        self.has_moved_significantly = False
        self.is_currently_stationary = False
        self.stationary_since_frame = None
        self.is_abandoned = False
        self.abandoned_at_frame: int | None = None
        self.abandoned_at_real_frame: int | None = None
        self.abandoned_bbox: list | None = None
        self.abandoned_reason: str | None = None
        self.display_label = class_name
        self.abandoned_max_frames = int(self.config.target_fps)
        self.last_seen_while_abandoned = None
        self.update_history(initial_bbox, initial_centroid, first_seen_frame, class_name)

    def update_history(self, bbox, centroid, processed_frame_num: int, class_name: str):
        self.bboxes.append(bbox)
        self.centroids.append(centroid)
        self.frames_seen_in.append(processed_frame_num)
        self.class_names.append(class_name)
        self.last_seen_frame = processed_frame_num
        self.class_name = class_name
        if not self.has_moved_significantly:
            self.initial_centroids_for_movement_check.append(centroid)

    def get_current_class_name(self):
        return self.class_names[-1] if self.class_names else self.class_name

    def get_current_bbox(self):
        return self.bboxes[-1] if self.bboxes else None

    def check_initial_movement(self):
        if self.has_moved_significantly:
            return True
        if len(self.initial_centroids_for_movement_check) == self.config.min_frames_initial_movement_check:
            first_c = self.initial_centroids_for_movement_check[0]
            last_c = self.initial_centroids_for_movement_check[-1]
            if euclidean_distance(first_c, last_c) > self.config.initial_movement_centroid_threshold_pixels:
                self.has_moved_significantly = True
            self.initial_centroids_for_movement_check.clear()
        return self.has_moved_significantly

    def check_stationarity(self, current_frame: int):
        if not self.has_moved_significantly or not self.bboxes or len(self.bboxes) < 2:
            self.is_currently_stationary = False
            self.stationary_since_frame = None
            return False
        current_bbox = self.bboxes[-1]
        prev_bbox = self.bboxes[-2]
        iou = calculate_iou(current_bbox, prev_bbox)
        if iou >= self.config.stationary_iou_threshold:
            if not self.is_currently_stationary:
                self.is_currently_stationary = True
                self.stationary_since_frame = current_frame
        else:
            self.is_currently_stationary = False
            self.stationary_since_frame = None
        return self.is_currently_stationary

    def update_abandoned_state(self, current_frame: int):
        if self.is_abandoned:
            if self.bboxes:
                self.last_seen_while_abandoned = current_frame
            elif self.last_seen_while_abandoned is not None:
                frames_not_seen = current_frame - self.last_seen_while_abandoned
                if frames_not_seen >= self.abandoned_max_frames:
                    self.is_abandoned = False
                    self.abandoned_at_frame = None
                    self.abandoned_at_real_frame = None
                    self.abandoned_bbox = None
                    self.abandoned_reason = None
                    self.display_label = self.class_name
                    self.last_seen_while_abandoned = None

    def mark_as_abandoned(self, frame_num: int, real_processed_frame: int, reason: str):
        if not self.is_abandoned:
            self.is_abandoned = True
            self.abandoned_at_frame = frame_num
            self.abandoned_at_real_frame = real_processed_frame
            self.abandoned_bbox = self.get_current_bbox()
            self.abandoned_reason = reason
            self.display_label = f"ABANDONED ({reason}) {self.class_name} ID:{self.track_id}"
            self.last_seen_while_abandoned = frame_num

    def get_display_info(self, confidence: float | None = None):
        if not self.bboxes:
            return None, None, None
        bbox = self.bboxes[-1]
        current_class = self.get_current_class_name()
        if self.is_abandoned:
            label = f"ID:{self.track_id}, ABANDONED ({self.abandoned_reason}) {current_class}"
        else:
            label = f" ID:{self.track_id} {current_class}"
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        color = (0, 255, 0) if not self.is_abandoned else (0, 0, 255)
        return bbox, label, color

# --- Candidate Abandoned Object ---
@dataclass
class CandidateAbandonedObject:
    track_id: int
    class_name: str
    bbox: list
    first_seen_frame: int
    last_seen_frame: int
    is_confirmed: bool = False

# --- Abandonment Detector ---
class AbandonmentDetector:
    """Handles mask logic and abandoned object logic."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.model_class_names = []
        self.abandoned_events_log: list[dict] = []
        self.person_mask = None
        self.frame_height = None
        self.frame_width = None
        self.candidate_objects: dict[int, CandidateAbandonedObject] = {}

    def initialize_mask(self, height: int, width: int):
        self.frame_height = height
        self.frame_width = width
        self.person_mask = np.zeros((height, width), dtype=np.int32)

    def update_person_mask(self, detections_results):
        if self.person_mask is None:
            return
        self.person_mask = np.maximum(0, self.person_mask - self.config.mask_decay_rate)
        if detections_results and detections_results[0].boxes is not None:
            for box_data in detections_results[0].boxes:
                cls_id = int(box_data.cls[0])
                class_name = self.model_class_names[cls_id] if self.model_class_names and cls_id < len(self.model_class_names) else "Unknown"
                if class_name == self.config.person_class_name:
                    bbox = list(map(int, box_data.xyxy[0]))
                    x1, y1, x2, y2 = bbox
                    self.person_mask[y1:y2, x1:x2] = self.config.mask_initial_value

    def check_mask_intersection(self, bbox):
        if self.person_mask is None:
            return False
        x1, y1, x2, y2 = bbox
        mask_region = self.person_mask[y1:y2, x1:x2]
        return np.any(mask_region > self.config.mask_threshold)

    def set_model_class_names(self, names_list):
        self.model_class_names = names_list

    def get_abandoned_events_log(self):
        return self.abandoned_events_log

    def process_detections(self, detections_results, processed_frame_num: int, current_frame_num: int):
        current_frame_all_tracked_ids = set()
        self.update_person_mask(detections_results)
        if detections_results and detections_results[0].boxes is not None:
            for box_data in detections_results[0].boxes:
                track_id = int(box_data.id[0]) if box_data.id is not None and len(box_data.id) > 0 else None
                if track_id is None:
                    continue
                current_frame_all_tracked_ids.add(track_id)
                cls_id = int(box_data.cls[0])
                class_name = self.model_class_names[cls_id] if self.model_class_names and cls_id < len(self.model_class_names) else "Unknown"
                conf = float(box_data.conf[0])
                bbox = list(map(int, box_data.xyxy[0]))
                centroid = get_centroid(bbox)
                if conf < 0.5:
                    continue
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = TrackedObject(track_id, class_name, bbox, centroid, processed_frame_num, self.config)
                else:
                    obj = self.tracked_objects[track_id]
                    obj.update_history(bbox, centroid, processed_frame_num, class_name)
        for track_id, obj in list(self.tracked_objects.items()):
            is_seen_this_frame = track_id in current_frame_all_tracked_ids
            if obj.class_name not in self.config.target_object_classes:
                if not is_seen_this_frame and (processed_frame_num - obj.last_seen_frame > self.config.history_len_frames * 2):
                    del self.tracked_objects[track_id]
                continue
            if is_seen_this_frame and not obj.is_abandoned:
                current_bbox = obj.get_current_bbox()
                if current_bbox and self.check_mask_intersection(current_bbox):
                    if track_id not in self.candidate_objects:
                        self.candidate_objects[track_id] = CandidateAbandonedObject(
                            track_id=track_id,
                            class_name=obj.class_name,
                            bbox=current_bbox,
                            first_seen_frame=processed_frame_num,
                            last_seen_frame=processed_frame_num
                        )
                    else:
                        self.candidate_objects[track_id].last_seen_frame = processed_frame_num
                        self.candidate_objects[track_id].bbox = current_bbox
                else:
                    if track_id in self.candidate_objects:
                        candidate = self.candidate_objects[track_id]
                        if not candidate.is_confirmed:
                            candidate.is_confirmed = True
                            obj.mark_as_abandoned(processed_frame_num, current_frame_num, "person_left")
                            del self.candidate_objects[track_id]
            if not obj.is_abandoned:
                if not obj.has_moved_significantly:
                    obj.check_initial_movement()
                if obj.has_moved_significantly:
                    if is_seen_this_frame:
                        obj.check_stationarity(processed_frame_num)
                    else:
                        if not obj.is_abandoned:
                            obj.is_currently_stationary = False
                            obj.stationary_since_frame = None
            if obj.is_abandoned:
                if obj.abandoned_bbox and obj.abandoned_at_frame is not None:
                    log_entry = {
                        "frame_id": processed_frame_num,
                        "real_frame_id": current_frame_num,
                        "bbox": obj.get_current_bbox(),
                        "class_name": obj.class_name,
                        "track_id": obj.track_id,
                        "reason": obj.abandoned_reason
                    }
                    self.abandoned_events_log.append(log_entry)
                obj.update_abandoned_state(processed_frame_num)
            if not is_seen_this_frame and not obj.is_abandoned and \
               (processed_frame_num - obj.last_seen_frame > self.config.history_len_frames * 2):
                del self.tracked_objects[track_id]

    def get_objects_for_display(self):
        display_data = []
        for track_id, obj in self.tracked_objects.items():
            if obj.is_abandoned or obj.class_name in self.config.target_object_classes:
                bbox, label_text, color = obj.get_display_info(confidence=None)
                if bbox:
                    if obj.is_abandoned:
                        color = (0, 0, 255)
                    display_data.append({'bbox': bbox, 'label': label_text, 'color': color})
        for track_id, candidate in self.candidate_objects.items():
            if not candidate.is_confirmed:
                display_data.append({
                    'bbox': candidate.bbox,
                    'label': f"Candidate {candidate.class_name}",
                    'color': (0, 255, 255)
                })
        return display_data

# --- Video Processor ---
class VideoProcessor:
    """Handles video reading, frame processing, and visualization."""
    def __init__(self, video_path: str, model_type: str, model_name: str, config: AppConfig):
        self.video_path = video_path
        self.config = config
        self.detector = DetectionModel(model_type, model_name)
        self.abandonment_handler = AbandonmentDetector(config)
        self.abandonment_handler.set_model_class_names(self.detector.get_model_class_names())
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open video file {video_path}")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.process_nth_frame = 1
        if self.original_fps and self.original_fps > self.config.target_fps:
            self.process_nth_frame = int(round(self.original_fps / self.config.target_fps))
        self.process_nth_frame = max(1, self.process_nth_frame)
        self.abandonment_handler.initialize_mask(self.frame_height, self.frame_width)
        self._print_fps_info()

    def _print_fps_info(self):
        formatted_fps = f"{self.original_fps:.2f}" if self.original_fps else "N/A"
        print(f"Original video FPS: {formatted_fps}")
        if self.process_nth_frame > 1:
            print(f"Processing every {self.process_nth_frame}-th frame to target {self.config.target_fps:.0f} FPS.")
        else:
            print(f"Processing all frames (original FPS not significantly above {self.config.target_fps:.0f} FPS or FPS info unavailable).")

    def run(self):
        total_frames_read = 0
        processed_frames_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            total_frames_read += 1
            if total_frames_read % self.process_nth_frame != 0:
                continue
            processed_frames_count += 1
            current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            detection_results = self.detector.track_objects(frame)
            self.abandonment_handler.process_detections(detection_results, processed_frames_count, current_frame_num)
            if self.config.display_video:
                output_frame = frame.copy()
                current_detections_this_frame = {}
                if detection_results and detection_results[0].boxes is not None:
                    for box_data in detection_results[0].boxes:
                        track_id = int(box_data.id[0]) if box_data.id is not None and len(box_data.id) > 0 else None
                        if track_id:
                            current_detections_this_frame[track_id] = float(box_data.conf[0])
                if self.abandonment_handler.person_mask is not None:
                    mask_normalized = self.abandonment_handler.person_mask.astype(np.float32) / self.config.mask_initial_value
                    mask_normalized = np.clip(mask_normalized, 0, 1)
                    colored_mask = np.zeros_like(output_frame)
                    colored_mask[..., 2] = (mask_normalized * 255).astype(np.uint8)
                    alpha = 0.5
                    output_frame = cv2.addWeighted(output_frame, 1, colored_mask, alpha, 0)
                for track_id, obj_instance in self.abandonment_handler.tracked_objects.items():
                    should_draw_box = obj_instance.is_abandoned or (obj_instance.last_seen_frame == processed_frames_count)
                    if should_draw_box and obj_instance.bboxes:
                        current_conf = current_detections_this_frame.get(track_id)
                        bbox, label_text, color = obj_instance.get_display_info(confidence=current_conf)
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                            if label_text:
                                cv2.putText(output_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow('Abandoned Object Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        if self.config.display_video:
            cv2.destroyAllWindows()
        print("Video processing complete.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Detect abandoned objects in a video.')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--model-type', type=str, required=True, choices=['YOLO', 'RTDETR'], help="Model type: YOLO or RTDETR.")
    parser.add_argument('--model', type=str, required=True, help="Name of the AI model (e.g., 'yolov8n.pt', 'rtdetr-l.pt').")
    parser.add_argument('--target-fps', type=float, help="Override target processing FPS.")
    parser.add_argument('--iou-thresh', type=float, help="Override stationary IoU threshold.")
    parser.add_argument('--display-video', action='store_true', help="Enable video display during processing.")
    args = parser.parse_args()
    cli_config_overrides = {}
    if args.target_fps is not None: cli_config_overrides['target_fps'] = args.target_fps
    if args.iou_thresh is not None: cli_config_overrides['stationary_iou_threshold'] = args.iou_thresh
    if args.display_video: cli_config_overrides['display_video'] = True
    app_config = AppConfig(**cli_config_overrides)
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
    try:
        processor = VideoProcessor(
            video_path=args.video_path,
            model_type=args.model_type,
            model_name=args.model,
            config=app_config
        )
        processor.run()
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
