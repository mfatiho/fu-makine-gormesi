import os
import argparse
from ultralytics import YOLO, RTDETR, NAS
import cv2
import math
import collections
from dataclasses import dataclass, field

# Define the target classes
# Person is included for abandonment logic
TARGET_CLASSES = ["backpack", "handbag", "suitcase"]

# Helper functions
def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union
    denominator = float(boxAArea + boxBArea - interArea)
    iou = interArea / denominator if denominator > 0 else 0.0
    return iou

# --- Configuration ---
@dataclass
class AppConfig:
    target_fps: float = 10.0
    history_len_seconds: float = 7.0
    min_frames_initial_movement_check_seconds: float = 1.0
    initial_movement_centroid_threshold_pixels: int = 25
    stationary_check_window_duration_sec: float = 1.0
    stationary_check_min_presence_ratio: float = 0.7 # Min % of frames object must be present in window
    stationary_iou_threshold: float = 0.75
    abandonment_duration_sec: float = 5.0 # General abandonment timer

    # New: Person-related abandonment config
    person_class_name: str = "person"
    person_association_distance_threshold_pixels: int = 75 # How close a person needs to be to an object
    person_disappearance_timeout_seconds: float = 3.0 # How long person can be gone before association is broken
    abandonment_after_person_leaves_duration_seconds: float = 1.0 # How long object must be stationary AFTER person confirmed left
    
    target_object_classes: list[str] = field(default_factory=lambda: ["backpack", "handbag", "suitcase"])
    
    # Derived properties
    history_len_frames: int = field(init=False)
    min_frames_initial_movement_check: int = field(init=False)
    stationary_check_window_frames: int = field(init=False)
    stationary_check_min_samples_in_window: int = field(init=False)
    abandonment_duration_frames: int = field(init=False)
    person_disappearance_timeout_frames: int = field(init=False)
    abandonment_after_person_leaves_duration_frames: int = field(init=False)
    display_video: bool = False # New parameter for controlling display

    def __post_init__(self):
        self.history_len_frames = int(self.target_fps * self.history_len_seconds)
        self.min_frames_initial_movement_check = int(self.target_fps * self.min_frames_initial_movement_check_seconds)
        self.stationary_check_window_frames = int(self.target_fps * self.stationary_check_window_duration_sec)
        self.stationary_check_min_samples_in_window = int(self.stationary_check_window_frames * self.stationary_check_min_presence_ratio)
        self.abandonment_duration_frames = int(self.target_fps * self.abandonment_duration_sec)
        self.person_disappearance_timeout_frames = int(self.target_fps * self.person_disappearance_timeout_seconds)
        self.abandonment_after_person_leaves_duration_frames = int(self.target_fps * self.abandonment_after_person_leaves_duration_seconds)

# --- Detection Model ---
class DetectionModel:
    def __init__(self, model_type: str, model_name: str):
        self.model_name = model_name
        self.model_type = model_type
        try:
            if model_type == "YOLO":
                self.model = YOLO(model_name)
            elif model_type == "RTDETR":
                self.model = RTDETR(model_name)
            elif model_type == "NAS":
                self.model = NAS(model_name)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            print(f"Successfully loaded {model_type} model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name} of type {model_type}: {e}")
            raise # Re-raise to stop execution if model fails to load

    def get_model_class_names(self):
        return self.model.names

    def track_objects(self, frame):
        try:
            # Track all classes, AbandonmentDetector will filter for persons vs target items
            return self.model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return self.model(frame, verbose=False)


# --- Tracked Object ---
class TrackedObject:
    def __init__(self, track_id: int, class_name: str, initial_bbox, initial_centroid, first_seen_frame: int, config: AppConfig):
        self.track_id = track_id
        self.class_name = class_name
        self.config = config
        
        self.bboxes = collections.deque(maxlen=config.history_len_frames)
        self.centroids = collections.deque(maxlen=config.history_len_frames)
        self.frames_seen_in = collections.deque(maxlen=config.history_len_frames) # Processed frame numbers

        self.last_seen_frame = first_seen_frame
        self.first_seen_frame = first_seen_frame
        
        self.initial_centroids_for_movement_check = collections.deque(maxlen=config.min_frames_initial_movement_check)
        self.has_moved_significantly = False
        
        self.is_currently_stationary = False
        self.stationary_since_frame = None # Processed frame number when current stationary period started
        self.is_abandoned = False
        self.abandoned_at_frame: int | None = None # New: Store frame when marked abandoned
        self.abandoned_at_real_frame: int | None = None # New: Store frame when marked abandoned
        self.abandoned_bbox: list | None = None    # New: Store bbox when marked abandoned
        self.abandoned_reason: str | None = None   # New: Store reason
        self.display_label = class_name # Default label

        # New attributes for person association
        self.associated_person_id: int | None = None
        self.last_frame_person_proximate: int | None = None
        self.person_confirmed_departed_since_frame: int | None = None # Frame when person disappearance timeout was met

        self.update_history(initial_bbox, initial_centroid, first_seen_frame)

    def update_history(self, bbox, centroid, processed_frame_num: int):
        self.bboxes.append(bbox)
        self.centroids.append(centroid)
        self.frames_seen_in.append(processed_frame_num)
        self.last_seen_frame = processed_frame_num

        if not self.has_moved_significantly:
            self.initial_centroids_for_movement_check.append(centroid)

    def check_initial_movement(self):
        if self.has_moved_significantly: # Already confirmed
            return True
        
        if len(self.initial_centroids_for_movement_check) == self.config.min_frames_initial_movement_check:
            first_c = self.initial_centroids_for_movement_check[0]
            last_c = self.initial_centroids_for_movement_check[-1]
            if euclidean_distance(first_c, last_c) > self.config.initial_movement_centroid_threshold_pixels:
                self.has_moved_significantly = True
                print(f"Object ID {self.track_id} ({self.class_name}) confirmed to have moved significantly.")
            # Clear after check to prevent re-check or continuous growth if logic changes
            self.initial_centroids_for_movement_check.clear()
        return self.has_moved_significantly

    def check_stationarity(self, current_processed_frame: int):
        if not self.has_moved_significantly:
            return False # Cannot be stationary for abandonment if it hasn't moved first

        window_bboxes = []
        window_frames = []
        for i in range(len(self.frames_seen_in) - 1, -1, -1):
            frame_in_hist = self.frames_seen_in[i]
            if current_processed_frame - frame_in_hist < self.config.stationary_check_window_frames:
                window_frames.append(frame_in_hist)
                window_bboxes.append(self.bboxes[i])
            else:
                break
        window_bboxes.reverse()
        window_frames.reverse()

        is_stationary_this_check = False
        if len(window_bboxes) >= self.config.stationary_check_min_samples_in_window:
            # Compare current bbox (latest in window_bboxes) with earliest bbox in that window
            current_bbox_in_hist = self.bboxes[-1] # Should be same as window_bboxes[-1]
            earliest_bbox_in_window = window_bboxes[0]
            
            iou = calculate_iou(current_bbox_in_hist, earliest_bbox_in_window)
            if iou >= self.config.stationary_iou_threshold:
                is_stationary_this_check = True

        if is_stationary_this_check:
            self.is_currently_stationary = True
            if self.stationary_since_frame is None: # Start of new stationary period
                self.stationary_since_frame = window_frames[0] # Frame when this period started
                print(f"Object ID {self.track_id} ({self.class_name}) started stationary period at frame {self.stationary_since_frame}.")
        else: # Not stationary in this check, or not enough samples
            self.is_currently_stationary = False
            self.stationary_since_frame = None
            # If it moves, it's no longer considered for current abandonment period
            if not self.is_abandoned: # Don't reset label if already marked abandoned
                 self.display_label = self.class_name
        
        return self.is_currently_stationary

    def update_person_association(self, current_persons_objects: list['TrackedObject'], current_processed_frame: int):
        if self.is_abandoned or self.class_name == self.config.person_class_name: # Persons don't associate with other persons this way
            return

        if not self.centroids: return 

        my_centroid = self.centroids[-1]
        closest_person_obj = None
        min_dist = float('inf')

        for p_obj in current_persons_objects:
            if not p_obj.centroids: continue
            person_centroid = p_obj.centroids[-1]
            dist = euclidean_distance(my_centroid, person_centroid)
            
            if dist < self.config.person_association_distance_threshold_pixels and dist < min_dist:
                min_dist = dist
                closest_person_obj = p_obj
        
        if closest_person_obj:
            self.associated_person_id = closest_person_obj.track_id
            self.last_frame_person_proximate = current_processed_frame
            self.person_confirmed_departed_since_frame = None # Reset: person is (back) in proximity
            # print(f"Object ID {self.track_id} associated with Person ID {self.associated_person_id} at frame {current_processed_frame}")
        else: # No person proximate in the current frame
            if self.associated_person_id is not None and self.last_frame_person_proximate is not None:
                # If a person was previously associated and is now gone for the timeout period
                if (current_processed_frame - self.last_frame_person_proximate) > self.config.person_disappearance_timeout_frames:
                    if self.person_confirmed_departed_since_frame is None: # Mark departure confirmation time only once
                        self.person_confirmed_departed_since_frame = current_processed_frame
                        # print(f"Object ID {self.track_id}: Person ID {self.associated_person_id} confirmed departed at frame {current_processed_frame}.")
                    # Note: We don't clear self.associated_person_id here, to remember it *had* an owner.
                    # The state is now "owner (ID {self.associated_person_id}) departed at frame X"

    def mark_as_abandoned(self, frame_num: int, real_processed_frame: int, reason: str):
        if not self.is_abandoned: # Mark only once
            self.is_abandoned = True
            self.abandoned_at_frame = frame_num
            self.abandoned_at_real_frame = real_processed_frame
            self.abandoned_bbox = self.get_current_bbox() # Capture current bbox
            self.abandoned_reason = reason
            self.display_label = f"ABANDONED ({reason}) {self.class_name}"
            print(f"!!! Object ID {self.track_id} ({self.class_name}) marked as ABANDONED at frame {frame_num} (Reason: {reason}) !!!")

    def check_abandonment(self, current_processed_frame: int, real_processed_frame: int | None = None):
        if self.is_abandoned: return True
        if not self.is_currently_stationary: return False # Must be stationary for any abandonment rule based on it

        # Rule 1: Standard stationary timer (long duration)
        if self.stationary_since_frame is not None and \
           (current_processed_frame - self.stationary_since_frame) >= self.config.abandonment_duration_frames:
            self.mark_as_abandoned(current_processed_frame, real_processed_frame, "timer")
            return True

        # Rule 2: Person departed, and object stationary for a (potentially shorter) period since person departure
        if self.associated_person_id is not None and \
           self.person_confirmed_departed_since_frame is not None and \
           self.stationary_since_frame is not None:
            
            # Effective start of "abandonment after person left" check:
            # When the object became stationary OR when the person was confirmed departed, WHICHEVER IS LATER.
            # The object must have been stationary for the required duration *since* this effective start.
            effective_check_start_frame = max(self.stationary_since_frame, self.person_confirmed_departed_since_frame)

            if (current_processed_frame - effective_check_start_frame) >= self.config.abandonment_after_person_leaves_duration_frames:
                self.mark_as_abandoned(current_processed_frame, real_processed_frame, "person_left")
                return True
            
        return False

    def get_current_bbox(self):
        return self.bboxes[-1] if self.bboxes else None
    
    def get_display_info(self, confidence=None):
        label_text = None # Default to no label
        color = (0, 255, 0) # Default green

        if self.is_abandoned:
            label_text = f"{self.display_label} ID:{self.track_id}" # self.display_label is already set by mark_as_abandoned
            color = (0, 0, 255) # Red for abandoned
        elif self.is_currently_stationary and self.has_moved_significantly:
            color = (0, 255, 255) # Yellow for stationary (but no label text)
        # Else, color remains green (for actively tracked, non-stationary, non-abandoned)
        
        return self.get_current_bbox(), label_text, color

# --- Abandonment Detector ---
class AbandonmentDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.model_class_names = [] # To be set by VideoProcessor after model loads
        self.abandoned_events_log: list[dict] = [] # New: For logging abandonment events

    def set_model_class_names(self, names_list):
        self.model_class_names = names_list

    def clear_logs(self): # New method
        self.abandoned_events_log = []
        # Potentially clear states of tracked_objects if this instance is reused for multiple videos
        # For now, test.py will create a new VideoProcessor (and thus new AbandonmentDetector) per video.
        # If reusing, would need: self.tracked_objects = {}

    def get_abandoned_events_log(self): # New method
        return self.abandoned_events_log

    def process_detections(self, detections_results, processed_frame_num: int, current_frame_num: int):
        current_frame_all_tracked_ids = set()
        current_frame_persons_list: list[TrackedObject] = []

        if detections_results and detections_results[0].boxes is not None:
            boxes = detections_results[0].boxes
            for box_data in boxes:
                track_id = int(box_data.id[0]) if box_data.id is not None and len(box_data.id) > 0 else None
                if track_id is None: 
                    continue # Cannot process without track_id
                
                current_frame_all_tracked_ids.add(track_id)
                cls_id = int(box_data.cls[0])
                class_name = self.model_class_names[cls_id] if self.model_class_names and cls_id < len(self.model_class_names) else "Unknown"
                conf = float(box_data.conf[0])
                bbox = list(map(int, box_data.xyxy[0]))
                centroid = get_centroid(bbox)

                if conf < 0.5: # Basic confidence filter
                    continue

                if track_id not in self.tracked_objects:
                    # Create new TrackedObject for any class, including persons
                    self.tracked_objects[track_id] = TrackedObject(track_id, class_name, bbox, centroid, processed_frame_num, self.config)
                else:
                    obj = self.tracked_objects[track_id]
                    obj.class_name = class_name # Tracker might re-ID with a different class
                    obj.update_history(bbox, centroid, processed_frame_num)
                
                # Collect current persons
                if self.tracked_objects[track_id].class_name == self.config.person_class_name:
                    current_frame_persons_list.append(self.tracked_objects[track_id])
        
        # --- Logic Application ---
        for track_id, obj in list(self.tracked_objects.items()): 
            is_seen_this_frame = track_id in current_frame_all_tracked_ids
            
            # Store previous abandoned state to detect transition
            was_abandoned_before_check = obj.is_abandoned

            # 1. Update Person Association (for target objects)
            if obj.class_name in self.config.target_object_classes and not obj.is_abandoned:
                obj.update_person_association(current_frame_persons_list, processed_frame_num)

            # 2. Process only target objects for abandonment logic
            if obj.class_name not in self.config.target_object_classes:
                if not is_seen_this_frame and (processed_frame_num - obj.last_seen_frame > self.config.history_len_frames * 2):
                     # print(f"Removing stale non-target object ID {track_id} ({obj.class_name}) last seen at frame {obj.last_seen_frame}")
                     del self.tracked_objects[track_id] # Clean up old non-target objects too
                continue 

            # 3. Abandonment Pipeline for Target Objects
            if not obj.is_abandoned: # Don't re-evaluate if already abandoned
                if not obj.has_moved_significantly:
                    obj.check_initial_movement()
                
                if obj.has_moved_significantly:
                    if is_seen_this_frame:
                        obj.check_stationarity(processed_frame_num) # Updates obj.is_currently_stationary
                        obj.check_abandonment(processed_frame_num, current_frame_num)   # Uses new person association state
                    else: # Target object not seen this frame
                        if not obj.is_abandoned : # If it wasn't abandoned yet
                            obj.is_currently_stationary = False # Assume movement if not seen
                            obj.stationary_since_frame = None
                            # Person association timeout will naturally occur if person also not seen
                            # obj.display_label = obj.class_name # Reset label?
            
            # NEW: Log if object just became abandoned in this frame's processing
            if obj.is_abandoned and not was_abandoned_before_check:
                if obj.abandoned_bbox and obj.abandoned_at_frame is not None: # Ensure these were set
                    log_entry = {
                        "frame_id": obj.abandoned_at_frame, # Use the frame it was marked abandoned
                        "bbox": obj.abandoned_bbox,         # Use the bbox at abandonment
                        "class_name": obj.class_name,
                        "track_id": obj.track_id,
                        "reason": obj.abandoned_reason
                    }
                    self.abandoned_events_log.append(log_entry)
            
            # Clean up old target objects not seen for a very long time AND not abandoned
            if not is_seen_this_frame and not obj.is_abandoned and \
               (processed_frame_num - obj.last_seen_frame > self.config.history_len_frames * 2) :
                 # print(f"Removing stale target object ID {track_id} ({obj.class_name}) last seen at frame {obj.last_seen_frame}")
                 del self.tracked_objects[track_id]


    def get_objects_for_display(self):
        display_data = []
        # Confidence is tricky here as it's per-detection, TrackedObject is an aggregation.
        # For simplicity, confidence is not passed to get_display_info from here.
        # It will be added if TrackedObject has a way to store current frame confidence if seen.
        for track_id, obj in self.tracked_objects.items():
            # Only display objects that are either targets, or persons if you want to see them too
            # For now, let's focus on displaying target_object_classes and persons seen recently
            is_target = obj.class_name in self.config.target_object_classes
            is_person = obj.class_name == self.config.person_class_name


            # We need processed_frame_num to check 'seen_recently' effectively here.
            # This method is called outside the main loop context where processed_frame_num is available.
            # For now, let's just display if abandoned or if it's a target object.
            # A better way would be to pass processed_frame_num to this method.
            # Or, filter for display in VideoProcessor where processed_frame_num is available.

            if obj.is_abandoned or is_target: # Show all target items, and any abandoned items
                bbox, label_text, color = obj.get_display_info(confidence=None) 
                if bbox:
                    display_data.append({'bbox': bbox, 'label': label_text, 'color': color})
            elif is_person and obj.bboxes: # Optionally display persons if they have current bbox
                bbox, label_text, color = obj.get_display_info(confidence=None)
                if bbox:
                     display_data.append({'bbox': bbox, 'label': label_text, 'color': color})


        return display_data

# --- Video Processor ---
class VideoProcessor:
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
                # cv2.imshow('Frame', frame) # Optionally display skipped frames
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            
            processed_frames_count += 1
            # get current frame number in the video
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect abandoned objects in a video using SOLID principles.')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--model-type', type=str, required=True, choices=['YOLO', 'RTDETR'], help="Model type: YOLO or RTDETR.")
    parser.add_argument('--model', type=str, required=True, help="Name of the AI model (e.g., 'yolov8n.pt', 'rtdetr-l.pt').")
    
    # Allow overriding some config parameters from CLI for experimentation
    parser.add_argument('--target-fps', type=float, help="Override target processing FPS.")
    parser.add_argument('--abandonment-sec', type=float, help="Override general abandonment duration in seconds.")
    parser.add_argument('--iou-thresh', type=float, help="Override stationary IoU threshold.")
    parser.add_argument('--person-assoc-dist', type=int, help="Override person association distance in pixels.")
    parser.add_argument('--person-disappear-timeout', type=float, help="Override person disappearance timeout in seconds.")
    parser.add_argument('--abandon-after-person-left-sec', type=float, help="Override abandonment (stationary) duration after person leaves.")
    parser.add_argument('--display-video', action='store_true', help="Enable video display during processing.")

    args = parser.parse_args()

    # Create config, potentially overriding defaults with CLI args
    cli_config_overrides = {}
    if args.target_fps is not None: cli_config_overrides['target_fps'] = args.target_fps
    if args.abandonment_sec is not None: cli_config_overrides['abandonment_duration_sec'] = args.abandonment_sec
    if args.iou_thresh is not None: cli_config_overrides['stationary_iou_threshold'] = args.iou_thresh
    if args.person_assoc_dist is not None: cli_config_overrides['person_association_distance_threshold_pixels'] = args.person_assoc_dist
    if args.person_disappear_timeout is not None: cli_config_overrides['person_disappearance_timeout_seconds'] = args.person_disappear_timeout
    if args.abandon_after_person_left_sec is not None: cli_config_overrides['abandonment_after_person_leaves_duration_seconds'] = args.abandon_after_person_left_sec
    if args.display_video: cli_config_overrides['display_video'] = True
    
    app_config = AppConfig(**cli_config_overrides)


    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
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
