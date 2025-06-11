import os
import argparse
import glob
import cv2
from main import AppConfig, VideoProcessor # Assuming main.py is in the same directory or accessible in PYTHONPATH

# Define model configurations
# 'name' is for the output directory
# 'type' is for the DetectionModel class ('YOLO' or 'RTDETR')
# 'file' is the actual model file name (e.g., yolov8n.pt)
MODEL_CONFIGS = [
    {'name': 'yolov10l', 'type': 'YOLO', 'file': 'yolov10l.pt'}, # Example for yolov10
    {'name': 'yolo_nas_l', 'type': 'NAS', 'file': 'yolo_nas_l.pt'}, # Example for yolov10
    {'name': 'rtdetr-l', 'type': 'RTDETR', 'file': 'rtdetr-l.pt'}, # Example for RTDETR
    {'name': 'yolo12l', 'type': 'YOLO', 'file': 'yolo12l.pt'}, 
    {'name': 'yolo11l', 'type': 'YOLO', 'file': 'yolo11l.pt'},
]

# Common video extensions
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv']

def run_test(videos_directory: str, results_base_dir: str = "results", model_name: str = ""):
    if not os.path.isdir(videos_directory):
        print(f"Error: Videos directory not found at {videos_directory}")
        return

    os.makedirs(results_base_dir, exist_ok=True)

    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(videos_directory, ext)))
    
    if not video_files:
        print(f"No video files found in {videos_directory}")
        return

    print(f"Found {len(video_files)} videos to process.")
    if model_name:
        model_configs = [model_config for model_config in MODEL_CONFIGS if model_config['name'] == model_name]
    else:   
        model_configs = MODEL_CONFIGS

    for model_config in model_configs:
        model_name_for_dir = model_config['name']
        model_type = model_config['type']
        model_file = model_config['file']
        
        model_results_dir = os.path.join(results_base_dir, model_name_for_dir)
        os.makedirs(model_results_dir, exist_ok=True)
        
        print(f"\n--- Testing Model: {model_name_for_dir} (File: {model_file}, Type: {model_type}) ---")

        for video_path in video_files:
            video_filename = os.path.basename(video_path)
            output_txt_filename = os.path.splitext(video_filename)[0] + '.txt'
            output_txt_path = os.path.join(model_results_dir, output_txt_filename)

            print(f"  Processing video: {video_filename}...")

            # Configure AppConfig for testing (no display, etc.)
            # You might want to adjust other AppConfig defaults here if needed for specific tests
            test_app_config = AppConfig(display_video=False) 
                                        # Add other overrides if necessary, e.g. target_fps for consistency
                                        # test_app_config.target_fps = 10 
                                        # test_app_config.__post_init__() # Recalculate frame counts if target_fps changes

            try:
                # Each video processing should be independent, so new processor instance
                processor = VideoProcessor(
                    video_path=video_path,
                    model_type=model_type,
                    model_name=model_file, # Use the 'file' for the model name parameter
                    config=test_app_config
                )
                processor.run() # This will populate the logs in abandonment_handler

                abandoned_objects_log = processor.abandonment_handler.get_abandoned_events_log()
                
                # Get video dimensions for normalization
                video_width = processor.frame_width
                video_height = processor.frame_height
                print(f"Video dimensions: {video_width}x{video_height}")
                if video_width == 0 or video_height == 0:
                    print(f"Error: Video dimensions are 0x0 for {video_path}")
                    continue
                
                with open(output_txt_path, 'w') as f:
                    for event in abandoned_objects_log:
                        bbox = event['bbox'] # [x1, y1, x2, y2]
                        
                        # Convert to YOLO 1.1 format: class_id center_x center_y width height
                        # All coordinates normalized (0-1)
                        center_x = ((bbox[0] + bbox[2]) / 2) / video_width
                        center_y = ((bbox[1] + bbox[3]) / 2) / video_height
                        width = (bbox[2] - bbox[0]) / video_width
                        height = (bbox[3] - bbox[1]) / video_height
                        
                        # class_id = 0 for abandoned objects
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                print(f"Results saved to: {output_txt_path} ({len(abandoned_objects_log)} abandoned events)")
                
                # Clear logs in the handler if we were to reuse the same processor instance,
                # but since we create a new VideoProcessor, its AbandonmentDetector is also new.
                # processor.abandonment_handler.clear_logs() # Not strictly necessary here

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    Error processing video {video_filename} with model {model_name_for_dir}: {e}")
                # Optionally, log this error to a file as well
                error_log_path = os.path.join(model_results_dir, "_errors.log")
                with open(error_log_path, 'a') as f_err:
                    f_err.write(f"Error processing {video_filename}: {e}\n")

    print("\n--- All tests completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run abandoned object detection tests for multiple models and videos.")
    parser.add_argument("--videos-dir", type=str, help="Directory containing video files to process.")
    parser.add_argument("--results-dir", type=str, default="results", help="Base directory to save test results.")
    parser.add_argument("--model", type=str, default="", help="Path to the model configurations file.")
    
    args = parser.parse_args()
    run_test(args.videos_dir, args.results_dir, args.model) 