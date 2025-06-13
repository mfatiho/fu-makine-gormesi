import os
import zipfile
import argparse
import pandas as pd
from pathlib import Path

class AnnotationHandler:
    def __init__(self, pred_file: str, gt_zip_file: str, folder_inside_zip: str = "obj_train_data"):
        self.pred_file = pred_file
        self.gt_zip_file = gt_zip_file
        self.folder_inside_zip = folder_inside_zip
        self.gt_dict = self._load_gt_from_zip()
        self.pred_dict = self._load_pred_file()

    def _load_gt_from_zip(self):
        """
        ZIP içindeki obj_train_data klasöründeki tüm YOLO ground truth dosyalarını okur.
        Dönüş: frame_id -> list of [class_id, x, y, w, h]
        """
        gt_dict = {}
        with zipfile.ZipFile(self.gt_zip_file, 'r') as zipf:
            file_list = [f for f in zipf.namelist() if f.startswith(self.folder_inside_zip) and f.endswith(".txt")]
            for file_name in sorted(file_list):
                base = os.path.basename(file_name)
                try:
                    frame_id = int(base.split('_')[-1].split('.')[0])
                except ValueError:
                    continue  # dosya adı çevirilemiyorsa atla

                with zipf.open(file_name) as file:
                    lines = file.read().decode('utf-8').strip().splitlines()
                    boxes = []
                    for line in lines:
                        if line.strip() == "":
                            continue
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id, x, y, w, h = map(float, parts)
                        boxes.append([int(class_id), x, y, w, h])
                    gt_dict[frame_id] = boxes
        return gt_dict

    def _load_pred_file(self):
        """
        Tahmin verisini okur. Format: class_id frame_id x y w h
        """
        pred_dict = {}
        if not os.path.exists(self.pred_file):
            return pred_dict
            
        with open(self.pred_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                class_id, frame_id, x, y, w, h = map(float, parts)
                pred_dict.setdefault(int(frame_id), []).append([int(class_id), x, y, w, h])
        return pred_dict

    @staticmethod
    def _compute_iou(box1, box2):
        def to_corners(box):
            x, y, w, h = box
            return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

        b1 = to_corners(box1)
        b2 = to_corners(box2)

        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2])
        yi2 = min(b1[3], b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def evaluate(self, iou_threshold=0.5):
        correct = 0
        total_gt = 0
        total_pred = 0

        for frame_id in sorted(self.gt_dict.keys()):
            gt_boxes = self.gt_dict.get(frame_id, [])
            pred_boxes = self.pred_dict.get(frame_id, [])

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            matched = set()
            for gt in gt_boxes:
                gt_class, *gt_box = gt
                best_iou = 0
                best_idx = -1
                for idx, pred in enumerate(pred_boxes):
                    if idx in matched:
                        continue
                    pred_class, *pred_box = pred
                    if pred_class != gt_class:
                        continue
                    iou = self._compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= iou_threshold:
                    correct += 1
                    matched.add(best_idx)

        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_gt if total_gt > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct / max(total_gt, total_pred) if max(total_gt, total_pred) > 0 else 0

        return {
            "correct": correct,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

class EvaluationTable:
    def __init__(self, predicts_dir: str, annotations_dir: str = "../annotations", iou_threshold: float = 0.5):
        self.predicts_dir = Path(predicts_dir)
        self.annotations_dir = Path(annotations_dir)
        self.iou_threshold = iou_threshold
        self.models = ["yolov10l", "yolo11l", "yolo12l", "rtdetr-l"]
        self.metrics = ["precision", "recall", "accuracy", "f1"]
        self.videos = [f"video{i}.avi" for i in range(1, 12)]
        self.video_zip_mapping = self._get_video_zip_mapping()
        
    def _get_video_zip_mapping(self):
        """Get the mapping between video files and their corresponding ground truth zip files"""
        mapping = {
            "video1.avi": "task_aboda-video1.avi_annotations_2025_06_03_07_36_23_yolo 1.1.zip",
            "video2.avi": "task_aboda-video2.avi_annotations_2025_06_03_07_41_26_yolo 1.1.zip",
            "video3.avi": "task_aboda-video3.avi_annotations_2025_06_03_07_44_13_yolo 1.1.zip",
            "video4.avi": "task_aboda-video4.avi_annotations_2025_06_03_07_45_39_yolo 1.1.zip",
            "video5.avi": "task_aboda-video5.avi_annotations_2025_06_03_08_07_34_yolo 1.1.zip",
            "video6.avi": "task_aboda-video6.avi_annotations_2025_06_03_08_18_13_yolo 1.1.zip",
            "video7.avi": "task_aboda-video7.avi_annotations_2025_06_03_10_58_27_yolo 1.1.zip",
            "video8.avi": "task_aboda-video8.avi_annotations_2025_06_03_11_08_35_yolo 1.1.zip",
            "video9.avi": "task_aboda-video9.avi_annotations_2025_06_03_11_20_19_yolo 1.1.zip",
            "video10.avi": "task_aboda-video10.avi_annotations_2025_06_03_11_23_25_yolo 1.1.zip",
            "video11.avi": "task_aboda-video11.avi_annotations_2025_06_03_11_28_30_yolo 1.1.zip"
        }
        return mapping
        
    def collect_results(self):
        """Collect evaluation results for all models and videos"""
        results = {}
        
        for model in self.models:
            model_results = {}
            
            for video in self.videos:
                # Get the corresponding ground truth zip file
                if video not in self.video_zip_mapping:
                    print(f"No ground truth mapping found for {video}")
                    model_results[video] = {
                        "precision": 0, "recall": 0, "accuracy": 0, "f1": 0
                    }
                    continue
                
                gt_zip_file = self.annotations_dir / self.video_zip_mapping[video]
                
                if not gt_zip_file.exists():
                    print(f"Ground truth file not found: {gt_zip_file}")
                    model_results[video] = {
                        "precision": 0, "recall": 0, "accuracy": 0, "f1": 0
                    }
                    continue
                
                # Construct the prediction file path
                video_name = video.replace('.avi', '')
                pred_file = self.predicts_dir / model / f"{video_name}.txt"
                
                if pred_file.exists():
                    try:
                        handler = AnnotationHandler(
                            pred_file=str(pred_file),
                            gt_zip_file=str(gt_zip_file)
                        )
                        metrics = handler.evaluate(iou_threshold=self.iou_threshold)
                        model_results[video] = metrics
                    except Exception as e:
                        print(f"Error evaluating {model}/{video}: {e}")
                        model_results[video] = {
                            "precision": 0, "recall": 0, "accuracy": 0, "f1": 0
                        }
                else:
                    print(f"Prediction file not found: {pred_file}")
                    model_results[video] = {
                        "precision": 0, "recall": 0, "accuracy": 0, "f1": 0
                    }
            
            results[model] = model_results
        
        return results
    
    def create_table(self, results):
        """Create pandas DataFrame table from results"""
        table_data = []
        
        for model in self.models:
            for metric in self.metrics:
                row = {"Model": model, "Metric": metric}
                
                for video in self.videos:
                    if video in results[model]:
                        value = results[model][video][metric]
                        row[video] = f"{value:.3f}"
                    else:
                        row[video] = "0.000"
                
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def save_table(self, df, output_file: str = "evaluation_results.csv"):
        """Save table to CSV file"""
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    def print_table(self, df):
        """Print formatted table to console"""
        print("\n" + "="*150)
        print("EVALUATION RESULTS TABLE")
        print("="*150)
        
        # Set pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 8)
        
        print(df.to_string(index=False))
        print("="*150)
        
        # Calculate and print averages
        print("\nAVERAGE RESULTS BY MODEL:")
        print("-"*50)
        
        for model in self.models:
            model_data = df[df['Model'] == model]
            print(f"\n{model.upper()}:")
            
            for metric in self.metrics:
                metric_row = model_data[model_data['Metric'] == metric]
                if not metric_row.empty:
                    video_cols = [col for col in df.columns if col.startswith('video')]
                    values = []
                    for col in video_cols:
                        try:
                            values.append(float(metric_row[col].iloc[0]))
                        except:
                            values.append(0.0)
                    avg = sum(values) / len(values) if values else 0
                    print(f"  {metric:10}: {avg:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models against ground truth in table format.")
    parser.add_argument("--predicts-dir", required=True, help="Predictions directory containing model subdirectories")
    parser.add_argument("--annotations-dir", default="../annotations", help="Annotations directory containing ground truth zip files")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--output-csv", default="evaluation_results.csv", help="Output CSV file name")
    
    args = parser.parse_args()

    evaluator = EvaluationTable(
        predicts_dir=args.predicts_dir,
        annotations_dir=args.annotations_dir,
        iou_threshold=args.iou_threshold
    )
    
    print("Collecting evaluation results...")
    results = evaluator.collect_results()
    
    print("Creating evaluation table...")
    df = evaluator.create_table(results)
    
    evaluator.print_table(df)
    evaluator.save_table(df, args.output_csv)

if __name__ == "__main__":
    main()

