import os
import zipfile
import argparse

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

        return {
            "correct": correct,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "precision": precision,
            "recall": recall
        }
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO predictions against ground truth.")
    parser.add_argument("--pred-file", required=True, help="Tahmin dosyasının yolu (örneğin: video1.txt)")
    parser.add_argument("--gt-zip-file", required=True, help="Ground truth ZIP dosyasının yolu (örneğin: ground_truth.zip)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU eşiği (varsayılan: 0.5)")
    
    args = parser.parse_args()

    handler = AnnotationHandler(pred_file=args.pred_file, gt_zip_file=args.gt_zip_file)
    metrics = handler.evaluate(iou_threshold=args.iou_threshold)

    print("✅ Doğru Tahmin:", metrics["correct"])
    print("📦 Toplam GT:", metrics["total_gt"])
    print("🔍 Toplam Tahmin:", metrics["total_pred"])
    print("🎯 Precision: {:.2f}".format(metrics["precision"]))
    print("🧠 Recall: {:.2f}".format(metrics["recall"]))

if __name__ == "__main__":
    main()

