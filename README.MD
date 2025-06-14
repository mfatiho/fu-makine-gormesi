### **A Spatiotemporal Mask-Based Approach for Abandoned Object Detection in Video Surveillance**

**Abstract**

This study presents an innovative spatiotemporal approach for abandoned object detection (AOD) in video surveillance systems. To overcome the challenges faced by traditional owner-based tracking methods in complex and crowded scenes, a dynamic mapping technique called "Person Mask" has been developed. This technique maintains a short-term memory of the presence and movement of people in the video scene. The status of an object being abandoned is determined by its stillness and its interaction with this dynamic person mask. To evaluate the effectiveness of our method, four different object detection models, including YOLOv10L, YOLOv11L, YOLOv12L, and RT-DETR-L, were tested on the public ABODA dataset [1]. The experimental results demonstrated the feasibility of the proposed approach, and particularly, the RT-DETR-L model showed superior performance compared to other models with 40.9% Precision and a 16.9% F1-Score. The source code for the project is available on GitHub [2].

**Keywords:** Abandoned Object Detection, Video Surveillance, Computer Vision, Person Mask, Deep Learning, Object Tracking, ABODA.

---

### **Experiments and Setup**

#### **Dataset**
The experiments were conducted on the **ABODA (Abandoned Object Dataset)** [1], a public dataset specifically created for abandoned object detection. This dataset consists of 11 video sequences containing various real-world scenarios that are challenging for AOD systems, such as crowded scenes, significant lighting changes, night scenes, and indoor/outdoor environments. Ground truth data for each video is provided in YOLO format.

#### **Evaluated Models**
To measure the system's performance with different detection backbones, four models were tested:
-   **YOLOv10L**
-   **YOLOv11L**
-   **YOLOv12L**
-   **RT-DETR-L**

#### **Evaluation Metrics**
Standard metrics were used to measure model performance:
-   **Precision:** `TP / (TP + FP)` - How many of the detected abandoned objects are correct.
-   **Recall:** `TP / (TP + FN)` - How many of the actual abandoned objects were detected.
-   **Accuracy:** `TP / (TP + FP + FN)` - The accuracy within the total detections and ground truth.
-   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)` - The harmonic mean of Precision and Recall.

---

### **Results and Discussion**

The following table summarizes the performance of the four models on the 11 videos.

**Table 1: Performance Comparison of Models on Videos**
| Model    | Metric    | video1 | video2 | video3 | video4 | video5 | video6 | video7 | video8 | video9 | video10 | video11 | **Average** |
|:---------|:----------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:--------|:--------|:-------------|
| YOLOv10L | precision | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 0.000  | 1.000   | 0.000   | **0.182**    |
|          | recall    | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.245  | 0.000  | 0.227   | 0.000   | **0.043**    |
|          | f1        | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.393  | 0.000  | 0.371   | 0.000   | **0.069**    |
| YOLOv11L | precision | 1.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 0.000  | 1.000   | 0.000   | **0.273**    |
|          | recall    | 0.219  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.245  | 0.000  | 0.227   | 0.000   | **0.063**    |
|          | f1        | 0.360  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.393  | 0.000  | 0.371   | 0.000   | **0.102**    |
| YOLOv12L | precision | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
|          | recall    | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
|          | f1        | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000   | 0.000   | **0.000**    |
| RT-DETR-L| precision | 1.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 1.000  | 1.000  | 1.000  | 0.500   | 0.000   | **0.409**    |
|          | recall    | 0.218  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.212  | 0.246  | 0.282  | 0.228   | 0.000   | **0.108**    |
|          | f1        | 0.358  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.350  | 0.394  | 0.441  | 0.314   | 0.000   | **0.169**    |

**Discussion:**
The results clearly show that the **RT-DETR-L** model performed superiorly in this task compared to the other tested models. It provided the most balanced results, with an average **Precision of 40.9%** and an **F1-Score of 16.9%**. The fact that the YOLOv12L model failed to make any successful detections in any of the videos suggests that the model is not suitable for this specific dataset and task or that it requires additional fine-tuning.

The success of RT-DETR-L may stem from its Transformer-based architecture's ability to track objects more consistently. This allows our stationarity and mask intersection analyses to work more stably. The overall low metrics indicate the difficulty of the scenarios in the ABODA dataset (e.g., low light, occlusion, small objects). The failure of any model to succeed in some videos (e.g., video2, video3) shows that these videos contain particularly challenging scenarios.

---

### **Conclusion**

In this study, a spatiotemporal "Person Mask" approach was presented as an alternative to traditional methods for abandoned object detection. The developed algorithm was tested with different deep learning models, and RT-DETR-L was observed to give the most promising results. The method holds the potential for a flexible and effective solution, especially in scenarios where owner tracking is difficult.

---

### **References**

[1] ABODA: Abandoned Object Dataset. GitHub. [https://github.com/kevinlin311tw/ABODA](https://github.com/kevinlin311tw/ABODA)
