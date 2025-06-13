#!/bin/bash
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolov10l
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model rtdetr-l
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolo12l
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolo11l
