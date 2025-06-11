#!/bin/bash
python test.py --videos-dir ABODA --results-dir results --model yolov10l
python test.py --videos-dir ABODA --results-dir results --model rtdetr-l
python test.py --videos-dir ABODA --results-dir results --model yolo12l
python test.py --videos-dir ABODA --results-dir results --model yolo11l
