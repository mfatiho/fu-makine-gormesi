# PowerShell version of run_test_all.sh

python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolov10l --display-video
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model rtdetr-l --display-video
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolo12l --display-video
python ../src/test.py --videos-dir ../ABODA --results-dir ../results --model yolo11l --display-video