#!/bin/bash

# Map video numbers to annotation zip files
get_zip() {
  case $1 in
    1)  echo "../annotations/task_aboda-video1.avi_annotations_2025_06_03_07_36_23_yolo 1.1.zip" ;;
    2)  echo "../annotations/task_aboda-video2.avi_annotations_2025_06_03_07_41_26_yolo 1.1.zip" ;;
    3)  echo "../annotations/task_aboda-video3.avi_annotations_2025_06_03_07_44_13_yolo 1.1.zip" ;;
    4)  echo "../annotations/task_aboda-video4.avi_annotations_2025_06_03_07_45_39_yolo 1.1.zip" ;;
    5)  echo "../annotations/task_aboda-video5.avi_annotations_2025_06_03_08_07_34_yolo 1.1.zip" ;;
    6)  echo "../annotations/task_aboda-video6.avi_annotations_2025_06_03_08_18_13_yolo 1.1.zip" ;;
    7)  echo "../annotations/task_aboda-video7.avi_annotations_2025_06_03_10_58_27_yolo 1.1.zip" ;;
    8)  echo "../annotations/task_aboda-video8.avi_annotations_2025_06_03_11_08_35_yolo 1.1.zip" ;;
    9)  echo "../annotations/task_aboda-video9.avi_annotations_2025_06_03_11_20_19_yolo 1.1.zip" ;;
    10) echo "../annotations/task_aboda-video10.avi_annotations_2025_06_03_11_23_25_yolo 1.1.zip" ;;
    11) echo "../annotations/task_aboda-video11.avi_annotations_2025_06_03_11_28_30_yolo 1.1.zip" ;;
    *)  echo "" ;;
  esac
}

for detdir in ../results/*/; do
  for predfile in "$detdir"video*.txt; do
    vnum=$(basename "$predfile" | sed -E 's/video([0-9]+)\.txt/\1/')
    zipfile=$(get_zip $vnum)
    if [[ -n "$zipfile" && -f "$predfile" ]]; then
      python ../src/annotations.py --pred-file "$predfile" --gt-zip-file "$zipfile"
    fi
  done
done