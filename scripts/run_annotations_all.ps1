# PowerShell version of run_annotations_all.sh

function Get-ZipFile {
    param([int]$VideoNum)
    switch ($VideoNum) {
        1  { '../annotations/task_aboda-video1.avi_annotations_2025_06_03_07_36_23_yolo 1.1.zip' }
        2  { '../annotations/task_aboda-video2.avi_annotations_2025_06_03_07_41_26_yolo 1.1.zip' }
        3  { '../annotations/task_aboda-video3.avi_annotations_2025_06_03_07_44_13_yolo 1.1.zip' }
        4  { '../annotations/task_aboda-video4.avi_annotations_2025_06_03_07_45_39_yolo 1.1.zip' }
        5  { '../annotations/task_aboda-video5.avi_annotations_2025_06_03_08_07_34_yolo 1.1.zip' }
        6  { '../annotations/task_aboda-video6.avi_annotations_2025_06_03_08_18_13_yolo 1.1.zip' }
        7  { '../annotations/task_aboda-video7.avi_annotations_2025_06_03_10_58_27_yolo 1.1.zip' }
        8  { '../annotations/task_aboda-video8.avi_annotations_2025_06_03_11_08_35_yolo 1.1.zip' }
        9  { '../annotations/task_aboda-video9.avi_annotations_2025_06_03_11_20_19_yolo 1.1.zip' }
        10 { '../annotations/task_aboda-video10.avi_annotations_2025_06_03_11_23_25_yolo 1.1.zip' }
        11 { '../annotations/task_aboda-video11.avi_annotations_2025_06_03_11_28_30_yolo 1.1.zip' }
        default { '' }
    }
}

$resultsDirs = Get-ChildItem -Directory ../results
foreach ($detdir in $resultsDirs) {
    $predFiles = Get-ChildItem -Path $detdir.FullName -Filter 'video*.txt'
    foreach ($predFile in $predFiles) {
        if ($predFile.Name -match 'video(\d+)\.txt') {
            $vnum = [int]$Matches[1]
            $zipfile = Get-ZipFile $vnum
            if ($zipfile -and (Test-Path $predFile.FullName)) {
                python ../src/annotations.py --pred-file $predFile.FullName --gt-zip-file $zipfile
            }
        }
    }
} 