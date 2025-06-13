# PowerShell script to run annotations evaluation for all models

# Run the evaluation with all models and videos
python ../src/annotations.py --predicts-dir ../predictions --annotations-dir ../annotations --output-csv evaluation_results.csv

Write-Host "Evaluation completed. Results saved to evaluation_results.csv" 