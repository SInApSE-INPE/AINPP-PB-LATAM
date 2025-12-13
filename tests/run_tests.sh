#!/bin/bash
# Initialize modules
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

# Load Anaconda
echo "Loading Anaconda..."
module load anaconda3/2024.10
# Initialize conda for shell interaction
eval "$(/scratch/app/anaconda3/2024.10/bin/conda shell.bash hook)"

# Activate Environment
echo "Activating Environment..."
conda activate /scratch/ideeps/adriano.almeida/sinapse || echo "Conda activate returned $?"

# Verify Python
which python
python --version
pip list | grep torch

# Run Tests
echo "---------------------------------------------------"
echo "Running Unit Tests (test_metrics.py)"
python /prj/ideeps/adriano.almeida/benchmark/tests/test_metrics.py
EXIT_CODE_1=$?
echo "Exit Code: $EXIT_CODE_1"

echo "---------------------------------------------------"
echo "Running Integration Tests (test_evaluation_pipeline.py)"
python /prj/ideeps/adriano.almeida/benchmark/tests/test_evaluation_pipeline.py
EXIT_CODE_2=$?
echo "Exit Code: $EXIT_CODE_2"

if [ $EXIT_CODE_1 -eq 0 ] && [ $EXIT_CODE_2 -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "TESTS FAILED"
    exit 1
fi
