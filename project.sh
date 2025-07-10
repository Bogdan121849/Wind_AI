#!/bin/bash
#SBATCH --time=03:20:00         
#SBATCH --partition=gpu         
#SBATCH --gres=gpu:1            
#SBATCH --mem=32000              

# Load required modules
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

# (Optional) Activate a virtual environment if needed
source /home2/s5549329/windAI_rug/venv/bin/activate

python --version

# Run your Python project
python /home2/s5549329/windAI_rug/WindAi/deep_learning/preprocessing/preprocessing_dl_data_region.py

deactivate