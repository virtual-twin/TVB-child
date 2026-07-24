#!/bin/bash
#SBATCH --job-name=./results/EI_Tuning_20240917_130608
#SBATCH --output=./results/EI_Tuning_20240917_130608/logs/subject_fitting_%A_%a.out
#SBATCH --error=./results/EI_Tuning_20240917_130608/logs/subject_fitting_%A_%a.err
#SBATCH --array=1-49%50
#SBATCH --time=12:00:00
#SBATCH --mem=16G

# Activate Conda env
# conda init
# conda activate EI_TVBC

# Go to directory
cd /data/cephfs-1/home/users/pillem_c/work/EI_TVBC/ei_tuning_tvbc

# Get the subject ID from the subjects.txt file
subject=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./results/EI_Tuning_20240917_130608/subjects.txt)

# Run the Python script for the current subject
conda run -n EI_TVBC python process_subject.py --sc_path ./test_data/SCs --fc_path ./test_data/FCs/FC_batch1 --experiment_path ./results/EI_Tuning_20240917_130608 --subject "${subject}"
