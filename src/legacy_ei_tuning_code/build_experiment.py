# Script to build a SLURM array job to fit multiple subjects
import os
import re
import shutil
from datetime import datetime

def extract_subject_id(folder_name):
    # Assuming the subject ID is a number at the end of the folder name
    # Modify this regex pattern if your folder naming convention is different
    match = re.search(r'\d+$', folder_name)
    return match.group() if match else None

def generate_slurm_script(subjects, experiment_folder, sc_path, fc_path):
    script = f"""#!/bin/bash
#SBATCH --job-name={experiment_folder}
#SBATCH --output={experiment_folder}/logs/subject_fitting_%A_%a.out
#SBATCH --error={experiment_folder}/logs/subject_fitting_%A_%a.err
#SBATCH --array=1-{len(subjects)}%50
#SBATCH --time=12:00:00
#SBATCH --mem=16G

# Activate Conda env
# conda init
# conda activate EI_TVBC

# Go to directory
cd /data/cephfs-1/home/users/pillem_c/work/EI_TVBC/ei_tuning_tvbc

# Get the subject ID from the subjects.txt file
subject=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {experiment_folder}/subjects.txt)

# Run the Python script for the current subject
conda run -n EI_TVBC python process_subject.py --sc_path {sc_path} --fc_path {fc_path} --experiment_path {experiment_folder} --subject "${{subject}}"
"""
    return script

def main():
    # Create experiment folder with current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"./results/EI_Tuning_{current_datetime}"
    os.makedirs(experiment_folder)
    print(f"Created experiment folder: {experiment_folder}")

    # Create logs folder inside the experiment folder
    os.makedirs(os.path.join(experiment_folder, "logs"))
    # Create fitting results folder inside the experiment folder
    os.makedirs(os.path.join(experiment_folder, "fits"))


    # Copy fitting script so we know the exact version used in the future
    source_file = "./process_subject.py"
    shutil.copy(source_file, experiment_folder)

    # Get subdirectories for SC and FC
    # sc_path = '/data/cephfs-2/unmirrored/groups/ritter/VirtualChildBrain/SCs'
    # fc_path = '/data/cephfs-2/unmirrored/groups/ritter/VirtualChildBrain/FCs'
    sc_path = './test_data/SCs'
    fc_path = './test_data/FCs/FC_batch1'
    subdirs_SC = [d for d in os.listdir(sc_path) if os.path.isdir(os.path.join(sc_path, d))]
    subdirs_FC = [d for d in os.listdir(fc_path) if os.path.isdir(os.path.join(fc_path, d))]

    # Find the intersection of subdirectories that occur in both SC and FC
    subjects = list(set(subdirs_SC) & set(subdirs_FC))
    subjects.sort()

    # Write subjects to subjects.txt in the experiment folder
    with open(os.path.join(experiment_folder, 'subjects.txt'), 'w') as f:
        for subject in subjects:
            f.write(f"{subject}\n")
    
    # Generate SLURM script
    slurm_script = generate_slurm_script(subjects, experiment_folder, sc_path, fc_path)
    
    # Write SLURM script to file in the experiment folder
    with open(os.path.join(experiment_folder, 'submit_job.sh'), 'w') as f:
        f.write(slurm_script)
    
    print(f"Generated subjects.txt with {len(subjects)} subjects")
    print(f"Generated submit_job.sh in {experiment_folder}")
    print(f"To submit the job, run: sbatch {experiment_folder}/submit_job.sh")

if __name__ == "__main__":
    main()