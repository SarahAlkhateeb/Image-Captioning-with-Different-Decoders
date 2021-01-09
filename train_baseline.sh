#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --output=/datainbackup/2020-DAT450-DIT245/ronkkoj/group15/DIT245-image-captioning/slurm-%j.out
#SBATCH --error=/datainbackup/2020-DAT450-DIT245/ronkkoj/group15/DIT245-image-captioning/slurm-%j.error
#SBATCH --chdir=/datainbackup/2020-DAT450-DIT245/ronkkoj/group15/DIT245-image-captioning # Working directory
source /opt/local/anaconda3/etc/profile.d/conda.sh ; conda activate /datainbackup/home/2019/ronkkoj/.conda/envs/dit245_group15
python /datainbackup/2020-DAT450-DIT245/ronkkoj/group15/DIT245-image-captioning/train.py baseline --model baseline --max_caption_length 50 --batch_size 32 --epochs 1 --workers 32
