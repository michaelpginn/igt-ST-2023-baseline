#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=4         # Number of requested cores
#SBATCH --mem=64G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=train_igt.%j.out      # Output file name
#SBATCH --error=train_igt.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
source ~/AutoIGT/bin/activate
cd "~/projects/rrg-msilfver/mginn/Automatic-IGT-Glossing/src"
python token_class_model.py train --lang arp