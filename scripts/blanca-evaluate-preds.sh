#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:v100
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-kann
#SBATCH --out=eval_igt.%j.out      # Output file name
#SBATCH --error=eval_igt.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/AutoIGT/Automatic-IGT-Glossing/src"

#for lang in arp git lez nyb ddo usp ntu
#do
python3 evaluate.py --pred ../src/ntu_output_preds_closed --gold ../../GlossingSTPrivate/splits/Natugu/ntu-dev-track1-uncovered
python3 evaluate.py --pred ../src/ntu_output_preds_open--gold ../../GlossingSTPrivate/splits/Natugu/ntu-dev-track2-uncovered

#done
