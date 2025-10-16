#!/bin/bash
#SBATCH --job-name=NEST_SFA
#SBATCH --output=log/nest_sfa_%A_%a.out
#SBATCH --error=err/nest_sfa_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --partition=smp,interactive,gpu
#SBATCH --time=1:00:00
#SBATCH --array=0-300

# Move to working directory
cd /rhomes/fschmi69/EI_Cluster_SFA/Source/Simulate_NEST

# Load modules and activate environment
module load python3/anaconda3-2024.06-py3.12
conda activate NEST

# Export environment variable (optional: include array index to separate outputs)
export output_path=ModulateFig1_100_15/

# Run the simulation (pass array ID if script needs it)
python SFA_Modulate_SLURM.py
