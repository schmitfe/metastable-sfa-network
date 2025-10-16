#!/bin/bash
#SBATCH --job-name=NEST_SFA_WM
#SBATCH --output=log/nest_sfa_wm_%A_%a.out
#SBATCH --error=err/nest_sfa_wm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --partition=smp,interactive,gpu
#SBATCH --time=1:00:00
#SBATCH --array=0-300

# Move to working directory
cd /rhomes/fschmi69/EI_Cluster_SFA/Simulate_NEST

# Load modules and activate environment
module load python3/anaconda3-2024.06-py3.12
conda activate NEST

# Make Python respect allocated CPUs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Export environment variables for the simulation
export attn_factor=0.075
export check_win_end_ms=800
export check_win_start_ms=600
export I_stim_E_fac=0.4
export ITI_ms_max=2000
export ITI_ms_min=1000
export jep=6.6
export N_trials=200
export p_ctrl=0.5
export p_ctrl_pre=0.0
export q_stc=0.2
export stim_dur_ms=200

# Optional: include array index to separate outputs
export output_path="WM_DATA/"

# Run the simulation (pass array ID if script needs it via env SLURM_ARRAY_TASK_ID)
python SFA_WM.py
