#!/bin/bash
#
## BEGIN SBATCH directives
#SBATCH --job-name=test_seq
#SBATCH --output=res_seq.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_shared
#SBATCH --account=oatmil
##SBATCH --mail-type=ALL
##SBATCH --mail-user=sonia.mazelet@polytechnique.edu
## END SBATCH directives

## To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.11

## Execution
python run.py -model LTFGW_MLP