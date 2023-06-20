#!/bin/bash
#
## BEGIN SBATCH directives
#SBATCH --job-name=OTGNN
#SBATCH --output=res_seq9.txt
#
#SBATCH --ntasks=1
#SBATCH --time=35:00:00
#SBATCH --mem=15GB
#SBATCH --partition=cpu_shared
#SBATCH --account=oatmil
##SBATCH --mail-type=ALL
##SBATCH --mail-user=sonia.mazelet@polytechnique.edu
## END SBATCH directives

## To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.11

## Execution
python run.py -dropout 0.8 -model LTFGW_MLP_dropout_relu -number_of_seeds 10 -nepochs 1500 -dataset cornell_directed