#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p general
#SBATCH -G a100:1
#SBATCH -t 0-12:00:00
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE


module load mamba/latest

source activate NLP_HW3

bash test_rap.sh