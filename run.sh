#!/bin/bash

#SBATCH -p edu-biai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --thread-spec=16
#SBATCH --gres=gpu:0

python3.10 play.py

