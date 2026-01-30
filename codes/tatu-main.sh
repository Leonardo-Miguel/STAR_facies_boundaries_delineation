#!/bin/bash

#SBATCH --job-name=
#SBATCH --nodes=1
#SBATCH --output=log_10c.out
#SBATCH --gpus=1

srun singularity run --nv \
      -B/pgeoprj,/pgeodsv,/pgeo,/tatu,/scr01 \
      /tatu/container_images/go-deep.sif \
      python test_supervised.py
