#!/bin/bash 
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --output=log-cpu.out

hostname # displays the node name in which the script is beeing run 
nvidia-smi # displays information about the GPU usage of the node in which the script is being run

# quando é preciso resetar o ambiente completamente sem apagá-lo, assim os pacotes serão reinstalados nas versões desejadas
#source deactivate leo_env
#conda remove --name leo_env --all

# tenta criar o ambiente
if conda env update --file environment.yml; then # updates existing environment or creates a new one from the environment.yml file (SEM  FLAG --FILE, ELE JÁ VAI PROCURAR POR ESSE NOME QUE É DEFAULT)
    echo "Environment created successfully."
else
    echo "Environment creation failed. Aborting..."
    exit 1
fi

source activate leo_env  # activates the environment. The name of the environment is the first line in the environment.yml file 
conda activate leo_env
conda info --envs # lists all known conda environments

# Verifica se o ambiente está ativado corretamente
which python
python -c "import sys; print('Active python:', sys.executable)"

# COLOCANDO OCAMINHO TODO PARA GARANTIR QUE O PYTHON UTILIZADO SEJA O DO MEU AMBIENTE
srun /apps/conda/leonardo.sousa/.envs/leo_env/bin/torchrun \
  --rdzv_endpoint=localhost:29400 \
  main_save_attributes_volumes.py