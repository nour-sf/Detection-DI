#!/usr/bin/env bash
#! /bin/bash
#PBS -S /bin/bash

##


#! /bin/bash

#!/eiting one node
#SBATCH -N1
#requesting 12 cpus
#SBATCH -n12
#requesting 1 k40 GPUs
#SBATCH --gres=gpu:k40:1
#SBATCH -p GPU

source /share/data/ahmed/cobweb-setup/bash_start
conda activate tf_gpu


## change to your directory! 
DATADIR=/share/data2/visitor2/Direct-Imaging-Project/CNN
cd $DATADIR

python CNN_loop2.py

