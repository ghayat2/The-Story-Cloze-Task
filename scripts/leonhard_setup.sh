#!/usr/bin/env bash
ssh $1@euler.ethz.ch
# Run in ssh session:
cd /cluster/home/$1
# Loads GCC and Python with Tensorflow included

module load python_gpu/3.7.1
pip3 install --user -r requirements.txt
bsub -n 4 -W 4:00 -R "rusage[mem=4096,ngpus_excl_p=1]" "python ./train.py"

# To download all model files:
scp -r lhidde@login.leonhard.ethz.ch:/cluster/home/lhidde/nlu-project1/runs/{TIMESTAMP} runs

scp -r lhidde@login.leonhard.ethz.ch:/cluster/home/lhidde/nlu-project1/runs/1555572213/checkpoints/model-30000* runs
scp lhidde@login.leonhard.ethz.ch:/cluster/home/lhidde/nlu-project1/runs/1555572213/checkpoints/checkpoint runs
