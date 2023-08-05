####################
#
# Example Job for HTCondor
#
####################

#---------------------------------------------
# Name your batch so it's easy to distinguish in condor_q.
JobBatchName = "Pretrain-Architecture-and-Eval"

# --------------------------------------------
# Executable: Choose cu version depends on docker_image
executable   = /mnt/fast/nobackup/users/nt00601/miniconda3/envs/cu118_py311/bin/python
# executable   = /mnt/fast/nobackup/users/nt00601/miniconda3/envs/cu117_py311/bin/python

# ---------------------------------------------------
# Universe (vanilla, docker): Choose CUDADriverVersion depends on what's shown on condor_status 
# see https://docs.pages.surrey.ac.uk/research_computing/condor/tips.html#cuda-requirements
universe     = docker
docker_image = nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
#docker_image = nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
#docker_image = nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# -------------------------------------------------
# Event, out and error logs
error  = c$(cluster).p$(process).error

# --------------------------------------
# GPU, Storage and CUDA Requirements for the Job 
# All of these requirements say: We are using rtx2080 or rtx3090
requirements = (CUDAGlobalMemoryMb > 10000) && (CUDAGlobalMemoryMb <  75000) && (CUDACapability > 7) && \
               (HasWeka)

# Resources
request_GPUs 	= 4
+GPUMem 	= 10000
request_CPUs   	= 1
request_memory 	= 15G

#This job will complete in less than 1 hour
+JobRunTime = 4

#This job can checkpoint
+CanCheckpoint = true

# Request for guaranteed run time(measured in s to match epoch runtime). 0 mean job is happy to checkpoint and move at any time.
# This lets Condor remove our job ASAP if a machine needs rebooting. Useful when we can checkpoint and restore
MaxJobRetirementTime = 0

# -----------------------------------
arguments = $(script) -s veri -t veri -a $(a) --no-pretrained --root $(root) --height 224 --width 224 --optim sgd --lr 0.5 --weight-decay 2e-05 --lr-scheduler sequential --label-smooth --max-epoch 600 --train-batch-size 128 --test-batch-size 100 --save-dir $(save)

root = /mnt/fast/nobackup/users/nt00601/content
save = /mnt/fast/nobackup/users/nt00601/CourseWork/logs/Architectures/Finetune
script = /mnt/fast/nobackup/users/nt00601/CourseWork/main.py

queue 1 a in resnet50, resnet50_fc512, resnet101, resnet50_fc512, resnet101_fc512, resnet152_fc512
