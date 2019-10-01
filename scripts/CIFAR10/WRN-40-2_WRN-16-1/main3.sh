#!/bin/sh

export CUDA_HOME=/opt/cuda-9.0.176.1/
source activate pytorch

EXECUTABLE_FILE=/afs/inf.ed.ac.uk/user/s17/s1771851/git/FewShotKnowledgeTransfer/main.py
LOG_DIR=/afs/inf.ed.ac.uk/user/s17/s1771851/logs
PRETRAINED_MODELS_DIR=/disk/scratch/s1771851/Pretrained/
DATASETS_DIR=/disk/scratch/s1771851/Datasets/Pytorch

python ${EXECUTABLE_FILE} \
--dataset CIFAR10 \
--batch_size 128 \
--learning_rate 0.1 \
--n_images_per_class 75 \
--scale_n_iters 1 \
--KD_alpha 0.9 \
--KD_temperature 4 \
--AT_beta 1000 \
--KT_mode KD+AT \
--pretrained_models_path ${PRETRAINED_MODELS_DIR} \
--teacher_architecture WRN-40-2 \
--student_architecture WRN-16-1 \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--save_final_model True \
--save_model_path ${LOG_DIR} \
--seeds 0 1 2 \
--workers 4 \
--use_gpu True