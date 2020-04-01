#!/bin/bash

# sh scripts/run_with_cv.sh "GT4HistOCR/corpus" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" 10 500 20 300 1 0.01 0.9 2 10 20 0.2
# qsub -V -cwd -j y -l cuda=1 scripts/run_with_cv.sh "/home/space/datasets/text/GT4HistOCR" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" 10 500 20 300 1 0.01 0.9 2 10 20 0.2

cd ~/pml

python src/train_with_cv.py --BASE_DIR=$1 --DATA_SET_NAME=$2 --HOLDOUT_BOOK=$3 --FRAME_SIZE=$4 \
--MAX_IMAGE_WIDTH=$5 --MAX_IMAGE_HEIGHT=$6 --HIDDEN_LAYER_SIZE=$7 \
--HIDDEN_LAYERS_NUM=$8 --LEARNING_RATE=$9 --MOMENTUM="${10}" --EPOCHS="${11}" \
--CLIPPING_VALUE="${12}" --BATCH_SIZE="${13}" --DROPOUT_RATIO="${14}"

