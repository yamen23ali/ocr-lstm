#!/bin/bash

# sh scripts/run.sh "GT4HistOCR/corpus" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" 10 500 20 300 1 0.01 0.9 2 10 20 0.2
# qsub -V -cwd -j y -l cuda=1 scripts/run.sh "/home/pml_07/ocr_shared/dataset/" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" 10 500 20 1300 1 0.05 0.9 2 0.3 300 0.3

cd ~/pml

python src/train.py --BASE_DIR=$1 --DATA_SET_NAME=$2 --HOLDOUT_BOOK=$3 --FRAME_SIZE=$4 \
--MAX_IMAGE_WIDTH=$5 --MAX_IMAGE_HEIGHT=$6 --HIDDEN_LAYER_SIZE=$7 \
--HIDDEN_LAYERS_NUM=$8 --LEARNING_RATE=$9 --MOMENTUM="${10}" --EPOCHS="${11}" \
--CLIPPING_VALUE="${12}" --BATCH_SIZE="${13}" --DROPOUT_RATIO="${14}"

