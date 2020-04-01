#!/bin/bash

#  sh scripts/run_load_and_evaluate.sh "GT4HistOCR/corpus" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" "model_eb3b7e9eeda52a2a7d7e29b383261131b12312cf_dict.pth" 10 800 2 960 85 2
# qsub -V -cwd -j y -l cuda=1 scripts/run_load_and_evaluate.sh "/home/pml_07/ocr_shared/dataset/" "RefCorpus-ENHG-Incunabula" "1499-CronicaCoellen" "model_eb3b7e9eeda52a2a7d7e29b383261131b12312cf_dict.pth" 10 800 2 960 85 2

cd ~/pml

python src/load_and_evaluate.py --BASE_DIR=$1 --DATA_SET_NAME=$2 --HOLDOUT_BOOK=$3 --MODEL_NAME=$4 --FRAME_SIZE=$5 --HIDDEN_LAYER_SIZE=$6 \
--HIDDEN_LAYERS_NUM=$7 --MAX_IMAGE_WIDTH=$8 --MAX_IMAGE_HEIGHT=$9 --BATCH_SIZE="${10}"

