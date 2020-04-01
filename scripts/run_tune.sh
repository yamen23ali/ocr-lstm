#!/bin/bash

# sh scripts/run_tune.sh "GT4HistOCR/corpus" "RefCorpus-ENHG-Incunabula" lr
# sh scripts/run_tune.sh "/home/space/datasets/text/GT4HistOCR" "RefCorpus-ENHG-Incunabula" lr

python src/tune.py --BASE_DIR=$1 --DATA_SET_NAME=$2 --PARAM_NAME=$3 

