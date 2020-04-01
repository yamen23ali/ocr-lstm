import sys

import os
import argparse
import base64

from ocr_utils import *
from train_apply import train_apply
import hashlib

#============= Get and Parse Arguments

parser=argparse.ArgumentParser()

parser.add_argument('--BASE_DIR', help='Base Directory Of The Dataset')
parser.add_argument('--DATA_SET_NAME', help='Data Set Name')
parser.add_argument('--HOLDOUT_BOOK', help='The Book we want to use as holdout')
parser.add_argument('--FRAME_SIZE', help='Text line image frame size')
parser.add_argument('--MAX_IMAGE_WIDTH', help='Max width for the text line image')
parser.add_argument('--MAX_IMAGE_HEIGHT', help='Max height for the text line image')
parser.add_argument('--HIDDEN_LAYER_SIZE', help='Hidden Layer Size')
parser.add_argument('--HIDDEN_LAYERS_NUM', help='Hidden Layers Number')
parser.add_argument('--LEARNING_RATE', help='Learning Rate')
parser.add_argument('--MOMENTUM', help='Learning Rate Momentum')
parser.add_argument('--EPOCHS', help='Number of epochs')
parser.add_argument('--CLIPPING_VALUE', help='Number of epochs')
parser.add_argument('--BATCH_SIZE', help='Batch Size')
parser.add_argument('--DROPOUT_RATIO', help='Dropout Ratio')

args=parser.parse_args()

# ========= Create model name
print(f'Model parameters {args}')

BASE_DIR = args.BASE_DIR
DATA_SET_NAME = args.DATA_SET_NAME
HOLDOUT_BOOK = args.HOLDOUT_BOOK

args_bytes = str(args).encode('utf-8')
args_hash= hashlib.sha1(args_bytes).hexdigest()
MODEL_NAME= f'model_{args_hash}'

print(f'Model Name : {MODEL_NAME}')

#========= Hyper parameters 

# Image parameters
FRAME_SIZE = int(args.FRAME_SIZE)
MAX_IMAGE_WIDTH = int(args.MAX_IMAGE_WIDTH)
MAX_IMAGE_HEIGHT = int(args.MAX_IMAGE_HEIGHT)

# NN parameters
HIDDEN_LAYER_SIZE = int(args.HIDDEN_LAYER_SIZE)
HIDDEN_LAYERS_NUM = int(args.HIDDEN_LAYERS_NUM) # Number of LSTM cells to stack

# Training parameters
LEARNING_RATE = float(args.LEARNING_RATE)
MOMENTUM = float(args.MOMENTUM)
EPOCHS = int(args.EPOCHS)
CLIPPING_VALUE = float(args.CLIPPING_VALUE)
BATCH_SIZE = int(args.BATCH_SIZE)
DROPOUT_RATIO= float(args.DROPOUT_RATIO)

_names = ['Training', 'Test', 'Holdout']

train_pred, test_pred, holdout_pred = train_apply(MODEL_NAME, base_dir=BASE_DIR, dataset_name=DATA_SET_NAME, 
                                                  holdout_book=HOLDOUT_BOOK, frame_size=FRAME_SIZE,
                                                  max_image_height=MAX_IMAGE_HEIGHT, max_image_width=MAX_IMAGE_WIDTH,
                                                  hidden_layer_size=HIDDEN_LAYER_SIZE, hidden_layer_num=HIDDEN_LAYERS_NUM, 
                                                  learning_rate=LEARNING_RATE, momentum=MOMENTUM, epochs=EPOCHS,
                                                  clipping_value=CLIPPING_VALUE, batch_size=BATCH_SIZE, 
                                                  dropout_ratio=DROPOUT_RATIO)

for predictions, name in zip([train_pred, test_pred, holdout_pred], _names):

  print( f'============== Infering {name}')
  print_predicted_text(predictions)
  
  error = get_prediction_error(predictions)

  print(f'Error {error} \n')
