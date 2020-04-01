import sys

import os
import torch
import glob
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse
import base64

from ocr_data_loader import *
from ocr_utils import *
from ocr_image_transformations import *
from ocr_model import OCRModel
from torchvision import transforms
from torch import nn
import torchvision
from PIL import Image
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
TRAIN_TEST_SPLIT = .8
CLIPPING_VALUE = float(args.CLIPPING_VALUE)
BATCH_SIZE = int(args.BATCH_SIZE)
DROP_OUT_RATIO= float(args.DROPOUT_RATIO)
K = 5

# ============= Load and prepare Data
transformation = transforms.Compose([
    FixLineImage(),
    ImageThumbnail(MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH),
    transforms.ToTensor(),
    ImageTensorPadding(MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH),
    UnfoldImage(1, FRAME_SIZE, FRAME_SIZE)
])

ocr_cv_dataloader = OCRCVDataLoader(base_dir = BASE_DIR, dataset_name = DATA_SET_NAME,
	transformation=transformation, batch_size=BATCH_SIZE, k=K)

dataset = ocr_cv_dataloader.get_dataset()

# Fixed values ( i.e.: not configurable)
ALPHABET_SIZE = len(dataset.alphabet)
INPUT_DIMENSION = MAX_IMAGE_HEIGHT * FRAME_SIZE

print(f'Max image height {MAX_IMAGE_HEIGHT}')
print(f'Max image width {MAX_IMAGE_WIDTH}')
print(f'ِAlphabet Size {ALPHABET_SIZE}')

#============== Define and train the model
errors = []
for i in range(0,K):
    print("======== Training For Split ", i)
    train_data, test_data = ocr_cv_dataloader.load_data(i)
    
    # Define and train the model
    model = OCRModel(INPUT_DIMENSION, HIDDEN_LAYER_SIZE, HIDDEN_LAYERS_NUM, ALPHABET_SIZE, dropout_ratio=DROP_OUT_RATIO)
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    model.train_model(train_data, optimizer, EPOCHS, CLIPPING_VALUE)
    model.eval()
    
    predictions = model.predict(test_data, dataset.alphabet)
    error = get_prediction_error(predictions)
    errors.append(error)
    
    print(f'Split {i}, model accuracy is {error}')

print("Final model mean error {}".format(np.mean(errors)))
